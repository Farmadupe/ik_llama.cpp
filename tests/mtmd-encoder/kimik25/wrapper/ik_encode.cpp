// Minimal driver for the ik_llama.cpp kimik25 vision tower.
#include "clip.h"
#include "clip-impl.h" // clip_image_f32 internals, for the --ik-image-preprocessing dump
#include "ggml.h"

// STB_IMAGE_STATIC keeps our stbi_* local to this TU so it can't clash with the
// stb implementation already compiled into libmtmd.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#include "nlohmann/json.hpp"
#include "base64.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

// --ik-image-preprocessing dump target (wiped when the flag is set).
static const char * IK_PRE_DIR = "/tmp/kimik25_pre";

static std::string basename_of(const std::string & p) {
    auto s = p.find_last_of('/');
    return s == std::string::npos ? p : p.substr(s + 1);
}

int main(int argc, char ** argv) {
    bool use_cpu      = false;
    bool ik_image_preprocessing = false;
    const char * mmproj = nullptr;
    std::vector<std::string> images;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--cpu") == 0) {
            use_cpu = true;
        } else if (std::strcmp(argv[i], "--ik-image-preprocessing") == 0) {
            ik_image_preprocessing = true;
        } else if (!mmproj) {
            mmproj = argv[i];
        } else {
            images.push_back(argv[i]);
        }
    }
    if (!mmproj || images.empty()) {
        fprintf(stderr, "usage: %s [--cpu] [--ik-image-preprocessing] <mmproj.gguf> <img1> [img2 ...]\n",
                argv[0]);
        return 1;
    }
    if (ik_image_preprocessing) {
        // wipe so stale views from a previous run's image set cannot leak
        // into ref_encode.py's *.pre.bin glob
        std::error_code ec;
        std::filesystem::remove_all(IK_PRE_DIR, ec);
        std::filesystem::create_directories(IK_PRE_DIR, ec);
        if (ec) {
            fprintf(stderr, "failed to create %s\n", IK_PRE_DIR);
            return 1;
        }
    }
    // hardcoded output filename; keeps stdout/argv free for ad-hoc debugging
    const char * out_path = "ik_embeddings.json";

    clip_context_params p{};
    p.use_gpu          = !use_cpu;
    p.verbosity        = GGML_LOG_LEVEL_ERROR;
    p.flash_attn_type  = CLIP_FLASH_ATTN_TYPE_DISABLED;
    p.image_min_tokens = -1;
    p.image_max_tokens = -1;
    p.kq_type          = GGML_TYPE_F32;

    clip_init_result res = clip_init(mmproj, p);
    clip_ctx * ctx = res.ctx_v;
    if (!ctx) {
        fprintf(stderr, "failed to load vision model from %s\n", mmproj);
        return 1;
    }

    const int n_embd    = clip_n_mmproj_embd(ctx);
    // IK_THREADS overrides for debugging (threaded-kernel suspects)
    const int n_threads = std::getenv("IK_THREADS")
        ? std::max(1, atoi(std::getenv("IK_THREADS")))
        : (int) std::max(1u, std::thread::hardware_concurrency());

    nlohmann::json out = nlohmann::json::object();
    for (size_t i = 0; i < images.size(); i++) {
        const std::string & path = images[i];
        int nx, ny, nc;
        unsigned char * data = stbi_load(path.c_str(), &nx, &ny, &nc, 3);
        if (!data) {
            fprintf(stderr, "failed to load image %s\n", path.c_str());
            return 1;
        }

        clip_image_u8 * u8 = clip_image_u8_init();
        clip_build_img_from_pixels(data, nx, ny, u8);
        stbi_image_free(data);

        clip_image_f32_batch * batch = clip_image_f32_batch_init();
        if (!clip_image_preprocess(ctx, u8, batch)) {
            fprintf(stderr, "preprocess failed for %s\n", path.c_str());
            return 1;
        }

        clip_image_f32 * f32      = clip_image_f32_get_img(batch, 0);
        const int        n_tokens = clip_n_output_tokens(ctx, f32);

        // --ik-image-preprocessing: dump the preprocessed (resized+normalized,
        // interleaved RGB row-major) f32 image for parity checks against the HF
        // processor and for ref_encode.py's encoder-only mode.
        if (ik_image_preprocessing) {
            const std::string dpath = std::string(IK_PRE_DIR) + "/" + basename_of(path) + ".pre.bin";
            std::ofstream ds(dpath, std::ios::binary);
            const int32_t dims[2] = {f32->nx, f32->ny};
            ds.write(reinterpret_cast<const char *>(dims), sizeof(dims));
            ds.write(reinterpret_cast<const char *>(f32->buf.data()), f32->buf.size() * sizeof(float));
            fprintf(stderr, "dumped preprocessed %dx%d f32 to %s\n", f32->nx, f32->ny, dpath.c_str());
        }

        std::vector<float> vec((size_t) n_tokens * n_embd);
        if (!clip_image_batch_encode(ctx, n_threads, batch, vec.data())) {
            fprintf(stderr, "encode failed for %s\n", path.c_str());
            return 1;
        }

        const std::string base = basename_of(path);
        // [n_tokens x n_embd]: one base64 string per token row, each the row's
        // n_embd raw little-endian f32 bytes. See tests/mtmd-encoder/README.md.
        nlohmann::json rows = nlohmann::json::array();
        for (int t = 0; t < n_tokens; t++) {
            const float * r = vec.data() + (size_t) t * n_embd;
            rows.push_back(base64::encode(reinterpret_cast<const char *>(r),
                                          (size_t) n_embd * sizeof(float)));
        }
        out[base] = std::move(rows);

        clip_image_f32_batch_free(batch);
        clip_image_u8_free(u8);
    }
    std::ofstream os(out_path);
    os << out.dump();
    os.close();

    clip_free(ctx);
    if (res.ctx_a) clip_free(res.ctx_a);
    return 0;
}
