// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "mtmd.h"
#include "mtmd-helper.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <vector>

//#define MTMD_AUDIO_DEBUG

#define MINIAUDIO_IMPLEMENTATION
#ifndef MTMD_AUDIO_DEBUG
#   define MA_NO_ENCODING
#endif
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize2.h"

#define LOG_INF(...) do { fprintf(stderr, __VA_ARGS__); fflush(stderr); } while (0)
#define LOG_ERR(...) do { fprintf(stderr, __VA_ARGS__); fflush(stderr); } while (0)

size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks * chunks) {
    size_t n_tokens = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        auto chunk = mtmd_input_chunks_get(chunks, i);
        n_tokens += mtmd_input_chunk_get_n_tokens(chunk);
    }
    return n_tokens;
}

llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks * chunks) {
    llama_pos n_pos = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        auto chunk = mtmd_input_chunks_get(chunks, i);
        n_pos += mtmd_input_chunk_get_n_pos(chunk);
    }
    return n_pos;
}

// helper struct to make working with embd batch easier
// note: this will be removed after llama_batch_ext refactoring
struct decode_embd_batch {
    int n_pos_per_embd;
    int n_mmproj_embd;
    std::vector<llama_pos>      pos;
    std::vector<llama_pos>      pos_view; // used by mrope
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch batch;
    decode_embd_batch(float * embd, int32_t n_tokens, int n_pos_per_embd, int n_mmproj_embd) : n_pos_per_embd(n_pos_per_embd), n_mmproj_embd(n_mmproj_embd) {
        pos     .resize(n_tokens * n_pos_per_embd);
        n_seq_id.resize(n_tokens);
        seq_ids .resize(n_tokens + 1);
        logits  .resize(n_tokens);
        seq_id_0.resize(1);
        seq_ids [n_tokens] = nullptr;
        batch = {
            /*n_tokens       =*/ n_tokens,
            /*tokens         =*/ nullptr,
            /*embd           =*/ embd,
            /*pos            =*/ pos.data(),
            /*n_seq_id       =*/ n_seq_id.data(),
            /*seq_id         =*/ seq_ids.data(),
            /*logits         =*/ logits.data(),
        };
    }

    // Assign rope indexes.
    void set_pos(
        mtmd_input_chunk_type chunk_type,
        //Temporal index of first token
        llama_pos pos_0,
        llama_seq_id seq_id,
        int n_tokens,
        ///////////////////
        //batch offsets
        ///////////////////
        //Offset into batch
        int offset = 0,
        // Image dims and offset into image
        int nx = 0,
        int ny = 0,
        int image_offset = 0
    ) {
        if (n_pos_per_embd == 1) {
            // normal
            for (int i = 0; i < n_tokens; i++) {
                batch.pos[offset + i] = pos_0 + i;
            }
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            // M-RoPE for image
            GGML_ASSERT(image_offset + n_tokens <= nx * ny);
            for (int k = 0; k < n_tokens; k++) {
                int y = (image_offset + k) / nx;
                int x = (image_offset + k) % nx;
                int i = offset + k;
                pos[i                     ] = pos_0;
                pos[i + batch.n_tokens    ] = pos_0 + y;
                pos[i + batch.n_tokens * 2] = pos_0 + x;
                pos[i + batch.n_tokens * 3] = 0; // last pos dim is unused
            }
        } else {
            // M-RoPE for text/audio
            for (int i = 0; i < n_tokens; i++) {
                int j = offset + i;
                pos[j                     ] = pos_0 + i;
                pos[j + batch.n_tokens    ] = pos_0 + i;
                pos[j + batch.n_tokens * 2] = pos_0 + i;
                pos[j + batch.n_tokens * 3] = 0; // last pos dim is unused
            }
        }
        seq_id_0[0] = seq_id;
        for (int i = 0; i < n_tokens; i++) {
            batch.n_seq_id[offset + i] = 1;
            batch.seq_id  [offset + i] = seq_id_0.data();
            batch.logits  [offset + i] = false;
        }
    }

    llama_batch get_view(int offset, int n_tokens) {
        llama_pos * pos_ptr;
        pos_view.clear();
        pos_view.reserve(n_tokens * n_pos_per_embd);
        if (n_pos_per_embd > 1) {
            // mrope
            // for example, with layout of src: 1234...1234...1234...1234...
            //       offset 2 will give us dst: 34...34...34...34...
            for (int i = 0; i < n_pos_per_embd; i++) {
                // assume n_tokens is less than or equal to batch.n_tokens
                // batch.n_tokens is number of **total** tokens
                // n_tokens is number of viewed token
                size_t src_idx = i * batch.n_tokens + offset;
                pos_view.insert(pos_view.end(),
                    pos.data() + src_idx,
                    pos.data() + src_idx + n_tokens);
            }
            pos_ptr = pos_view.data();
        } else {
            // normal
            pos_ptr = pos.data() + offset;
        }
        return {
            /*n_tokens       =*/ n_tokens,
            /*tokens         =*/ nullptr,
            /*embd           =*/ batch.embd     + offset * n_mmproj_embd,
            /*pos            =*/ pos_ptr,
            /*n_seq_id       =*/ batch.n_seq_id + offset,
            /*seq_id         =*/ batch.seq_id   + offset,
            /*logits         =*/ batch.logits   + offset,
        };
    }
};

static int32_t mtmd_helper_decode_image_chunk_impl(
        mtmd_context * ctx,
        struct llama_context * lctx,
        const mtmd_input_chunk * chunk,
        float * encoded_embd,
        llama_pos n_past,
        llama_seq_id seq_id,
        int32_t n_batch,
        llama_pos * new_n_past,
        mtmd_helper_eval_batch_callback callback,
        void * callback_user_data) {
    auto chunk_type = mtmd_input_chunk_get_type(chunk);
    const char * name = chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        LOG_ERR("failed to decode chunk: input chunk not of image/audio type\n");
        return -1;
    }

    const llama_model * model = llama_get_model(lctx);
    int n_mmproj_embd = llama_model_n_embd(model);
    int n_pos_per_embd = mtmd_decode_use_mrope(ctx) ? 4 : 1;

    int32_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
    int32_t i_batch = 0;
    int32_t n_img_batches = (n_tokens + n_batch - 1) / n_batch;
    decode_embd_batch batch_embd(encoded_embd, n_tokens, n_pos_per_embd, n_mmproj_embd);

    int nx = 0;
    int ny = 0;
    if (mtmd_decode_use_mrope(ctx) && chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        const auto image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
        if (!image_tokens) {
            LOG_ERR("failed to decode chunk: image tokens are null\n");
            return -1;
        }
        nx = mtmd_image_tokens_get_nx(image_tokens);
        ny = mtmd_image_tokens_get_ny(image_tokens);
    }
    batch_embd.set_pos(chunk_type, n_past, seq_id, n_tokens, 0, nx, ny);

    if (mtmd_decode_use_non_causal(ctx)) {
        llama_set_causal_attn(lctx, false);
        // TODO @ngxson : need to make sure only one image is processed at a time, and n_ubatch must be enough to hold the image
    }

    while (i_batch < n_img_batches) { // split into batches
        int pos_offset = i_batch*n_batch;
        int n_tokens_batch = std::min(n_batch, n_tokens - pos_offset);
        llama_batch batch_embd_view = batch_embd.get_view(pos_offset, n_tokens_batch);

        LOG_INF("decoding %s batch %d/%d, n_tokens_batch = %d\n", name, i_batch+1, n_img_batches, n_tokens_batch);

        int64_t t1 = ggml_time_ms();
        int32_t ret = llama_decode(lctx, batch_embd_view);
        if (ret != 0) {
            LOG_ERR("failed to decode %s\n", name);
            llama_set_causal_attn(lctx, true); // restore causal attn
            return ret;
        }

        LOG_INF("%s decoded (batch %d/%d) in %" PRId64 " ms\n", name, i_batch+1, n_img_batches, ggml_time_ms() - t1);

        if (callback) {
            int32_t callback_ret = callback(callback_user_data, &batch_embd_view);
            if (callback_ret != 0) {
                LOG_ERR("failed to process %s decode callback\n", name);
                llama_set_causal_attn(lctx, true); // restore causal attn
                return callback_ret;
            }
        }

        i_batch++;
    }

    n_past += mtmd_input_chunk_get_n_pos(chunk);
    *new_n_past = n_past;

    if (mtmd_decode_use_non_causal(ctx)) {
        llama_set_causal_attn(lctx, true);
    }
    return 0;
}

// Helper function for decoding an image whose embeddings have already been calculated
int32_t mtmd_helper_decode_image_chunk(
        mtmd_context * ctx,
        struct llama_context * lctx,
        const mtmd_input_chunk * chunk,
        float * encoded_embd,
        llama_pos n_past,
        llama_seq_id seq_id,
        int32_t n_batch,
        llama_pos * new_n_past) {
    return mtmd_helper_decode_image_chunk_impl(
            ctx, lctx, chunk, encoded_embd, n_past, seq_id, n_batch, new_n_past, nullptr, nullptr);
}

int32_t mtmd_helper_eval_chunk_single(mtmd_context * ctx,
        struct llama_context * lctx,
        const mtmd_input_chunk * chunk,
        llama_pos n_past,
        llama_seq_id seq_id,
        int32_t n_batch,
        bool logits_last,
        llama_pos * new_n_past) {
    return mtmd_helper_eval_chunk_single_with_callback(
            ctx, lctx, chunk, n_past, seq_id, n_batch, logits_last, new_n_past, nullptr, nullptr);
}

int32_t mtmd_helper_eval_chunk_single_with_callback(mtmd_context * ctx,
        struct llama_context * lctx,
        const mtmd_input_chunk * chunk,
        llama_pos n_past,
        llama_seq_id seq_id,
        int32_t n_batch,
        bool logits_last,
        llama_pos * new_n_past,
        mtmd_helper_eval_batch_callback callback,
        void * callback_user_data) {
    int32_t ret;
    llama_batch text_batch = llama_batch_init(n_batch, 0, 1);
    auto chunk_type = mtmd_input_chunk_get_type(chunk);

    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        size_t n_tokens;
        const auto tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
        // LOG_INF("decoding text chunk, n_tokens = %zu\n", n_tokens);
        size_t i = 0;
        while (i < n_tokens) { // split into batches
            text_batch.n_tokens = 0; // clear the batch
            for (; i < n_tokens && text_batch.n_tokens < n_batch; i++) {
                int32_t j = text_batch.n_tokens;
                text_batch.token   [j]    = tokens[i];
                text_batch.pos     [j]    = n_past++;
                text_batch.n_seq_id[j]    = 1;
                text_batch.seq_id  [j][0] = seq_id;
                text_batch.logits  [j]    = false;

                text_batch.n_tokens++;
            }
            bool is_last_token = (i == n_tokens);
            if (logits_last && is_last_token) {
                text_batch.logits[text_batch.n_tokens - 1] = true;
            }
            ret = llama_decode(lctx, text_batch);
            if (ret != 0) {
                LOG_ERR("failed to decode text\n");
                llama_batch_free(text_batch);
                return ret;
            }
            if (callback) {
                int32_t callback_ret = callback(callback_user_data, &text_batch);
                if (callback_ret != 0) {
                    LOG_ERR("failed to process text decode callback\n");
                    llama_batch_free(text_batch);
                    return callback_ret;
                }
            }
            *new_n_past += text_batch.n_tokens;
        }

    } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE || chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        const char * name = chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
        int64_t t0 = ggml_time_ms();

        LOG_INF("encoding %s slice...\n", name);

        ret = mtmd_encode_chunk(ctx, chunk);
        if (ret != 0) {
            LOG_ERR("failed to encode %s slice\n", name);
            llama_batch_free(text_batch);
            return ret;
        }

        LOG_INF("%s slice encoded in %" PRId64 " ms\n", name, ggml_time_ms() - t0);

        float * embd = mtmd_get_output_embd(ctx);
        ret = mtmd_helper_decode_image_chunk_impl(
                ctx, lctx, chunk, embd, n_past, seq_id, n_batch, new_n_past, callback, callback_user_data);
        if (ret != 0) {
            LOG_ERR("failed to decode %s\n", name);
            llama_batch_free(text_batch);
            return ret;
        }
    } else {
        GGML_ABORT("chunk type not supported");
    }

    llama_batch_free(text_batch);
    return 0;
}

int32_t mtmd_helper_eval_coalesced(mtmd_context * ctx,
                                   struct llama_context * lctx,
                                   const struct mtmd_helper_coalesce_input * chunks,
                                   size_t n_chunks,
                                   llama_pos n_past,
                                   llama_seq_id seq_id,
                                   int32_t n_batch,
                                   bool logits_last,
                                   llama_pos * new_n_past,
                                   mtmd_helper_eval_batch_callback callback,
                                   void * callback_user_data) {
    const llama_model * model         = llama_get_model(lctx);
    // Row width llama_decode expects in batch.embd - matches what llm_build_inp_embd
    // allocates for lctx.inp_embd (hparams.n_embd). mmproj already produces rows this
    // wide for media chunks. Text rows from llama_input_embeddings may come out narrower
    // (tok_embd->ne[0]) and are zero-padded to n_mmproj_embd by passing this as the stride.
    const int           n_mmproj_embd  = llama_model_n_embd(model);
    const int           n_pos_per_embd = mtmd_decode_use_mrope(ctx) ? 4 : 1;

    if (n_chunks == 0) {
        *new_n_past = n_past;
        return 0;
    }

    // First pass: count totals for position accounting and progress reporting.
    // This pass reifies nothing.
    int32_t   n_total_tokens = 0;
    llama_pos n_pos_advance  = 0;
    for (size_t i = 0; i < n_chunks; i++) {
        const auto & chunk = chunks[i];
        if (chunk.is_text) {
            n_total_tokens += chunk.n_text_tokens;
            n_pos_advance  += chunk.n_text_tokens;
        } else {
            n_total_tokens += (int32_t) mtmd_input_chunk_get_n_tokens(chunk.chunk);
            n_pos_advance  += mtmd_input_chunk_get_n_pos(chunk.chunk);
        }
    }

    if (n_total_tokens == 0) {
        *new_n_past = n_past;
        return 0;
    }

    // Not compatible with gemma3/gemma4
    n_batch = std::min(n_batch, n_total_tokens);
    const int32_t n_calls = (n_total_tokens + n_batch - 1) / n_batch;

    std::vector<float> embd((size_t) n_batch * n_mmproj_embd);
    decode_embd_batch  batch_embd(embd.data(), n_batch, n_pos_per_embd, n_mmproj_embd);

    int32_t   n_filled    = 0;      // rows currently in embd, not yet decoded
    int32_t   i_flush     = 0;
    llama_pos t           = n_past; // position counter, see below

    auto flush = [&]() -> int32_t {
        if (n_filled == 0) {
            return 0;
        }
        llama_batch view = batch_embd.get_view(0, n_filled);

        int32_t ret = llama_decode(lctx, view);
        i_flush++;
        if (ret != 0) {
            LOG_ERR("%s: llama_decode failed on coalesced batch %d/%d\n", __func__, i_flush, n_calls);
            return ret;
        }

        if (callback) {
            int32_t cb_ret = callback(callback_user_data, &view);
            if (cb_ret != 0) {
                LOG_ERR("%s: callback failed\n", __func__);
                return cb_ret;
            }
        }

        n_filled = 0;
        return 0;
    };

    for (size_t i = 0; i < n_chunks; i++) {
        const auto & in = chunks[i];

        if (in.is_text) {
            // A run may straddle batch boundaries; look up each sub-span directly
            // into embd at the fill offset.
            int32_t done = 0;
            while (done < in.n_text_tokens) {
                if (n_filled == n_batch) {
                    int32_t ret = flush();
                    if (ret != 0) {
                        return ret;
                    }
                }
                const int32_t take = std::min(n_batch - n_filled, in.n_text_tokens - done);
                int32_t ret = llama_input_embeddings(lctx, in.text_tokens + done, take,
                                                     embd.data() + (size_t) n_filled * n_mmproj_embd,
                                                     n_mmproj_embd);
                if (ret != 0) {
                    LOG_ERR("%s: llama_input_embeddings failed on text input %zu\n", __func__, i);
                    return ret;
                }
                batch_embd.set_pos(MTMD_INPUT_CHUNK_TYPE_TEXT, t + done, seq_id, take, n_filled);
                n_filled += take;
                done     += take;
            }
            t += in.n_text_tokens;
        } else {
            const auto    type     = mtmd_input_chunk_get_type(in.chunk);
            const int32_t n_tokens = (int32_t) mtmd_input_chunk_get_n_tokens(in.chunk);

            if (type != MTMD_INPUT_CHUNK_TYPE_IMAGE && type != MTMD_INPUT_CHUNK_TYPE_AUDIO) {
                GGML_ABORT("media input must be IMAGE or AUDIO");
            }
            // Audio positions are text-like under M-RoPE (each token = 1 temporal tick).
            const bool image_mrope = type == MTMD_INPUT_CHUNK_TYPE_IMAGE && n_pos_per_embd == 4;
            int32_t nx = 0;
            int32_t ny = 0;
            if (image_mrope) {
                const auto * image_tokens = mtmd_input_chunk_get_tokens_image(in.chunk);
                nx = mtmd_image_tokens_get_nx(image_tokens);
                ny = mtmd_image_tokens_get_ny(image_tokens);
            }

            int32_t ret = mtmd_encode_chunk(ctx, in.chunk);
            if (ret != 0) {
                LOG_ERR("%s: mtmd_encode_chunk failed on media input %zu\n", __func__, i);
                return ret;
            }
            // Valid until the next mtmd_encode_chunk, which is after this chunk has
            // been fully drained into one or more batches below.
            const float * encoded = mtmd_get_output_embd(ctx);

            int32_t done = 0;
            while (done < n_tokens) {
                if (n_filled == n_batch) {
                    int32_t fret = flush();
                    if (fret != 0) {
                        return fret;
                    }
                }
                const int32_t take = std::min(n_batch - n_filled, n_tokens - done);
                std::memcpy(embd.data() + (size_t) n_filled * n_mmproj_embd,
                            encoded + (size_t) done * n_mmproj_embd,
                            (size_t) take * n_mmproj_embd * sizeof(float));
                if (image_mrope) {
                    batch_embd.set_pos(type, t, seq_id, take, n_filled, nx, ny, done);
                } else {
                    batch_embd.set_pos(type, t + done, seq_id, take, n_filled);
                }
                n_filled += take;
                done     += take;
            }
            t += image_mrope ? 1 : n_tokens;
        }
    }

    if (logits_last && n_filled > 0) {
        batch_embd.batch.logits[n_filled - 1] = true;
    }
    {
        int32_t ret = flush();
        if (ret != 0) {
            return ret;
        }
    }

    *new_n_past = n_past + n_pos_advance;
    return 0;
}

int32_t mtmd_helper_eval_chunks(mtmd_context * ctx,
                                struct llama_context * lctx,
                                const mtmd_input_chunks * chunks,
                                llama_pos n_past,
                                llama_seq_id seq_id,
                                int32_t n_batch,
                                bool logits_last,
                                llama_pos * new_n_past) {
    size_t n_chunks = mtmd_input_chunks_size(chunks);
    if (n_chunks == 0) {
        LOG_ERR("no chunks to eval\n");
        return 0;
    }

    for (size_t i = 0; i < n_chunks; i++) {
        bool chunk_logits_last = (i == n_chunks - 1) && logits_last;
        auto chunk = mtmd_input_chunks_get(chunks, i);

        int32_t res = mtmd_helper_eval_chunk_single(ctx, lctx, chunk, n_past, seq_id, n_batch, chunk_logits_last, &n_past);
        if (res != 0) {
            LOG_ERR("failed to eval chunk %zu\n", i);
            return res;
        }
        *new_n_past = n_past;
    }

    return 0;
}

namespace audio_helpers {

static bool is_audio_file(const char * buf, size_t len) {
    if (len < 12) {
        return false;
    }

    // RIFF ref: https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
    // WAV ref: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
    bool is_wav = memcmp(buf, "RIFF", 4) == 0 && memcmp(buf + 8, "WAVE", 4) == 0;
    bool is_mp3 = len >= 3 && (
        memcmp(buf, "ID3", 3) == 0 ||
        // Check for MPEG sync word (simplified check)
        ((unsigned char)buf[0] == 0xFF && ((unsigned char)buf[1] & 0xE0) == 0xE0)
    );
    bool is_flac = memcmp(buf, "fLaC", 4) == 0;

    return is_wav || is_mp3 || is_flac;
}

// returns true if the buffer is a valid audio file
static bool decode_audio_from_buf(const unsigned char * buf_in, size_t len, int target_sampler_rate, std::vector<float> & pcmf32_mono) {
    ma_result result;
    const int channels = 1;
    ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, channels, target_sampler_rate);
    ma_decoder decoder;

    result = ma_decoder_init_memory(buf_in, len, &decoder_config, &decoder);
    if (result != MA_SUCCESS) {
        return false;
    }

    ma_uint64 frame_count;
    ma_uint64 frames_read;
    result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

    pcmf32_mono.resize(frame_count);
    result = ma_decoder_read_pcm_frames(&decoder, pcmf32_mono.data(), frame_count, &frames_read);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

#ifdef MTMD_AUDIO_DEBUG
    // save audio to wav file
    ma_encoder_config config = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 1, target_sampler_rate);
    ma_encoder encoder;
    ma_encoder_init_file("output.wav", &config, &encoder);
    ma_encoder_write_pcm_frames(&encoder, pcmf32_mono.data(), pcmf32_mono.size(), &frames_read);
    ma_encoder_uninit(&encoder);
#endif

    ma_decoder_uninit(&decoder);
    return true;
}

} // namespace audio_helpers

mtmd_bitmap * mtmd_helper_bitmap_init_from_buf(mtmd_context * ctx, const unsigned char * buf, size_t len) {
    if (audio_helpers::is_audio_file((const char *)buf, len)) {
        std::vector<float> pcmf32;
        int bitrate = mtmd_get_audio_bitrate(ctx);
        if (bitrate < 0) {
            LOG_ERR("This model does not support audio input\n");
            return nullptr;
        }
        if (!audio_helpers::decode_audio_from_buf(buf, len, bitrate, pcmf32)) {
            LOG_ERR("Unable to read WAV audio file from buffer\n");
            return nullptr;
        }
        return mtmd_bitmap_init_from_audio(pcmf32.size(), pcmf32.data());
    }

    // otherwise, we assume it's an image
    mtmd_bitmap * result = nullptr;
    {
        int nx, ny, nc;
        auto * data = stbi_load_from_memory(buf, len, &nx, &ny, &nc, 3);
        if (!data) {
            LOG_ERR("%s: failed to decode image bytes\n", __func__);
            return nullptr;
        }
        result = mtmd_bitmap_init(nx, ny, data);
        stbi_image_free(data);
    }
    return result;
}

mtmd_bitmap * mtmd_helper_bitmap_init_from_file(mtmd_context * ctx, const char * fname) {
    std::vector<unsigned char> buf;
    FILE * f = fopen(fname, "rb");
    if (!f) {
        LOG_ERR("Unable to open file %s: %s\n", fname, strerror(errno));
        return nullptr;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf.resize(file_size);

    size_t n_read = fread(buf.data(), 1, file_size, f);
    fclose(f);
    if (n_read != (size_t)file_size) {
        LOG_ERR("Failed to read entire file %s", fname);
        return nullptr;
    }

    return mtmd_helper_bitmap_init_from_buf(ctx, buf.data(), buf.size());
}
