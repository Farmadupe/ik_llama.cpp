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

// One stream input: an owned copy of a text run, or a borrowed media chunk.
struct mtmd_helper_stream_input {
    std::vector<llama_token> text_tokens; // owned; empty for media
    const mtmd_input_chunk * chunk;       // borrowed; nullptr for text
    bool is_text() const { return chunk == nullptr; }
};

struct mtmd_helper_embd_stream {
    mtmd_context         * ctx;
    struct llama_context * lctx;
    std::vector<mtmd_helper_stream_input> inputs; // text owned; media chunks borrowed
    llama_seq_id           seq_id;
    int                    n_mmproj_embd;
    int                    n_pos_per_embd;
    bool                   use_non_causal;

    // cursor
    size_t    i_input    = 0;  // current descriptor
    int32_t   n_done_cur = 0;  // rows of the current descriptor already drained
    llama_pos t          = 0;  // temporal position counter
    int32_t   n_drained  = 0;  // total rows drained, including partial chunks
    int32_t   n_boundary = 0;  // mirror-safe high-water mark
    bool      failed     = false;

    // current media descriptor, encoded lazily on first touch
    std::vector<float>    cur_embd;
    bool                  cur_encoded     = false;
    mtmd_input_chunk_type cur_type        = MTMD_INPUT_CHUNK_TYPE_TEXT;
    int32_t               cur_n_tokens    = 0;
    int32_t               cur_nx          = 0;
    int32_t               cur_ny          = 0;
    bool                  cur_image_mrope = false;
};

mtmd_helper_embd_stream * mtmd_helper_embd_stream_init(
        mtmd_context * ctx,
        struct llama_context * lctx,
        const mtmd_input_chunk * const * chunks,
        size_t n_chunks,
        llama_pos n_past,
        llama_seq_id seq_id) {
    for (size_t i = 0; i < n_chunks; i++) {
        const auto type = mtmd_input_chunk_get_type(chunks[i]);
        if (type != MTMD_INPUT_CHUNK_TYPE_TEXT
            && type != MTMD_INPUT_CHUNK_TYPE_IMAGE
            && type != MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            LOG_ERR("%s: chunk %zu has unsupported type\n", __func__, i);
            return nullptr;
        }
    }

    mtmd_helper_embd_stream * stream = new mtmd_helper_embd_stream();
    stream->ctx    = ctx;
    stream->lctx   = lctx;
    stream->seq_id = seq_id;
    const llama_model * model = llama_get_model(lctx);
    // Row width llama_decode expects in batch.embd - matches what llm_build_inp_embd
    // allocates for lctx.inp_embd (hparams.n_embd). mmproj already produces rows this
    // wide for media chunks. Text rows from llama_input_embeddings may come out narrower
    // (tok_embd->ne[0]) and are zero-padded to n_mmproj_embd by passing this as the stride.
    stream->n_mmproj_embd  = llama_model_n_embd(model);
    stream->n_pos_per_embd = mtmd_decode_use_mrope(ctx) ? 4 : 1;
    stream->use_non_causal = mtmd_decode_use_non_causal(ctx);
    stream->t = n_past;
    stream->inputs.reserve(n_chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        mtmd_helper_stream_input in;
        if (mtmd_input_chunk_get_type(chunks[i]) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens = 0;
            const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunks[i], &n_tokens);
            if (n_tokens == 0) {
                continue; // drop empty text runs so the cursor always rests on real input
            }
            in.text_tokens.assign(tokens, tokens + n_tokens);
            in.chunk = nullptr;
        } else {
            in.chunk = chunks[i];
        }
        stream->inputs.push_back(std::move(in));
    }
    return stream;
}

int32_t mtmd_helper_embd_stream_drain(
        mtmd_helper_embd_stream * stream,
        struct mtmd_decode_embd_batch * dst,
        int32_t row_offset,
        int32_t n_max) {
    if (stream->failed) {
        return -1;
    }
    GGML_ASSERT(dst->n_pos_per_embd == stream->n_pos_per_embd);
    GGML_ASSERT(dst->n_mmproj_embd == stream->n_mmproj_embd);
    GGML_ASSERT(row_offset >= 0 && row_offset + n_max <= dst->batch.n_tokens);
    if (n_max <= 0) {
        return 0;
    }

    // Snapshot the cursor; on error the stream reverts to the state of the
    // last successful drain (rows already written into dst are abandoned by
    // the caller) and is marked dead.
    const size_t    snap_i_input    = stream->i_input;
    const int32_t   snap_n_done_cur = stream->n_done_cur;
    const llama_pos snap_t          = stream->t;
    const int32_t   snap_n_drained  = stream->n_drained;
    const int32_t   snap_n_boundary = stream->n_boundary;
    auto fail = [&]() -> int32_t {
        stream->i_input    = snap_i_input;
        stream->n_done_cur = snap_n_done_cur;
        stream->t          = snap_t;
        stream->n_drained  = snap_n_drained;
        stream->n_boundary = snap_n_boundary;
        stream->failed     = true;
        return -1;
    };

    int32_t n_written = 0;
    while (n_written < n_max && stream->i_input < stream->inputs.size()) {
        const auto & in = stream->inputs[stream->i_input];

        if (in.is_text()) {
            // A run may straddle drain calls; look up each sub-span directly
            // into dst at the fill offset.
            const int32_t n_text = (int32_t) in.text_tokens.size();
            const int32_t take = std::min(n_max - n_written, n_text - stream->n_done_cur);
            int32_t ret = llama_input_embeddings(stream->lctx, in.text_tokens.data() + stream->n_done_cur, take,
                                                 dst->batch.embd + (size_t)(row_offset + n_written) * stream->n_mmproj_embd,
                                                 stream->n_mmproj_embd);
            if (ret != 0) {
                LOG_ERR("%s: llama_input_embeddings failed on text input %zu\n", __func__, stream->i_input);
                return fail();
            }
            dst->set_pos(MTMD_INPUT_CHUNK_TYPE_TEXT, stream->t + stream->n_done_cur, stream->seq_id, take,
                         row_offset + n_written);
            n_written          += take;
            stream->n_done_cur += take;
            stream->n_drained  += take;
            stream->n_boundary  = stream->n_drained; // text tokens are individually atomic
            if (stream->n_done_cur == n_text) {
                stream->t += n_text;
                stream->i_input++;
                stream->n_done_cur = 0;
            }
        } else {
            if (!stream->cur_encoded) {
                // first touch of this media descriptor
                stream->cur_type     = mtmd_input_chunk_get_type(in.chunk);
                stream->cur_n_tokens = (int32_t) mtmd_input_chunk_get_n_tokens(in.chunk);
                if (stream->use_non_causal) {
                    // A non-causal chunk is emitted whole and alone: the caller
                    // must schedule a solo decode with causal attention off.
                    if (n_written > 0 || n_max < stream->cur_n_tokens) {
                        break;
                    }
                }
                stream->cur_image_mrope = stream->cur_type == MTMD_INPUT_CHUNK_TYPE_IMAGE && stream->n_pos_per_embd == 4;
                if (stream->cur_image_mrope) {
                    const auto * image_tokens = mtmd_input_chunk_get_tokens_image(in.chunk);
                    stream->cur_nx = mtmd_image_tokens_get_nx(image_tokens);
                    stream->cur_ny = mtmd_image_tokens_get_ny(image_tokens);
                }
                stream->cur_embd.resize((size_t) stream->cur_n_tokens * stream->n_mmproj_embd);
                int64_t t_enc = ggml_time_ms();
                int32_t ret = mtmd_encode_chunk_into(stream->ctx, in.chunk, stream->cur_embd.data());
                if (ret != 0) {
                    LOG_ERR("%s: mtmd_encode_chunk_into failed on media input %zu\n", __func__, stream->i_input);
                    return fail();
                }
                // LOG_INF("media chunk encoded in %" PRId64 " ms (%d tokens)\n", ggml_time_ms() - t_enc, stream->cur_n_tokens);
                stream->cur_encoded = true;
            }
            const int32_t take = std::min(n_max - n_written, stream->cur_n_tokens - stream->n_done_cur);
            std::memcpy(dst->batch.embd + (size_t)(row_offset + n_written) * stream->n_mmproj_embd,
                        stream->cur_embd.data() + (size_t) stream->n_done_cur * stream->n_mmproj_embd,
                        (size_t) take * stream->n_mmproj_embd * sizeof(float));
            if (stream->cur_image_mrope) {
                dst->set_pos(stream->cur_type, stream->t, stream->seq_id, take,
                             row_offset + n_written, stream->cur_nx, stream->cur_ny, stream->n_done_cur);
            } else {
                // Audio positions are text-like under M-RoPE (each token = 1 temporal tick).
                dst->set_pos(stream->cur_type, stream->t + stream->n_done_cur, stream->seq_id, take,
                             row_offset + n_written);
            }
            n_written          += take;
            stream->n_done_cur += take;
            stream->n_drained  += take;
            if (stream->n_done_cur == stream->cur_n_tokens) {
                stream->t += stream->cur_image_mrope ? 1 : stream->cur_n_tokens;
                stream->n_boundary = stream->n_drained; // media advances the boundary only when complete
                stream->i_input++;
                stream->n_done_cur  = 0;
                stream->cur_encoded = false;
                stream->cur_embd.clear();
                stream->cur_embd.shrink_to_fit();
                if (stream->use_non_causal) {
                    break; // emitted alone; rows after it need causal attention back on
                }
            }
        }
    }
    return n_written;
}

bool mtmd_helper_embd_stream_done(const mtmd_helper_embd_stream * stream) {
    return stream->i_input >= stream->inputs.size();
}

bool mtmd_helper_embd_stream_next_needs_non_causal(const mtmd_helper_embd_stream * stream) {
    return stream->use_non_causal
        && stream->i_input < stream->inputs.size()
        && !stream->inputs[stream->i_input].is_text();
}

int32_t mtmd_helper_embd_stream_next_n_tokens(const mtmd_helper_embd_stream * stream) {
    if (!mtmd_helper_embd_stream_next_needs_non_causal(stream)) {
        return 0;
    }
    return (int32_t) mtmd_input_chunk_get_n_tokens(stream->inputs[stream->i_input].chunk);
}

int32_t mtmd_helper_embd_stream_n_tokens_drained(const mtmd_helper_embd_stream * stream) {
    return stream->n_drained;
}

int32_t mtmd_helper_embd_stream_last_chunk_boundary(const mtmd_helper_embd_stream * stream) {
    return stream->n_boundary;
}

llama_pos mtmd_helper_embd_stream_n_pos(const mtmd_helper_embd_stream * stream) {
    return stream->t;
}

void mtmd_helper_embd_stream_free(mtmd_helper_embd_stream * stream) {
    delete stream;
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

    std::vector<const mtmd_input_chunk *> chunk_ptrs;
    chunk_ptrs.reserve(n_chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        chunk_ptrs.push_back(mtmd_input_chunks_get(chunks, i));
    }

    mtmd_helper_embd_stream * stream = mtmd_helper_embd_stream_init(
            ctx, lctx, chunk_ptrs.data(), chunk_ptrs.size(), n_past, seq_id);
    if (!stream) {
        LOG_ERR("failed to init embd stream\n");
        return -1;
    }

    const int n_embd         = llama_model_n_embd(llama_get_model(lctx));
    const int n_pos_per_embd = mtmd_decode_use_mrope(ctx) ? 4 : 1;
    std::vector<float> embd((size_t) n_batch * n_embd);
    mtmd_decode_embd_batch batch_embd(embd.data(), n_batch, n_pos_per_embd, n_embd);

    int32_t ret = 0;
    while (!mtmd_helper_embd_stream_done(stream)) {
        // a non-causal media chunk is decoded whole and alone, with causal
        // attention off for just that round
        const bool non_causal = mtmd_helper_embd_stream_next_needs_non_causal(stream);
        if (non_causal && mtmd_helper_embd_stream_next_n_tokens(stream) > n_batch) {
            LOG_ERR("non-causal media chunk (%d tokens) does not fit in n_batch (%d)\n",
                    mtmd_helper_embd_stream_next_n_tokens(stream), n_batch);
            ret = -1;
            break;
        }
        const int32_t n_rows = mtmd_helper_embd_stream_drain(stream, &batch_embd, 0, n_batch);
        if (n_rows <= 0) {
            LOG_ERR("failed to drain embd stream\n");
            ret = -1;
            break;
        }
        if (logits_last && mtmd_helper_embd_stream_done(stream)) {
            batch_embd.batch.logits[n_rows - 1] = true;
        }
        llama_batch batch_view = batch_embd.get_view(0, n_rows);
        if (non_causal) {
            llama_set_causal_attn(lctx, false);
        }
        ret = llama_decode(lctx, batch_view);
        if (non_causal) {
            llama_set_causal_attn(lctx, true);
        }
        if (ret != 0) {
            LOG_ERR("failed to decode\n");
            break;
        }
    }

    if (ret == 0) {
        *new_n_past = mtmd_helper_embd_stream_n_pos(stream);
    }
    mtmd_helper_embd_stream_free(stream);
    return ret;
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
