#ifndef MTMD_HELPER_H
#define MTMD_HELPER_H

#include "ggml.h"
#include "llama.h"
#include "mtmd.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// libmtmd helper functions
//
// Please note that these helpers are not guaranteed to be stable.
// BREAKING CHANGES are expected.
//

typedef int32_t (*mtmd_helper_eval_batch_callback)(void * user_data, const struct llama_batch * batch);

// Staging batch of embedding rows for llama_decode. Definition is in the C++
// section at the bottom of this header; C callers treat it as opaque.
struct mtmd_decode_embd_batch;

// helper function to construct a mtmd_bitmap from a file
// it calls mtmd_helper_bitmap_init_from_buf() internally
// returns nullptr on failure
// this function is thread-safe
MTMD_API mtmd_bitmap * mtmd_helper_bitmap_init_from_file(mtmd_context * ctx, const char * fname);

// helper function to construct a mtmd_bitmap from a buffer containing a file
// supported formats:
//     image: formats supported by stb_image: jpg, png, bmp, gif, etc.
//     audio: formats supported by miniaudio: wav, mp3, flac
// note: audio files will be auto-detected based on magic bytes
// returns nullptr on failure
// this function is thread-safe
MTMD_API mtmd_bitmap * mtmd_helper_bitmap_init_from_buf(mtmd_context * ctx, const unsigned char * buf, size_t len);

// like mtmd_helper_bitmap_init_from_buf, but consults the media memo on ctx:
// on a hit, pixel decoding is skipped entirely and the returned bitmap carries
// only identity (id, dims) plus the encoded source bytes. on a miss, decodes
// eagerly and attaches the source bytes so tokenize can memoize.
// in both cases the bitmap id (pixel-content hash) is set; audio is always eager.
MTMD_API mtmd_bitmap * mtmd_helper_bitmap_init_lazy(mtmd_context * ctx, const unsigned char * buf, size_t len);

// decode image or IMAGE_TEMPORAL container bytes into a pixel bitmap.
// no audio handling, no memo interaction.
MTMD_API mtmd_bitmap * mtmd_helper_decode_container(const unsigned char * buf, size_t len);

// helper to count the total number of tokens from a list of chunks, useful to keep track of KV cache
MTMD_API size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks * chunks);

// helper to count the total position of tokens from a list of chunks, useful to keep track of n_past
// normally, n_pos is equal to n_tokens, but for M-RoPE it is different
MTMD_API llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks * chunks);

// helper function that automatically:
// 1. run llama_decode() on text chunks
// 2. run mtmd_encode() on image chunks, then mtmd_get_output_embd() and then llama_decode()
// if any of the mtmd_encode() or llama_decode() calls return non-zero, stop and forward the error
// otherwise, returns 0 on success
// this function is NOT thread-safe
MTMD_API int32_t mtmd_helper_eval_chunks(mtmd_context * ctx,
                                         struct llama_context * lctx,
                                         const mtmd_input_chunks * chunks,
                                         llama_pos n_past,
                                         llama_seq_id seq_id,
                                         int32_t n_batch,
                                         bool logits_last,
                                         llama_pos * new_n_past);

// Input descriptor for the embd stream below. Either a run of raw text tokens
// (we'll look up their embeddings via the model's input-embedding LUT) or a single
// media chunk (we'll run the vision/audio encoder via mtmd_encode_chunk_into).
struct mtmd_helper_coalesce_input {
    bool                     is_text;        // true: text run; false: media chunk
    // text fields (valid iff is_text == true):
    const llama_token *      text_tokens;
    int32_t                  n_text_tokens;
    // media field (valid iff is_text == false):
    const mtmd_input_chunk * chunk;
};

// Resumable embedding stream over a coalesced multimodal input.
//
// The stream is a cursor over an array of mtmd_helper_coalesce_input
// descriptors. Each drain call fills up to n_max embedding rows (plus
// positions, seq ids and logits flags, all cleared to false) into a
// caller-provided staging batch. The stream can suspend anywhere, including
// in the middle of an image. It never calls llama_decode and never toggles
// causal attention; the caller owns both.
//
// The descriptor array is copied, but the token spans and chunks it
// references must outlive the stream.
//
// this API is NOT thread-safe
typedef struct mtmd_helper_embd_stream mtmd_helper_embd_stream;

// returns nullptr on invalid input (media descriptor that is not IMAGE/AUDIO)
MTMD_API mtmd_helper_embd_stream * mtmd_helper_embd_stream_init(
        mtmd_context * ctx,
        struct llama_context * lctx,
        const struct mtmd_helper_coalesce_input * inputs,
        size_t n_inputs,
        llama_pos n_past,
        llama_seq_id seq_id);

// Fill up to n_max rows into dst starting at row_offset. Returns the number
// of rows written (possibly 0), or a negative value on error (the stream is
// then dead; further drains keep failing).
//
// Returns fewer than n_max rows when:
//   - the stream ran out of input (check mtmd_helper_embd_stream_done)
//   - the next item is a media chunk of a non-causal model (check
//     mtmd_helper_embd_stream_next_needs_non_causal); the caller must then
//     schedule a decode with causal attention disabled and call drain again
//     with n_max >= mtmd_helper_embd_stream_next_n_tokens, which emits the
//     whole chunk in one call and then stops again
MTMD_API int32_t mtmd_helper_embd_stream_drain(
        mtmd_helper_embd_stream * stream,
        struct mtmd_decode_embd_batch * dst,
        int32_t row_offset,
        int32_t n_max);

// true once every input row has been drained
MTMD_API bool mtmd_helper_embd_stream_done(const mtmd_helper_embd_stream * stream);

// true if the next row to drain starts a media chunk that requires
// non-causal attention (gemma3 family); such a chunk is only ever emitted
// whole and alone
MTMD_API bool mtmd_helper_embd_stream_next_needs_non_causal(const mtmd_helper_embd_stream * stream);

// n_tokens of the pending non-causal media chunk, or 0 if there is none
MTMD_API int32_t mtmd_helper_embd_stream_next_n_tokens(const mtmd_helper_embd_stream * stream);

// total rows drained so far, including a partially drained media chunk
MTMD_API int32_t mtmd_helper_embd_stream_n_tokens_drained(const mtmd_helper_embd_stream * stream);

// rows drained up to the last point that is safe to mirror or rewind to:
// advances per row through text runs, but only at completion for media chunks
MTMD_API int32_t mtmd_helper_embd_stream_last_chunk_boundary(const mtmd_helper_embd_stream * stream);

// current temporal position counter; equals the final n_past once done() is
// true. While a media chunk is partially drained this stays at the position
// where the chunk started
MTMD_API llama_pos mtmd_helper_embd_stream_n_pos(const mtmd_helper_embd_stream * stream);

MTMD_API void mtmd_helper_embd_stream_free(mtmd_helper_embd_stream * stream);

// works like mtmd_helper_eval_chunks(), but only for a single chunk
// this function is NOT thread-safe
MTMD_API int32_t mtmd_helper_eval_chunk_single(mtmd_context * ctx,
                                               struct llama_context * lctx,
                                               const mtmd_input_chunk * chunk,
                                               llama_pos n_past,
                                               llama_seq_id seq_id,
                                               int32_t n_batch,
                                               bool logits_last,
                                               llama_pos * new_n_past);

// works like mtmd_helper_eval_chunk_single(), and calls callback after each successful llama_decode() batch
// the batch pointer is only valid for the duration of the callback
MTMD_API int32_t mtmd_helper_eval_chunk_single_with_callback(mtmd_context * ctx,
                                                             struct llama_context * lctx,
                                                             const mtmd_input_chunk * chunk,
                                                             llama_pos n_past,
                                                             llama_seq_id seq_id,
                                                             int32_t n_batch,
                                                             bool logits_last,
                                                             llama_pos * new_n_past,
                                                             mtmd_helper_eval_batch_callback callback,
                                                             void * callback_user_data);

// helper function to decode an image whose embeddings have already been calculated
// this helper will handle batching and pre/post decoding setup (for ex. gemma 3 requires non-causal attention)
// ret 0 on success, -1 on chunk not being a valid image chunk, 1 on decode failure
MTMD_API int32_t mtmd_helper_decode_image_chunk(mtmd_context * ctx,
                                                struct llama_context * lctx,
                                                const mtmd_input_chunk * chunk,
                                                float * encoded_embd,
                                                llama_pos n_past,
                                                llama_seq_id seq_id,
                                                int32_t n_batch,
                                                llama_pos * new_n_past);

#ifdef __cplusplus
} // extern "C"
#endif

//
// C++ wrappers
//

#ifdef __cplusplus

#include <vector>

// helper struct to make working with embd batch easier
// note: this will be removed after llama_batch_ext refactoring
struct mtmd_decode_embd_batch {
    int n_pos_per_embd;
    int n_mmproj_embd;
    std::vector<llama_pos>      pos;
    std::vector<llama_pos>      pos_view; // used by mrope
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0; // one entry per row; each batch.seq_id[i] points at its own entry
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch batch;
    mtmd_decode_embd_batch(float * embd, int32_t n_tokens, int n_pos_per_embd, int n_mmproj_embd) : n_pos_per_embd(n_pos_per_embd), n_mmproj_embd(n_mmproj_embd) {
        pos     .resize(n_tokens * n_pos_per_embd);
        n_seq_id.resize(n_tokens);
        seq_ids .resize(n_tokens + 1);
        logits  .resize(n_tokens);
        seq_id_0.resize(n_tokens);
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
        for (int i = 0; i < n_tokens; i++) {
            seq_id_0[offset + i] = seq_id;
            batch.n_seq_id[offset + i] = 1;
            batch.seq_id  [offset + i] = &seq_id_0[offset + i];
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

#endif // __cplusplus

#endif
