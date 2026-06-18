#pragma once

#include <cstdint>

// ---- Per-request preprocessing-stage instrumentation ----
//
// One request_trace is created at HTTP entry, threaded by raw pointer through
// the parse/tokenize free functions, carried across the task queue by a
// std::shared_ptr member on server_task, and finally read on the worker thread
// via slot.task->trace (the shared_ptr's pointee stays mutable through the
// const server_task).
//
// Threading invariant (why the fields are plain, non-atomic): the HTTP thread
// performs every write during parse/tokenize, all of which complete before the
// task is posted to the queue. The queue's mutex establishes happens-before, so
// the worker thread observes those writes; the worker then records the
// slot-launch checkpoint and prints. There is never concurrent access.
class request_trace {
public:
    // Stage checkpoints. mark_slot_launch() and mark_preprocess_finished() are set
    // on the worker thread; the rest are set on the HTTP thread.
    void mark_request_received();
    void mark_post_template();
    void mark_post_tokenize();
    void mark_slot_launch();

    // Stamp the end-of-preprocessing boundary: the model is about to run this
    // request's first forward pass. First-write-wins, so the earliest call (right
    // before the first prefill decode) defines the boundary even when prefill is
    // chunked. This is what arms print().
    void mark_preprocess_finished();

    // base64 decode cost, summed across calls.
    void add_base64_us(int64_t dt_us);

    // Multimodal input stats, set once in process_mtmd_prompt.
    void set_mtmd_stats(int64_t image_decode_us, int64_t mtmd_tokenize_us,
                        int64_t n_files, int64_t total_file_bytes);

    // Emit the preprocessing-timing report to stderr. Idempotent and gated on
    // mark_preprocess_finished(): the first call after the boundary is stamped
    // prints; every later call (and any call before the boundary) is a no-op.
    void print();

private:
    // Absolute wall-clock checkpoints (us).
    int64_t t_request_received_us = 0;
    int64_t t_post_template_us    = 0;
    int64_t t_post_tokenize_us    = 0;
    int64_t t_slot_launch_us      = 0;
    int64_t t_preprocess_end_us   = 0; // model about to run the first forward pass

    // Accumulated / per-request op costs (us) and input-shape stats.
    int64_t t_base64_us        = 0;
    int64_t t_image_decode_us  = 0;
    int64_t t_mtmd_tokenize_us = 0;
    int64_t n_files            = 0;
    int64_t total_file_bytes   = 0;

    bool printed_ = false;
};
