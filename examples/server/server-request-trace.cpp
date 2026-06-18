#include "server-request-trace.h"

#include "ggml.h" // ggml_time_us

#include <cstdio>
#include <cinttypes>

void request_trace::mark_request_received() { t_request_received_us = ggml_time_us(); }
void request_trace::mark_post_template()    { t_post_template_us    = ggml_time_us(); }
void request_trace::mark_post_tokenize()    { t_post_tokenize_us    = ggml_time_us(); }
void request_trace::mark_slot_launch()      { t_slot_launch_us      = ggml_time_us(); }

void request_trace::mark_preprocess_finished() {
    // First-write-wins: keep the earliest boundary (first prefill forward pass).
    if (!t_preprocess_end_us) t_preprocess_end_us = ggml_time_us();
}

void request_trace::add_base64_us(int64_t dt_us) { t_base64_us += dt_us; }

void request_trace::set_mtmd_stats(int64_t image_decode_us, int64_t mtmd_tokenize_us,
                                   int64_t n_files_, int64_t total_file_bytes_) {
    t_image_decode_us  = image_decode_us;
    t_mtmd_tokenize_us = mtmd_tokenize_us;
    n_files            = n_files_;
    total_file_bytes   = total_file_bytes_;
}

void request_trace::print() {
    if (printed_) return;
    // Gated on the end-of-preprocessing boundary having been stamped.
    if (!t_preprocess_end_us) return;
    printed_ = true;

    const int64_t t_decode = t_preprocess_end_us;

    auto sec_us = [](int64_t us) { return (double)us / 1.0e6; };

    // Avoid double-counting: base64 is broken out separately, so subtract it from
    // the "parse json + jinja" window. The non-chat path doesn't pass through
    // Jinja; base64 still gets accounted, just under "parse json" instead.
    const int64_t t_pre_tok = (t_post_template_us ? t_post_template_us : t_post_tokenize_us) - t_request_received_us;
    const int64_t t_json_jinja_minus_b64 = t_pre_tok - t_base64_us;

    // Format: name field 39 wide left-justified, number 8 wide right-justified
    // with 2 decimals - decimal points align across all rows.
    fprintf(stderr, "preprocess timing:\n");
    fprintf(stderr, "%-39s%8.2f s\n", t_post_template_us ? "* parse json + jinja:" : "* parse json:",
                                                   sec_us(t_json_jinja_minus_b64));
    fprintf(stderr, "%-39s%8.2f s\n", "* base64 decode:",            sec_us(t_base64_us));
    fprintf(stderr, "%-39s%8.2f s\n", "* image decode:",             sec_us(t_image_decode_us));
    fprintf(stderr, "%-39s%8.2f s\n", "* mtmd_tokenize:",            sec_us(t_mtmd_tokenize_us));
    fprintf(stderr, "%-39s%8.2f s\n", "* queue + slot select:",      sec_us(t_slot_launch_us - t_post_tokenize_us));
    fprintf(stderr, "%-39s%8.2f s\n", "* prompt cache + batch prep:", sec_us(t_decode - t_slot_launch_us));
    fprintf(stderr, "%-39s%8.2f s\n", "* total preprocessing time:", sec_us(t_decode - t_request_received_us));

    if (n_files > 0) {
        const double avg_mb = (double)total_file_bytes / (double)n_files / 1.0e6;
        fprintf(stderr, "mtmd input stats:\n");
        fprintf(stderr, "%-39s%8" PRId64 "\n",     "* images:",             n_files);
        fprintf(stderr, "%-39s%8.3f MB\n",        "* average image size:", avg_mb);
    }
}
