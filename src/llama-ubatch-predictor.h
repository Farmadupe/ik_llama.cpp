#pragma once

// Per-ubatch prefill-time predictor.

#include <cstdint>

class llama_ubatch_predictor {
public:
    llama_ubatch_predictor();

    // Feed one completed ubatch: N new tokens starting at sequence position
    // sequence_offset, took `seconds` of wall clock. (sequence_offset is a llama_pos.)
    void observe_ubatch(uint32_t n_tokens, int32_t sequence_offset, double seconds);

    // Predict a single ubatch. Returns 0 before any observation.
    double predict_ubatch_seconds(uint32_t n_tokens, int32_t sequence_offset) const;

private:
    static constexpr int NUM_BASIS_FEATURES = 3; // number of basis features

    // Build the feature row [1, N, N*(sequence_offset + (N+1)/2)] for a ubatch.
    static void features(uint32_t n_tokens, int32_t sequence_offset, double f[NUM_BASIS_FEATURES]);

    void fit();

    // Online normal equations: A = sum f f^T, b = sum f * y.
    double A[NUM_BASIS_FEATURES][NUM_BASIS_FEATURES];
    double b[NUM_BASIS_FEATURES];
    uint32_t n_obs;

    // Current least-squares solution (refreshed by fit() on every observation).
    double beta[NUM_BASIS_FEATURES];
};
