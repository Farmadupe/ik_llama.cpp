#include "llama-ubatch-predictor.h"

#include <cmath>

llama_ubatch_predictor::llama_ubatch_predictor() : n_obs(0) {
    for (int i = 0; i < NUM_BASIS_FEATURES; ++i) {
        b[i]    = 0.0;
        beta[i] = 0.0;
        for (int j = 0; j < NUM_BASIS_FEATURES; ++j) {
            A[i][j] = 0.0;
        }
    }
}

void llama_ubatch_predictor::features(uint32_t n_tokens, int32_t sequence_offset, double f[NUM_BASIS_FEATURES]) {
    const double N = (double) n_tokens;
    f[0] = 1.0;                                              // weight floor (full expert sweep + always-touched residue)
    f[1] = N;                                                // active-parameter compute
    f[2] = N * ((double) sequence_offset + 0.5*(N + 1.0));   // causal KV sweep, N*(offset + (N+1)/2)
}

void llama_ubatch_predictor::observe_ubatch(uint32_t n_tokens, int32_t sequence_offset, double seconds) {
    double f[NUM_BASIS_FEATURES];
    features(n_tokens, sequence_offset, f);
    for (int i = 0; i < NUM_BASIS_FEATURES; ++i) {
        b[i] += f[i] * seconds;
        for (int j = 0; j < NUM_BASIS_FEATURES; ++j) {
            A[i][j] += f[i] * f[j];
        }
    }
    ++n_obs;
    fit();
}

void llama_ubatch_predictor::fit() {
    for (int i = 0; i < NUM_BASIS_FEATURES; ++i) {
        beta[i] = 0.0;
    }
    if (n_obs == 0) {
        return;
    }

    // Column scaling
    double s[NUM_BASIS_FEATURES];
    for (int j = 0; j < NUM_BASIS_FEATURES; ++j) {
        s[j] = std::sqrt(A[j][j]);
        if (s[j] <= 0.0) {
            s[j] = 1.0; // dead column: leave its coefficient at 0
        }
    }

    double M[NUM_BASIS_FEATURES][NUM_BASIS_FEATURES];
    double r[NUM_BASIS_FEATURES];
    const double ridge = 1e-9; // relative to scaled unit diagonal
    for (int i = 0; i < NUM_BASIS_FEATURES; ++i) {
        r[i] = b[i] / s[i];
        for (int j = 0; j < NUM_BASIS_FEATURES; ++j) {
            M[i][j] = A[i][j] / (s[i] * s[j]);
        }
        M[i][i] += ridge;
    }

    // Gaussian elimination with partial pivoting on the 3x3 scaled system M y = r.
    double y[NUM_BASIS_FEATURES];
    for (int col = 0; col < NUM_BASIS_FEATURES; ++col) {
        int piv = col;
        double best = std::fabs(M[col][col]);
        for (int row = col + 1; row < NUM_BASIS_FEATURES; ++row) {
            const double v = std::fabs(M[row][col]);
            if (v > best) {
                best = v;
                piv  = row;
            }
        }
        if (piv != col) {
            for (int k = 0; k < NUM_BASIS_FEATURES; ++k) {
                const double t = M[col][k]; M[col][k] = M[piv][k]; M[piv][k] = t;
            }
            const double t = r[col]; r[col] = r[piv]; r[piv] = t;
        }
        const double diag = M[col][col];
        if (std::fabs(diag) < 1e-300) {
            continue; // should not happen with ridge, but stay safe
        }
        for (int row = col + 1; row < NUM_BASIS_FEATURES; ++row) {
            const double factor = M[row][col] / diag;
            if (factor == 0.0) {
                continue;
            }
            for (int k = col; k < NUM_BASIS_FEATURES; ++k) {
                M[row][k] -= factor * M[col][k];
            }
            r[row] -= factor * r[col];
        }
    }
    for (int row = NUM_BASIS_FEATURES - 1; row >= 0; --row) {
        double acc = r[row];
        for (int k = row + 1; k < NUM_BASIS_FEATURES; ++k) {
            acc -= M[row][k] * y[k];
        }
        const double diag = M[row][row];
        y[row] = (std::fabs(diag) < 1e-300) ? 0.0 : acc / diag;
    }

    for (int j = 0; j < NUM_BASIS_FEATURES; ++j) {
        beta[j] = y[j] / s[j];
    }
}

double llama_ubatch_predictor::predict_ubatch_seconds(uint32_t n_tokens, int32_t sequence_offset) const {
    double f[NUM_BASIS_FEATURES];
    features(n_tokens, sequence_offset, f);
    double t = 0.0;
    for (int i = 0; i < NUM_BASIS_FEATURES; ++i) {
        t += beta[i] * f[i];
    }
    return t;
}
