// Standalone driver for llama_ubatch_predictor, exercised by the pytest suite in this
// directory (test_ubatch_predictor.py). It is deliberately dumb: it owns no policy,
// it just replays whatever the harness feeds it and prints predictions.
//
// Contract (every malformed input exits 1 with no message; clean EOF exits 0):
//
//   argv:  none
//   stdin: repeated cycles of five whitespace-separated numbers
//            predict_n_tokens predict_sequence_offset
//            observe_n_tokens observe_sequence_offset observe_seconds
//
// Per cycle it prints predict_ubatch_seconds(predict_n_tokens, predict_sequence_offset)
// on its own line, then folds (observe_n_tokens, observe_sequence_offset,
// observe_seconds) into the fit. The first cycle predicts against an untrained model
// (which returns 0); the harness, not the driver, decides that that is acceptable.

#include "llama-ubatch-predictor.h"

#include <cstdint>
#include <cstdio>
#include <iostream>

int main(int argc, char **) {
    if (argc != 1) {
        return 1;
    }

    llama_ubatch_predictor predictor;

    for (;;) {
        uint32_t predict_n_tokens = 0;
        // A failed read here is a clean stop only if it was end-of-input.
        if (!(std::cin >> predict_n_tokens)) {
            return std::cin.eof() ? 0 : 1;
        }
        int32_t  predict_sequence_offset = 0;
        uint32_t observe_n_tokens        = 0;
        int32_t  observe_sequence_offset = 0;
        double   observe_seconds         = 0.0;
        if (!(std::cin >> predict_sequence_offset
                       >> observe_n_tokens
                       >> observe_sequence_offset
                       >> observe_seconds)) {
            return 1;
        }

        const double predicted =
            predictor.predict_ubatch_seconds(predict_n_tokens, predict_sequence_offset);
        std::printf("%.10g\n", predicted);

        predictor.observe_ubatch(observe_n_tokens, observe_sequence_offset, observe_seconds);
    }
}
