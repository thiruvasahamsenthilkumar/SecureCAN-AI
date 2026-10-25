// Compiles src/detector.cpp for the host and runs it over feature vectors
// exported from Python, so the on-device integer path can be checked against
// sklearn without an ESP32 in the loop.
//
//   g++ -O2 -I../src -I../model -o harness host_harness.cpp ../src/detector.cpp
//   ./harness features.csv > host_predictions.csv
//
// Also runs the full extract_features() path over synthetic windows to confirm
// the device-side feature code agrees with the Python implementation.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include "detector.h"

static int run_feature_csv(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }

    char line[8192];
    if (!fgets(line, sizeof line, f)) { fclose(f); return 1; }   // header

    printf("row,predicted,latency_ns\n");
    long row = 0;
    double total_ns = 0;

    while (fgets(line, sizeof line, f)) {
        float feats[N_FEATURES];
        char *tok = strtok(line, ",\n");
        for (int i = 0; i < N_FEATURES && tok; i++) {
            feats[i] = strtof(tok, nullptr);
            tok = strtok(nullptr, ",\n");
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        uint8_t cls = classify(feats);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        total_ns += ns;

        printf("%ld,%u,%.0f\n", row++, (unsigned)cls, ns);
    }
    fclose(f);

    fprintf(stderr, "classified %ld vectors | mean %.0f ns/inference\n",
            row, total_ns / (row ? row : 1));
    return 0;
}

// --- feature-extraction cross-check -------------------------------------
// Rebuilds the windows that scripts/export_test_vectors.py wrote out, pushes
// them through window_push/extract_features, and prints the resulting feature
// vector so Python can diff it against its own.

static int run_window_csv(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }

    char line[512];
    if (!fgets(line, sizeof line, f)) { fclose(f); return 1; }   // header

    printf("window");
    for (int i = 0; i < N_FEATURES; i++) printf(",f%d", i);
    printf(",predicted\n");

    long cur_win = -1;
    long emitted = 0;
    bool open = false;

    auto flush = [&]() {
        if (!open || !window_full()) return;
        float feats[N_FEATURES];
        extract_features(feats);
        printf("%ld", cur_win);
        for (int i = 0; i < N_FEATURES; i++) printf(",%.6f", feats[i]);
        printf(",%u\n", (unsigned)classify(feats));
        emitted++;
    };

    while (fgets(line, sizeof line, f)) {
        long w;
        double ts;
        unsigned id, dlc, b[8];
        if (sscanf(line, "%ld,%lf,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                   &w, &ts, &id, &dlc, &b[0], &b[1], &b[2], &b[3],
                   &b[4], &b[5], &b[6], &b[7]) != 12) continue;

        if (w != cur_win) {
            flush();
            window_reset();
            cur_win = w;
            open = true;
        }

        can_msg_t m;
        m.timestamp_us = (uint32_t)(ts * 1e6);
        m.can_id = (uint16_t)id;
        m.dlc = (uint8_t)dlc;
        for (int k = 0; k < 8; k++) m.data[k] = (uint8_t)b[k];
        window_push(&m);
    }
    flush();
    fclose(f);

    fprintf(stderr, "extracted %ld windows\n", emitted);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <features.csv> [--windows]\n", argv[0]);
        return 2;
    }
    bool windows = (argc > 2 && strcmp(argv[2], "--windows") == 0);
    return windows ? run_window_csv(argv[1]) : run_feature_csv(argv[1]);
}
