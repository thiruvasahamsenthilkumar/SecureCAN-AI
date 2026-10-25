// Host harness: runs the exact on-device code path (src/detector.cpp) over the
// full capture and dumps features + predictions for comparison against sklearn.
//
//   g++ -O2 -std=c++17 tests/host_eval.cpp src/detector.cpp -o build/host_eval
//   ./build/host_eval data/can_traffic.csv tests/host_predictions.csv
//
// Windowing (size 50, stride 25) and the window-labelling rule mirror
// extract_features() in scripts/pipeline.py so the two sides are comparable.

#include "../src/detector.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>

static const int STRIDE = CAN_WINDOW / 2;

int main(int argc, char **argv) {
    const char *in_path  = argc > 1 ? argv[1] : "data/can_traffic.csv";
    const char *out_path = argc > 2 ? argv[2] : "tests/host_predictions.csv";

    FILE *f = fopen(in_path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", in_path); return 1; }

    char line[512];
    if (!fgets(line, sizeof line, f)) { fclose(f); return 1; }  // header

    std::vector<can_msg_t> msgs;
    std::vector<int> labels;
    while (fgets(line, sizeof line, f)) {
        double ts; int cid, dlc, b[8], lab;
        if (sscanf(line, "%lf,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
                   &ts, &cid, &dlc, &b[0], &b[1], &b[2], &b[3],
                   &b[4], &b[5], &b[6], &b[7], &lab) != 12) continue;
        can_msg_t m;
        m.timestamp_us = (uint32_t)(ts * 1e6 + 0.5);   // what micros() would give
        m.can_id = (uint16_t)cid;
        m.dlc    = (uint8_t)dlc;
        for (int k = 0; k < 8; k++) m.data[k] = (uint8_t)b[k];
        msgs.push_back(m);
        labels.push_back(lab);
    }
    fclose(f);
    printf("[host] %zu messages from %s\n", msgs.size(), in_path);

    FILE *out = fopen(out_path, "w");
    if (!out) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }
    fprintf(out, "idx,true,pred");
    for (int i = 0; i < N_FEATURES; i++) fprintf(out, ",f%d", i);
    fprintf(out, "\n");

    std::vector<double> t_extract, t_classify;
    long n_windows = 0, n_correct = 0;

    for (size_t s = 0; s + CAN_WINDOW < msgs.size(); s += STRIDE) {
        window_reset();
        for (int i = 0; i < CAN_WINDOW; i++) window_push(&msgs[s + i]);

        float feats[N_FEATURES];
        auto t0 = std::chrono::steady_clock::now();
        extract_features(feats);
        auto t1 = std::chrono::steady_clock::now();
        uint8_t pred = classify(feats);
        auto t2 = std::chrono::steady_clock::now();

        t_extract.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        t_classify.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());

        // window label = majority attack class, if attacks are >30% of the window
        int hist[5] = {0}, n_atk = 0;
        for (int i = 0; i < CAN_WINDOW; i++) {
            int l = labels[s + i];
            if (l > 0) { hist[l]++; n_atk++; }
        }
        int truth = 0;
        if (n_atk > CAN_WINDOW * 0.3) {
            int best = 1;
            for (int c = 2; c < 5; c++) if (hist[c] > hist[best]) best = c;
            truth = best;
        }

        fprintf(out, "%ld,%d,%u", n_windows, truth, pred);
        for (int i = 0; i < N_FEATURES; i++) fprintf(out, ",%.9g", feats[i]);
        fprintf(out, "\n");

        if (pred == truth) n_correct++;
        n_windows++;
    }
    fclose(out);

    auto pct = [](std::vector<double> v, double p) {
        std::sort(v.begin(), v.end());
        return v[(size_t)(p * (v.size() - 1))];
    };
    double ext_med = pct(t_extract, 0.5), cls_med = pct(t_classify, 0.5);

    printf("[host] %ld windows | agreement with ground truth %.2f%%\n",
           n_windows, 100.0 * n_correct / n_windows);
    printf("[host] extract_features  median %.2f us   p99 %.2f us\n",
           ext_med, pct(t_extract, 0.99));
    printf("[host] classify (forest) median %.2f us   p99 %.2f us\n",
           cls_med, pct(t_classify, 0.99));
    printf("[host] total per window  median %.2f us\n", ext_med + cls_med);
    printf("[host] wrote %s\n", out_path);
    return 0;
}
