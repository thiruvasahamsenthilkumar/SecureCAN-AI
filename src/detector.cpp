#include "detector.h"
#include <math.h>
#include <string.h>

static can_msg_t win[CAN_WINDOW];
static uint16_t  win_n = 0;

static const char *CLASS_NAMES[] = {"NORMAL", "DOS", "FUZZY", "FAKE_RPM", "FAKE_GEAR"};

void window_reset(void) { win_n = 0; }
uint8_t window_full(void) { return win_n >= CAN_WINDOW; }

void window_push(const can_msg_t *m) {
    if (win_n < CAN_WINDOW) {
        win[win_n++] = *m;
    } else {
        memmove(win, win + 1, sizeof(can_msg_t) * (CAN_WINDOW - 1));
        win[CAN_WINDOW - 1] = *m;
    }
}

// Feature order must match FEATURES in scripts/pipeline.py.
void extract_features(float *out) {
    float iat_sum = 0, iat_sq = 0, iat_min = 1e9f;
    uint16_t iat_n = 0;

    for (uint16_t i = 1; i < win_n; i++) {
        float dt = (float)(win[i].timestamp_us - win[i - 1].timestamp_us) * 1e-6f;
        if (dt <= 0) continue;
        iat_sum += dt;
        iat_sq  += dt * dt;
        if (dt < iat_min) iat_min = dt;
        iat_n++;
    }
    if (iat_n == 0) { iat_n = 1; iat_sum = 1e-6f; iat_sq = 1e-12f; iat_min = 1e-6f; }

    float iat_mean = iat_sum / iat_n;
    float iat_var  = iat_sq / iat_n - iat_mean * iat_mean;
    if (iat_var < 0) iat_var = 0;

    float span = (float)(win[win_n - 1].timestamp_us - win[0].timestamp_us) * 1e-6f;
    if (span < 1e-6f) span = 1e-6f;

    // CAN ID histogram (linear scan; window is small so this stays cheap)
    uint16_t uniq_ids[CAN_WINDOW];
    uint16_t id_counts[CAN_WINDOW];
    uint16_t n_uniq = 0, max_count = 0;
    for (uint16_t i = 0; i < win_n; i++) {
        uint16_t j = 0;
        for (; j < n_uniq; j++) if (uniq_ids[j] == win[i].can_id) { id_counts[j]++; break; }
        if (j == n_uniq) { uniq_ids[n_uniq] = win[i].can_id; id_counts[n_uniq] = 1; n_uniq++; }
        if (id_counts[j] > max_count) max_count = id_counts[j];
    }
    float id_entropy = 0;
    for (uint16_t j = 0; j < n_uniq; j++) {
        float p = (float)id_counts[j] / win_n;
        id_entropy -= p * log2f(p);
    }

    // Byte statistics
    uint16_t bhist[256] = {0};
    uint32_t dlc_sum = 0, b_sum = 0, b_sq = 0, sat = 0, zero = 0, nbytes = 0;
    uint8_t  b_max = 0;
    for (uint16_t i = 0; i < win_n; i++) {
        dlc_sum += win[i].dlc;
        for (uint8_t k = 0; k < 8; k++) {
            uint8_t b = win[i].data[k];
            bhist[b]++; b_sum += b; b_sq += (uint32_t)b * b;
            if (b > b_max) b_max = b;
            if (b >= 250) sat++;
            if (b == 0)   zero++;
            nbytes++;
        }
    }
    float b_mean = (float)b_sum / nbytes;
    float b_var  = (float)b_sq / nbytes - b_mean * b_mean;
    if (b_var < 0) b_var = 0;

    float b_entropy = 0;
    for (uint16_t v = 0; v < 256; v++) {
        if (!bhist[v]) continue;
        float p = (float)bhist[v] / nbytes;
        b_entropy -= p * log2f(p);
    }

    out[0]  = (float)CAN_WINDOW / span;        // msg_rate
    out[1]  = iat_mean;                        // mean_iat
    out[2]  = sqrtf(iat_var);                  // std_iat
    out[3]  = iat_min;                         // min_iat
    out[4]  = (float)n_uniq;                   // unique_ids
    out[5]  = id_entropy;                      // id_entropy
    out[6]  = (float)max_count / CAN_WINDOW;   // dominant_id_ratio
    out[7]  = (float)dlc_sum / win_n;          // mean_dlc
    out[8]  = b_mean;                          // mean_byte
    out[9]  = sqrtf(b_var);                    // std_byte
    out[10] = (float)b_max;                    // max_byte
    out[11] = b_entropy;                       // byte_entropy
    out[12] = (float)sat / nbytes;             // saturated_ratio
    out[13] = (float)zero / nbytes;            // zero_ratio
}

// Quantize features to int8 using the scales exported from training.
static void quantize(const float *f, int8_t *q) {
    for (uint8_t i = 0; i < N_FEATURES; i++) {
        float v = (f[i] - FEAT_LO[i]) * FEAT_SCALE[i] - 127.0f;
        if (v > 127.0f)  v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        q[i] = (int8_t)v;
    }
}

// Majority vote over the quantized decision forest.
uint8_t classify(const float *features) {
    int8_t q[N_FEATURES];
    quantize(features, q);

    uint16_t votes[N_CLASSES] = {0};
    for (uint16_t t = 0; t < N_TREES; t++) {
        int16_t base = TREE_OFFSET[t];
        int16_t node = 0;
        while (NODE_FEATURE[base + node] >= 0) {
            int8_t f = NODE_FEATURE[base + node];
            node = (q[f] <= NODE_THRESH[base + node]) ? NODE_LEFT[base + node]
                                                      : NODE_RIGHT[base + node];
        }
        votes[NODE_CLASS[base + node]]++;
    }

    uint8_t best = 0;
    for (uint8_t c = 1; c < N_CLASSES; c++) if (votes[c] > votes[best]) best = c;
    return best;
}

const char *class_name(uint8_t c) {
    return (c < N_CLASSES) ? CLASS_NAMES[c] : "UNKNOWN";
}
