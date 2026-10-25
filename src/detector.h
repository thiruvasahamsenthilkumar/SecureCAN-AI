#pragma once
#include <stdint.h>
#include "../model/detector_quantized.h"

#define CAN_WINDOW 50

typedef struct {
    uint32_t timestamp_us;
    uint16_t can_id;
    uint8_t  dlc;
    uint8_t  data[8];
} can_msg_t;

// Rolling window of CAN messages -> feature vector -> class id.
void     window_reset(void);
void     window_push(const can_msg_t *m);
uint8_t  window_full(void);
void     extract_features(float *out);        // writes N_FEATURES floats
uint8_t  classify(const float *features);     // returns class index
const char *class_name(uint8_t c);
