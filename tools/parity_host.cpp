// Host harness for tests/test_parity.py.
//
// Compiles firmware/esp32/src/features.h for the laptop instead of the ESP32,
// reads frames on stdin, writes the 16 features per frame to stdout. The test
// then feeds the SAME frames to the Python StreamingExtractor and compares.
//
// This is the only way to be sure the two implementations agree. Reading both
// files side by side and concluding they look the same is how off-by-one bugs
// ship -- and a feature mismatch would be silent, because the firmware would
// still produce confident-looking answers from a model trained on different
// numbers.
//
// We compile the real header, not a copy. If features.h stops building, the
// parity test fails, which is the behaviour we want.
//
//   g++ -O2 -std=c++17 -I firmware/esp32/src -o parity_host tools/parity_host.cpp
//
// Two modes, because there are two things that have to agree:
//
//   (default)  features only.  stdout: 16 floats per frame, %.9g
//              checked against src/securecan/features.py
//
//   --nn       the whole chain. stdout: "<class> <confidence>" per frame
//              checked against TFLite's own int8 interpreter, fed the features
//              that features.py computed. So this mode catches a mistake in
//              nn.h even if the features are perfect, and vice versa.
//
// stdin (both): one frame per line -- t_us id dlc b0 b1 b2 b3 b4 b5 b6 b7

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "features.h"
#include "nn.h"

int main(int argc, char** argv) {
  const bool nn_mode = argc > 1 && strcmp(argv[1], "--nn") == 0;

  ScFeatureExtractor ex;
  float feats[SC_N_FEATURES];

  unsigned long t_us;
  unsigned id, dlc;
  unsigned b[8];

  while (scanf("%lu %u %u %u %u %u %u %u %u %u %u",
               &t_us, &id, &dlc,
               &b[0], &b[1], &b[2], &b[3], &b[4], &b[5], &b[6], &b[7]) == 11) {
    ScFrame f;
    f.t_us = (uint32_t)t_us;
    f.id = (uint16_t)id;
    f.dlc = (uint8_t)dlc;
    for (int i = 0; i < 8; ++i) f.data[i] = (uint8_t)b[i];

    ex.push(f, feats);

    if (nn_mode) {
      float conf = 0.0f;
      const int cls = sc_nn_predict(feats, &conf);
      printf("%d %.9g\n", cls, conf);
    } else {
      for (int i = 0; i < SC_N_FEATURES; ++i) {
        printf("%.9g%c", feats[i], i == SC_N_FEATURES - 1 ? '\n' : ' ');
      }
    }
  }
  return 0;
}
