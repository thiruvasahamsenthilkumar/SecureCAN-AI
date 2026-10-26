// SecureCAN AI -- feature extraction on the device.
//
// This is a direct port of StreamingExtractor in src/securecan/features.py.
// That file is the single source of truth; tests/test_parity.py feeds the same
// frames to both and asserts the 16 outputs agree. If you change one, change the
// other and re-run the parity test.
//
// Three things constrain the design, and all three are worth being able to
// explain in an interview:
//
// 1. NO LOOKAHEAD. The device sees one frame, must answer, then sees the next.
//    So every feature uses only the current frame plus a small memory of the
//    past, and state is updated AFTER the features are computed -- otherwise a
//    frame would see itself and dt_same_id would always be zero.
//
// 2. FIXED MEMORY. The per-ID table is a compile-time array sized by the number
//    of IDs learned from normal traffic. Unknown IDs are never inserted. That is
//    a security property, not an optimisation: the Fuzzy attack sprays random
//    IDs, and a table that grew on demand would be a remote memory-exhaustion
//    bug in the thing that is supposed to be the defence. The fact that an ID
//    has no slot is itself the is_known_id feature.
//
// 3. TIME IS uint32 MICROSECONDS, NOT float SECONDS. The Python side carries
//    Unix timestamps around 1.478e9 in float64. A 32-bit float at that
//    magnitude has a spacing of about 128 seconds, so every inter-frame gap
//    would round to exactly zero and half the features would be dead. On device
//    we use esp_timer / micros(), a relative microsecond counter. Unsigned
//    subtraction also handles the ~71 minute wraparound correctly for free:
//    (uint32_t)(now - then) is the true elapsed time even across the rollover.

#pragma once

#include <stdint.h>
#include <string.h>

#include "constants_data.h"  // SC_KNOWN_IDS, SC_N_KNOWN_IDS, the clip ceilings

// ---------------------------------------------------------------------------
// layout -- must match FEATURE_NAMES in features.py, in order
// ---------------------------------------------------------------------------
//   0  can_id_norm      6  byte3        12  dt_same_id
//   1  is_known_id      7  byte4        13  period_ratio
//   2  dlc_norm         8  byte5        14  payload_change
//   3  byte0            9  byte6        15  bus_rate
//   4  byte1           10  byte7
//   5  byte2           11  dt_prev

#define SC_N_FEATURES 16

static constexpr float SC_CAN_ID_MAX = 2047.0f;  // 0x7FF, 11-bit identifier
static constexpr uint32_t SC_BUS_RATE_WINDOW_US = 10000;  // 10 ms

// Defaults when there is no history for an ID -- first sighting, or the ID is
// unknown and deliberately not stored. Chosen to look unremarkable so the model
// leans on is_known_id rather than on a magic number.
static constexpr float SC_DEFAULT_DT_SAME_S = 1.0f;
static constexpr float SC_DEFAULT_PERIOD_RATIO = 1.0f;
static constexpr uint8_t SC_DEFAULT_PAYLOAD_CHANGE = 0;

// Size the ring for the worst case we must survive. Normal Hyundai Sonata
// traffic already runs at ~25-40 frames per 10 ms window, and a DoS flood at one
// frame every 0.3 ms adds ~34 more, so ~75 is the realistic peak. 128 gives
// headroom and keeps the wraparound a cheap bitmask (512 bytes of RAM).
//
// If it ever did overflow, the count would stick at 128 -- but bus_rate clips at
// SC_BUS_RATE_CLIP (30), so the feature saturates at 1.0 either way and parity
// with Python's unbounded list still holds.
#define SC_RING_SIZE 128

// ---------------------------------------------------------------------------
// bit counting
// ---------------------------------------------------------------------------

// Mirrors _POPCOUNT8 in features.py. A table beats __builtin_popcount here:
// Xtensa has no popcount instruction, so the builtin becomes a function call.
static const uint8_t SC_POPCOUNT8[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
};

// ---------------------------------------------------------------------------

struct ScFrame {
  uint32_t t_us;   // relative microseconds, from micros() / esp_timer_get_time()
  uint16_t id;     // 11-bit CAN identifier
  uint8_t dlc;     // 0..8
  uint8_t data[8];
};

class ScFeatureExtractor {
 public:
  ScFeatureExtractor() { reset(); }

  void reset() {
    memset(slots_, 0, sizeof(slots_));
    memset(ring_, 0, sizeof(ring_));
    ring_head_ = 0;
    ring_count_ = 0;
    have_prev_any_ = false;
    last_t_any_ = 0;
  }

  // Feed one frame, fill out[16]. This is the whole hot path.
  void push(const ScFrame& f, float* out) {
    // --- fields that need no history ---------------------------------------
    out[0] = (float)f.id / SC_CAN_ID_MAX;

    const int slot = findSlot(f.id);     // -1 == unknown ID, no table entry
    out[1] = (slot >= 0) ? 1.0f : 0.0f;

    const uint8_t dlc = f.dlc > 8 ? 8 : f.dlc;
    out[2] = (float)dlc / 8.0f;

    for (int i = 0; i < 8; ++i) {
      out[3 + i] = (float)f.data[i] / 255.0f;
    }

    // --- time since the previous frame on the bus --------------------------
    // The very first frame has no predecessor. Python seeds dt_prev with the
    // clip ceiling there (i.e. "a long time"), so we must too.
    float dt_prev_s;
    if (!have_prev_any_) {
      dt_prev_s = SC_DT_PREV_CLIP;
    } else {
      dt_prev_s = (float)(uint32_t)(f.t_us - last_t_any_) * 1e-6f;
    }
    out[11] = clip01(dt_prev_s, SC_DT_PREV_CLIP);

    // --- per-ID history ----------------------------------------------------
    float dt_same_s = SC_DEFAULT_DT_SAME_S;
    uint8_t change = SC_DEFAULT_PAYLOAD_CHANGE;

    if (slot >= 0 && slots_[slot].seen) {
      dt_same_s = (float)(uint32_t)(f.t_us - slots_[slot].last_t_us) * 1e-6f;
      uint8_t bits = 0;
      for (int i = 0; i < 8; ++i) {
        bits += SC_POPCOUNT8[slots_[slot].last_data[i] ^ f.data[i]];
      }
      change = bits;
    }

    out[12] = clip01(dt_same_s, SC_DT_SAME_CLIP);

    // period_ratio: how this gap compares with how often the ID *should*
    // arrive. A spoofed ID shows up ~10x too often, pushing the ratio to ~0.1.
    // This is the feature that catches gear and RPM spoofing, where every
    // individual frame is a perfectly legal CAN frame.
    float ratio = SC_DEFAULT_PERIOD_RATIO;
    if (slot >= 0 && SC_KNOWN_IDS[slot].period > 0.0f) {
      ratio = dt_same_s / SC_KNOWN_IDS[slot].period;
    }
    out[13] = clip01(ratio, SC_PERIOD_RATIO_CLIP);

    out[14] = (float)change / 64.0f;   // 8 bytes = 64 bits maximum

    // --- bus load ----------------------------------------------------------
    // Push, evict anything older than the window, then count. The -1 excludes
    // the current frame, matching `len(self.recent) - 1` in Python.
    ringPush(f.t_us);
    ringEvict(f.t_us);
    out[15] = clip01((float)(ring_count_ - 1), SC_BUS_RATE_CLIP);

    // --- update state, AFTER the features are computed ---------------------
    last_t_any_ = f.t_us;
    have_prev_any_ = true;

    if (slot >= 0) {
      slots_[slot].last_t_us = f.t_us;
      memcpy(slots_[slot].last_data, f.data, 8);
      slots_[slot].seen = true;
    }
    // Unknown ID: nothing stored. Fixed memory, and the Fuzzy attack cannot
    // make us allocate.
  }

  // Binary search of the sorted known-ID table. log2(38) ~ 6 comparisons.
  // Public because main.cpp prints whether an ID was recognised.
  static int findSlot(uint16_t id) {
    int lo = 0, hi = SC_N_KNOWN_IDS - 1;
    while (lo <= hi) {
      const int mid = (lo + hi) >> 1;
      const uint16_t v = SC_KNOWN_IDS[mid].id;
      if (v == id) return mid;
      if (v < id) lo = mid + 1;
      else hi = mid - 1;
    }
    return -1;
  }

 private:
  struct Slot {
    uint32_t last_t_us;
    uint8_t last_data[8];
    bool seen;
  };

  // clamp to [0, ceiling] then scale to [0, 1]. Everything the model sees is in
  // unit range, which is what makes INT8 quantisation keep its resolution.
  static inline float clip01(float v, float ceiling) {
    if (v < 0.0f) v = 0.0f;
    if (v > ceiling) v = ceiling;
    return v / ceiling;
  }

  void ringPush(uint32_t t_us) {
    ring_[ring_head_] = t_us;
    ring_head_ = (ring_head_ + 1) & (SC_RING_SIZE - 1);
    if (ring_count_ < SC_RING_SIZE) ring_count_++;
  }

  // Drop timestamps strictly older than (now - window). Python keeps a
  // timestamp exactly on the boundary, so the test here is `<`, not `<=`.
  void ringEvict(uint32_t now_us) {
    while (ring_count_ > 0) {
      const int tail = (ring_head_ - ring_count_ + SC_RING_SIZE)
                       & (SC_RING_SIZE - 1);
      // Unsigned age, so this is still correct across the uint32 rollover.
      if ((uint32_t)(now_us - ring_[tail]) > SC_BUS_RATE_WINDOW_US) {
        ring_count_--;
      } else {
        break;
      }
    }
  }

  Slot slots_[SC_N_KNOWN_IDS];
  uint32_t ring_[SC_RING_SIZE];
  int ring_head_;
  int ring_count_;
  uint32_t last_t_any_;
  bool have_prev_any_;
};
