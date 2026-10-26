// SecureCAN AI -- INT8 inference for the 16-32-16-5 classifier.
//
// This replaces TensorFlow Lite Micro. See tools/export_model.py for the full
// reason; the short version is that the only ESP32 Arduino packaging of TFLite
// Micro is a 2021 snapshot that does not compile against ESP-IDF 5, and this
// model is three dense layers, so running it by hand costs seventy lines and
// removes the dependency entirely.
//
// tests/test_parity.py runs the same frames through this code and through
// TFLite's own int8 interpreter and requires the predicted class to agree on
// every single frame. Hand-writing inference is only defensible if you then
// prove it matches the reference, so that test is part of the design.
//
//
// HOW INT8 INFERENCE ACTUALLY WORKS
// ---------------------------------
// A quantized tensor stores small integers and a rule for reading them as real
// numbers:
//
//     real = scale * (quantized - zero_point)
//
// `scale` is a float chosen once, at export time, from the observed range of
// that tensor. `zero_point` is the integer that means exactly 0.0, which matters
// because 0 has to be representable exactly -- padding and ReLU depend on it.
//
// For one output channel j of a dense layer, the float math is
//
//     y_j = sum_i (x_i * w_ji) + b_j
//
// Substituting real = s*(q - z) for x and w, and using the fact that our weights
// are symmetric (zero_point 0, so real_w = s_w * q_w):
//
//     y_j = s_x * s_wj * sum_i (q_x[i] - z_x) * q_w[j][i]  +  b_j
//           ^^^^^^^^^^^           ^^^^^^^^^^^^^^^^^^^^^^^
//           a float constant      a pure integer dot product
//
// So the whole layer is an integer dot product, and every float disappears into
// one per-channel constant. That integer sum needs more than 8 bits -- 16 terms
// of (at most 255 * 127) needs 23 bits, and the next layer's 32 terms needs 24 --
// so the accumulator is int32. That is the entire trick: int8 storage, int32
// arithmetic.
//
// The accumulator sits at scale (s_x * s_wj). The bias is pre-divided by that
// same scale at export time and stored as int32, which is why it can simply be
// added. To hand the result to the next layer it must be rescaled to that
// layer's own scale, which means multiplying by
//
//     M_j = (s_x * s_wj) / s_out
//
// a fraction below 1. Doing that in floating point on a microcontroller would
// defeat the point, so M_j was split on the host into an int32 multiplier and a
// shift, and we apply it with integer arithmetic only.
//
// Why bother with any of this? The float model is 6.6 KB and needs an FPU for
// every one of 1,157 multiply-accumulates. The int8 model is 1.7 KB of weights
// and runs on the integer unit. On this dataset it costs +0.002 percentage
// points of accuracy -- measured, see results/quantization.json.

#pragma once

#include <math.h>
#include <stdint.h>

#include "model_int8.h"

// ---------------------------------------------------------------------------
// fixed-point rescaling
//
// These two helpers are TFLite's MultiplyByQuantizedMultiplier, reimplemented.
// They are copied faithfully rather than simplified because the exported
// multipliers were computed to match them -- a "cleaner" rounding rule here
// would produce off-by-one differences against the reference interpreter, and
// the parity test would (correctly) fail.
// ---------------------------------------------------------------------------

// round((a * b) / 2^31) with saturation. The doubling is where the name comes
// from: a*b has 62 significant bits and we keep the top 31, which is why the
// exported multiplier is normalised into [2^30, 2^31) -- it spends the full
// available precision instead of throwing high bits away.
static inline int32_t sc_doubling_high_mul(int32_t a, int32_t b) {
  if (a == INT32_MIN && b == INT32_MIN) return INT32_MAX;  // the one overflow
  const int64_t ab = (int64_t)a * (int64_t)b;
  // Round half away from zero, which needs the nudge to follow the sign.
  const int64_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
  return (int32_t)((ab + nudge) / ((int64_t)1 << 31));
}

// Divide by 2^exponent, rounding half away from zero. A bare >> would round
// toward negative infinity and bias every negative activation downward.
static inline int32_t sc_rounding_shift(int32_t x, int exponent) {
  if (exponent <= 0) return x;
  const int32_t mask = ((int32_t)1 << exponent) - 1;
  const int32_t remainder = x & mask;
  const int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
  return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

static inline int32_t sc_requantize(int32_t acc, int32_t multiplier, int32_t shift) {
  const int left = shift > 0 ? shift : 0;
  const int right = shift > 0 ? 0 : -shift;
  return sc_rounding_shift(sc_doubling_high_mul(acc * ((int32_t)1 << left),
                                                multiplier),
                           right);
}

static inline int8_t sc_clamp_i8(int32_t v, int32_t lo, int32_t hi) {
  if (v < lo) return (int8_t)lo;
  if (v > hi) return (int8_t)hi;
  return (int8_t)v;
}

// ---------------------------------------------------------------------------
// one fully-connected layer
// ---------------------------------------------------------------------------

// `w` is row-major [out_dim][in_dim]: row j holds the weights for output j, so
// the dot product walks contiguous memory. On a cache-less MCU that is the
// difference between streaming flash and thrashing it.
static inline void sc_dense(const int8_t* in, int in_dim,
                            int8_t* out, int out_dim,
                            const int8_t* w, const int32_t* b,
                            const int32_t* mult, const int32_t* shift,
                            int32_t in_zero, int32_t out_zero,
                            int32_t act_min, int32_t act_max) {
  for (int j = 0; j < out_dim; ++j) {
    const int8_t* wrow = w + (size_t)j * in_dim;
    int32_t acc = b[j];
    for (int i = 0; i < in_dim; ++i) {
      acc += ((int32_t)in[i] - in_zero) * (int32_t)wrow[i];
    }
    acc = sc_requantize(acc, mult[j], shift[j]);
    out[j] = sc_clamp_i8(acc + out_zero, act_min, act_max);
  }
}

// ---------------------------------------------------------------------------
// the whole network
// ---------------------------------------------------------------------------

// Scratch buffers for the two hidden activations. Static, file-scope and
// compile-time sized: there is no malloc anywhere in this firmware, so the
// worst-case RAM use is known from the linker map rather than hoped for at
// runtime. 48 bytes total.
static int8_t sc_nn_h0[SC_L0_OUT];
static int8_t sc_nn_h1[SC_L1_OUT];
static int8_t sc_nn_logits[SC_L2_OUT];
static int8_t sc_nn_input[SC_L0_IN];

// Run one frame. Returns the predicted class index; writes the softmax
// probability of that class to `confidence` if non-null.
//
// `feats` holds the 16 features from features.h, already min-max scaled to
// [0, 1] with the training-set constants.
static inline int sc_nn_predict(const float* feats, float* confidence) {
  // Quantise the input. lrintf rounds half-to-even in the default FP mode,
  // matching numpy's rint, which is what the exporter assumed when it chose the
  // input scale -- so a feature exactly between two int8 steps lands on the
  // same side here as it did on the host.
  for (int i = 0; i < SC_L0_IN; ++i) {
    const int32_t q = (int32_t)lrintf(feats[i] / SC_IN_SCALE) + SC_IN_ZERO;
    sc_nn_input[i] = sc_clamp_i8(q, -128, 127);
  }

  sc_dense(sc_nn_input, SC_L0_IN, sc_nn_h0, SC_L0_OUT,
           SC_L0_W, SC_L0_B, SC_L0_M, SC_L0_SHIFT,
           SC_L0_IN_ZERO, SC_L0_OUT_ZERO, SC_L0_ACT_MIN, SC_L0_ACT_MAX);

  sc_dense(sc_nn_h0, SC_L1_IN, sc_nn_h1, SC_L1_OUT,
           SC_L1_W, SC_L1_B, SC_L1_M, SC_L1_SHIFT,
           SC_L1_IN_ZERO, SC_L1_OUT_ZERO, SC_L1_ACT_MIN, SC_L1_ACT_MAX);

  sc_dense(sc_nn_h1, SC_L2_IN, sc_nn_logits, SC_L2_OUT,
           SC_L2_W, SC_L2_B, SC_L2_M, SC_L2_SHIFT,
           SC_L2_IN_ZERO, SC_L2_OUT_ZERO, SC_L2_ACT_MIN, SC_L2_ACT_MAX);

  // The class is the argmax of the int8 logits. No softmax needed for the
  // decision: softmax is strictly increasing, so it cannot reorder anything.
  // Skipping it on the hot path also means the integer pipeline is the whole
  // pipeline -- no exp(), no float, nothing that could stall.
  int best = 0;
  for (int k = 1; k < SC_L2_OUT; ++k) {
    if (sc_nn_logits[k] > sc_nn_logits[best]) best = k;
  }

  // The confidence is only for the human reading the serial log, so it is worth
  // five expf() calls. Dequantise the logits, then softmax with the max
  // subtracted off -- without that, expf of a large logit overflows to inf and
  // the ratio comes out NaN.
  if (confidence) {
    float mx = -1e30f, sum = 0.0f, num = 0.0f;
    float logits[SC_L2_OUT];
    for (int k = 0; k < SC_L2_OUT; ++k) {
      logits[k] = SC_LOGIT_SCALE * (float)((int32_t)sc_nn_logits[k] - SC_LOGIT_ZERO);
      if (logits[k] > mx) mx = logits[k];
    }
    for (int k = 0; k < SC_L2_OUT; ++k) {
      const float e = expf(logits[k] - mx);
      sum += e;
      if (k == best) num = e;
    }
    *confidence = sum > 0.0f ? num / sum : 0.0f;
  }

  return best;
}

// Bytes of weights, biases and requantisation constants in flash. Reported at
// boot so the number in the README comes from the build, not from a guess.
static inline uint32_t sc_nn_flash_bytes(void) {
  return (uint32_t)(sizeof(SC_L0_W) + sizeof(SC_L0_B) + sizeof(SC_L0_M) +
                    sizeof(SC_L0_SHIFT) +
                    sizeof(SC_L1_W) + sizeof(SC_L1_B) + sizeof(SC_L1_M) +
                    sizeof(SC_L1_SHIFT) +
                    sizeof(SC_L2_W) + sizeof(SC_L2_B) + sizeof(SC_L2_M) +
                    sizeof(SC_L2_SHIFT));
}

// RAM held by the activations, for the same reason.
static inline uint32_t sc_nn_ram_bytes(void) {
  return (uint32_t)(sizeof(sc_nn_input) + sizeof(sc_nn_h0) +
                    sizeof(sc_nn_h1) + sizeof(sc_nn_logits));
}
