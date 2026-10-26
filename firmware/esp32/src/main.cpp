// SecureCAN AI -- ESP32 intrusion detection firmware.
//
// Reads CAN frames, extracts 16 features, runs a quantised INT8 neural network,
// and raises an alert. Everything on the device, no cloud, no network.
//
// WHY ON THE DEVICE AT ALL?
// A CAN bus at 500 kbit/s carries about 4,000 frames per second, which gives us
// ~250 us per frame. A round trip to a server is tens of milliseconds at best
// and needs connectivity a parked car may not have. Either the decision is local
// or it is too late to matter.
//
// WHAT IT DOES NOT DO: it never blocks or drops a frame. A false positive that
// suppressed a real brake or airbag message would be far more dangerous than the
// attack we were trying to stop. This is an intrusion DETECTION system: it
// reports, a human or a higher-level system decides.
//
// WHERE THE FRAMES COME FROM
// Wokwi has no CAN transceiver, so by default we replay test_frames.h -- real
// frames exported from the HCRL test split by tools/export_test_frames.py, with
// their ground-truth labels, so the serial output can be checked rather than
// just admired. Define SC_USE_TWAI to read a real bus through the ESP32's TWAI
// controller instead (TWAI is Espressif's name for their CAN peripheral; they
// avoid the "CAN" trademark).
//
// Build:  pio run -e esp32dev
// Wokwi:  open firmware/esp32/diagram.json, or `pio run -t upload` with wokwi-cli

// WHY THERE IS NO TENSORFLOW LITE MICRO HERE
// The plan was TFLite Micro. Its only ESP32 Arduino packaging is
// `TensorFlowLite_ESP32`, last released in 2021, and it fails three ways against
// the current platform (espressif32 8.x / ESP-IDF 5.5): two bundled LCD drivers
// call IDF functions that no longer exist, and its vendored flatbuffers assigns
// to a const member in `span`. The usual workaround is to pin the toolchain back
// two years, which makes the repo harder to build for anyone cloning it and
// still leaves us on an unmaintained library.
//
// This model is three dense layers and a softmax -- 1,157 parameters. The int8
// forward pass for that is seventy lines of integer arithmetic, so nn.h does it
// directly: no dependency, 1,740 bytes of weights instead of a 4,968-byte
// flatbuffer plus a parser and generic kernels. tests/test_parity.py checks the
// result against TFLite's own int8 interpreter on real frames and requires the
// predicted class to match on every one. See tools/export_model.py.

#include <Arduino.h>
#include <esp_cpu.h>   // esp_cpu_get_cycle_count -- see classify()

#include "features.h"
#include "nn.h"
#include "test_frames.h"

// ---------------------------------------------------------------------------
// board wiring -- see diagram.json
// ---------------------------------------------------------------------------
static constexpr int PIN_LED_OK = 2;     // green: bus looks clean
static constexpr int PIN_LED_ALERT = 4;  // red:   attack detected

static const char* CLASS_NAMES[SC_N_CLASSES] = {"Normal", "DoS", "Fuzzy",
                                                "Gear", "RPM"};

// The only way the feature extractor and the network can disagree is on the
// width of the vector between them, and that is knowable at compile time -- so
// check it there rather than at boot. A mismatch here means features.py gained
// or lost a feature and the model was not retrained; the old code would have
// read past the end of the input buffer and classified garbage confidently.
static_assert(SC_L0_IN == SC_N_FEATURES,
              "model input width != SC_N_FEATURES -- re-run quantize.py and "
              "tools/export_model.py after changing features.py");
static_assert(SC_L2_OUT == SC_N_CLASSES, "model output width != class count");

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------
//
// Every buffer in this firmware is a compile-time sized static: the per-ID table
// in features.h and the three activation arrays in nn.h. Nothing is allocated at
// runtime, so the worst-case RAM figure printed at boot is the real one and
// cannot drift. A detector that could fail to allocate mid-drive would be worse
// than useless.
static ScFeatureExtractor g_extractor;

// running tally, printed every SC_REPORT_EVERY frames
static uint32_t g_n = 0;
static uint32_t g_alerts = 0;
static uint32_t g_true_pos = 0, g_false_pos = 0, g_false_neg = 0, g_true_neg = 0;
static uint64_t g_total_us = 0;
static uint32_t g_max_us = 0;
static uint64_t g_total_feat_cyc = 0, g_total_infer_cyc = 0;
static uint32_t g_max_cyc = 0;

static constexpr uint32_t SC_REPORT_EVERY = 500;

// ---------------------------------------------------------------------------

static void reportModel() {
  Serial.printf("  classifier   16-32-16-5 MLP, int8, hand-rolled kernel\n");
  Serial.printf("  weights      %lu bytes in flash\n",
                (unsigned long)sc_nn_flash_bytes());
  Serial.printf("  activations  %lu bytes in RAM (static, no malloc)\n",
                (unsigned long)sc_nn_ram_bytes());
  Serial.printf("  in  quant    scale %.8f  zero %ld\n",
                SC_IN_SCALE, (long)SC_IN_ZERO);
  Serial.printf("  logit quant  scale %.8f  zero %ld\n",
                SC_LOGIT_SCALE, (long)SC_LOGIT_ZERO);
}

// Run one frame end to end. Returns the predicted class, writes the confidence.
// `elapsed_us` covers feature extraction AND inference, because that is the
// budget that has to fit in the 250 us between frames -- quoting inference
// alone would be flattering and wrong.
// Timing is taken from the CPU cycle counter (CCOUNT), not micros().
//
// This matters, and the first Wokwi run is why. micros() reported a mean of
// 1751 us per frame -- 7x over the 250 us budget, and about 363 cycles per
// multiply-accumulate, which is not a believable figure for 1,157 MACs on a
// 240 MHz core. Under the simulator micros() is derived from the host's clock,
// so when the simulation runs slower than real time every interval it reports
// is stretched by however far behind it is. It measures the laptop, not the
// chip.
//
// CCOUNT is a counter inside the simulated CPU that advances one tick per
// simulated cycle, so it is independent of how fast the host happens to be
// running. Dividing by the clock frequency gives the time the same code would
// take on silicon. We report both, so the gap between them is visible rather
// than something you have to take on trust.
//
// Features and inference are timed separately because they are different kinds
// of work -- a table lookup and a ring-buffer scan versus 1,157 integer MACs --
// and if the total ever misses budget, the split says which half to attack.
static int classify(const ScFrame& f, float* confidence,
                    uint32_t* elapsed_us, uint32_t* feat_cycles,
                    uint32_t* infer_cycles) {
  const uint32_t t0 = micros();
  const uint32_t c0 = esp_cpu_get_cycle_count();

  float feats[SC_N_FEATURES];
  g_extractor.push(f, feats);

  const uint32_t c1 = esp_cpu_get_cycle_count();
  const int best = sc_nn_predict(feats, confidence);
  const uint32_t c2 = esp_cpu_get_cycle_count();

  // Unsigned subtraction is correct across the 32-bit wraparound, which at
  // 240 MHz happens every ~17.9 seconds -- well inside a 1,200 frame replay.
  *feat_cycles = c1 - c0;
  *infer_cycles = c2 - c1;
  *elapsed_us = micros() - t0;
  return best;
}

static void report() {
  const uint32_t atk_true = g_true_pos + g_false_neg;
  const uint32_t norm_true = g_true_neg + g_false_pos;

  Serial.println();
  Serial.printf("--- after %lu frames ---------------------------------\n",
                (unsigned long)g_n);
  Serial.printf("  alerts raised   %lu\n", (unsigned long)g_alerts);
  Serial.printf("  vs ground truth %lu attack / %lu normal frames\n",
                (unsigned long)atk_true, (unsigned long)norm_true);
  if (atk_true) {
    Serial.printf("  detected        %lu  (recall %.2f%%)\n",
                  (unsigned long)g_true_pos,
                  100.0f * g_true_pos / atk_true);
    Serial.printf("  missed          %lu\n", (unsigned long)g_false_neg);
  }
  if (norm_true) {
    Serial.printf("  false alarms    %lu  (FPR %.4f%%)\n",
                  (unsigned long)g_false_pos,
                  100.0f * g_false_pos / norm_true);
  }
  const double n = g_n ? g_n : 1;
  const double mhz = (double)getCpuFrequencyMhz();
  const double feat_cyc = g_total_feat_cyc / n;
  const double infer_cyc = g_total_infer_cyc / n;

  Serial.println();
  Serial.printf("  per frame, measured with the CPU cycle counter at %.0f MHz:\n",
                mhz);
  Serial.printf("    features    %8.0f cycles  = %7.2f us\n",
                feat_cyc, feat_cyc / mhz);
  Serial.printf("    inference   %8.0f cycles  = %7.2f us\n",
                infer_cyc, infer_cyc / mhz);
  Serial.printf("    TOTAL       %8.0f cycles  = %7.2f us   "
                "(budget 250 us/frame at 500 kbit/s)\n",
                feat_cyc + infer_cyc, (feat_cyc + infer_cyc) / mhz);
  Serial.printf("    worst frame %8lu cycles  = %7.2f us\n",
                (unsigned long)g_max_cyc, g_max_cyc / mhz);
  Serial.printf("    headroom    %.1fx\n",
                250.0 / ((feat_cyc + infer_cyc) / mhz));
#ifdef SC_NN_PROFILE
  Serial.println("  where the inference cycles go (last frame):");
  Serial.printf("    quantise in %8lu\n", (unsigned long)sc_prof_quant);
  Serial.printf("    layer0      %8lu   (16->32)\n", (unsigned long)sc_prof_l0);
  Serial.printf("    layer1      %8lu   (32->16)\n", (unsigned long)sc_prof_l1);
  Serial.printf("    layer2      %8lu   (16->5)\n", (unsigned long)sc_prof_l2);
  Serial.printf("    argmax+soft %8lu\n", (unsigned long)sc_prof_soft);
#endif
  Serial.printf("  micros() agrees: mean %.1f us, max %lu us\n",
                (double)g_total_us / n, (unsigned long)g_max_us);
  Serial.println();
  Serial.println("  READ THE CALIBRATION ABOVE BEFORE QUOTING ANY OF THIS.");
  Serial.println("  Wokwi's ESP32 is not instruction-accurate: a dependent");
  Serial.println("  integer add that costs 4-6 cycles on silicon measures ~105");
  Serial.println("  here, so compute-bound cycle counts come out roughly 20x");
  Serial.println("  too high. The clock is honest; the CPU model is slow, and");
  Serial.println("  the two clocks agree only because both track wall time.");
  Serial.println("  So these figures establish CORRECTNESS, not latency --");
  Serial.println("  real silicon is needed for a latency number.");
  Serial.println("-----------------------------------------------------");
  Serial.println();
}

// ---------------------------------------------------------------------------

void setup() {
  Serial.begin(115200);
  delay(1500);  // let the USB/serial monitor attach before we print

  pinMode(PIN_LED_OK, OUTPUT);
  pinMode(PIN_LED_ALERT, OUTPUT);
  digitalWrite(PIN_LED_OK, HIGH);
  digitalWrite(PIN_LED_ALERT, LOW);

  Serial.println();
  Serial.println("=====================================================");
  Serial.println(" SecureCAN AI -- on-device CAN intrusion detection");
  Serial.println("=====================================================");
  // getCpuFrequencyMhz() returns uint32_t, not int, so it needs %lu and an
  // explicit cast -- printf has no type information and would otherwise read the
  // wrong number of bytes off the varargs stack for every argument after it.
  Serial.printf("  chip         %s rev %d, %lu MHz, %d core(s)\n",
                ESP.getChipModel(), ESP.getChipRevision(),
                (unsigned long)getCpuFrequencyMhz(), ESP.getChipCores());
  Serial.printf("  free heap    %lu bytes\n", (unsigned long)ESP.getFreeHeap());
  Serial.printf("  known IDs    %d (learned from normal traffic)\n",
                SC_N_KNOWN_IDS);
  Serial.printf("  features     %d per frame\n", SC_N_FEATURES);

  reportModel();

  // Is the clock we are about to measure with trustworthy?
  //
  // Never report a number from an instrument you have not checked. delay(100)
  // is 100 ms of wall time by definition, and at 240 MHz that is exactly
  // 24,000,000 CPU cycles. If CCOUNT says something far from that, then under
  // this simulator CCOUNT is not counting instructions -- it is being derived
  // from the host clock, and every per-frame figure below is measuring the
  // laptop rather than the chip.
  //
  // The ratio it prints is the correction factor, so the honest device number
  // can still be recovered from a distorted one.
  {
    const uint32_t c0 = esp_cpu_get_cycle_count();
    const uint32_t m0 = micros();
    delay(100);
    const uint32_t cyc = esp_cpu_get_cycle_count() - c0;
    const uint32_t us = micros() - m0;
    const double expected = 100000.0 * (double)getCpuFrequencyMhz();
    Serial.println();
    Serial.println("  --- clock sanity check: delay(100) ---");
    Serial.printf("    micros() elapsed   %lu us (expect ~100000)\n",
                  (unsigned long)us);
    Serial.printf("    CCOUNT elapsed     %lu cycles (expect %.0f)\n",
                  (unsigned long)cyc, expected);
    Serial.printf("    ratio measured/expected  %.3f\n", cyc / expected);
    Serial.println("    ratio ~1.0 => the counter tracks wall time correctly");
  }

  // The delay() check above only proves CCOUNT tracks wall time. It says
  // nothing about how fast the CPU actually RETIRES INSTRUCTIONS, because
  // delay() is just waiting. So measure that separately with a loop whose cost
  // is known.
  //
  // The body is a dependent integer add, so it cannot be vectorised or
  // reordered away, and `volatile` stops the whole loop being folded to a
  // constant. On real silicon this is a handful of cycles per iteration. If
  // the simulator charges far more, then every per-frame cycle figure in this
  // firmware is inflated by that same factor and must be divided out before it
  // means anything about real hardware.
  {
    volatile int32_t sink = 0;
    const int32_t N = 100000;
    const uint32_t c0 = esp_cpu_get_cycle_count();
    int32_t acc = 0;
    for (int32_t i = 0; i < N; ++i) acc += i ^ (acc >> 3);
    const uint32_t cyc = esp_cpu_get_cycle_count() - c0;
    sink = acc;
    (void)sink;
    Serial.println();
    Serial.println("  --- instruction-rate check: 100k dependent int adds ---");
    Serial.printf("    cycles             %lu\n", (unsigned long)cyc);
    Serial.printf("    cycles/iteration   %.2f  (real silicon: ~4-6)\n",
                  (double)cyc / N);
    Serial.println("    >> 10 means the simulated CPU is far slower than the");
    Serial.println("       chip, and per-frame cycle counts are inflated.");
  }

  Serial.printf("  test frames  %d (real HCRL frames, with labels)\n",
                SC_N_TEST_FRAMES);
  Serial.println("=====================================================");
  Serial.println();
  Serial.println("   t_us      ID DLC  payload                  verdict"
                 "    conf   us  truth");
  Serial.println("  ------------------------------------------------------"
                 "-----------------");
}

void loop() {
  static int idx = 0;

  if (idx >= SC_N_TEST_FRAMES) {
    report();
    Serial.println("replay complete -- halting");
    digitalWrite(PIN_LED_ALERT, LOW);
    digitalWrite(PIN_LED_OK, HIGH);
    while (true) delay(1000);
  }

  // Replay the exported frame. We use its recorded relative timestamp rather
  // than micros(), so the timing features see exactly the intervals the model
  // was trained on -- the Wokwi loop runs far slower than a real 500 kbit/s bus
  // and using wall-clock here would make every gap look enormous.
  ScFrame f;
  f.t_us = SC_TEST_T_US[idx];
  f.id = SC_TEST_ID[idx];
  f.dlc = SC_TEST_DLC[idx];
  memcpy(f.data, SC_TEST_DATA[idx], 8);

  const uint8_t truth = SC_TEST_LABEL[idx];

  float conf = 0.0f;
  uint32_t us = 0, feat_cyc = 0, infer_cyc = 0;
  const int cls = classify(f, &conf, &us, &feat_cyc, &infer_cyc);

  g_n++;
  g_total_us += us;
  if (us > g_max_us) g_max_us = us;
  g_total_feat_cyc += feat_cyc;
  g_total_infer_cyc += infer_cyc;
  if (feat_cyc + infer_cyc > g_max_cyc) g_max_cyc = feat_cyc + infer_cyc;

  const bool alert = (cls != 0);
  const bool is_attack = (truth != 0);
  if (alert) g_alerts++;
  if (alert && is_attack) g_true_pos++;
  else if (alert && !is_attack) g_false_pos++;
  else if (!alert && is_attack) g_false_neg++;
  else g_true_neg++;

  digitalWrite(PIN_LED_ALERT, alert ? HIGH : LOW);
  digitalWrite(PIN_LED_OK, alert ? LOW : HIGH);

  // Print every alert, but only every 25th normal frame -- otherwise the
  // interesting lines scroll past in a wall of "Normal".
  if (alert || (idx % 25) == 0) {
    char pay[25];
    const int n = f.dlc > 8 ? 8 : f.dlc;
    int o = 0;
    for (int i = 0; i < n && o < 24; ++i) {
      o += snprintf(pay + o, sizeof(pay) - o, "%02X ", f.data[i]);
    }
    pay[o > 0 ? o - 1 : 0] = '\0';

    const char* mark = "";
    if (alert && !is_attack) mark = "  <-- FALSE ALARM";
    else if (!alert && is_attack) mark = "  <-- MISSED";
    else if (alert && cls != truth) mark = "  <-- wrong class";

    Serial.printf("  %7lu  0x%03X  %d   %-24s %-9s %5.2f %4lu  %-7s%s\n",
                  (unsigned long)f.t_us, f.id, f.dlc, pay,
                  alert ? CLASS_NAMES[cls] : "ok",
                  conf, (unsigned long)us, CLASS_NAMES[truth], mark);
  }

  if (g_n % SC_REPORT_EVERY == 0) report();

  idx++;
}
