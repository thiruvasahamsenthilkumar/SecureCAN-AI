// SecureCAN AI - ESP32 CAN bus intrusion detector (Wokwi simulation)
//
// Replays scripted CAN traffic through the quantized random-forest detector
// and reports the predicted class plus per-window inference latency.

#include "detector.h"

static uint32_t sim_us = 0;
static uint32_t n_windows = 0, n_alerts = 0, lat_sum_us = 0;

// Each ECU keeps its own signal band, mirroring the training distribution.
static const uint16_t ECU_IDS[] = {0x0A0, 0x130, 0x18F, 0x260, 0x2C0, 0x316, 0x43F, 0x545};

static uint8_t rnd() {               // xorshift: deterministic across runs
    static uint32_t s = 2463534242u;
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return (uint8_t)(s >> 24);
}

static void push_normal() {
    can_msg_t m;
    sim_us += 10000 + (rnd() % 600) - 300;        // ~10 ms period
    m.timestamp_us = sim_us;
    m.can_id = ECU_IDS[rnd() % 8];
    m.dlc = 8;
    uint8_t base = (m.can_id & 0xFF) % 90;
    for (uint8_t k = 0; k < 8; k++) {
        int v = base + (int)(rnd() % 45) - 22;
        m.data[k] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
    window_push(&m);
}

static void push_dos() {
    can_msg_t m;
    sim_us += 300 + (rnd() % 600);                // 0.3-0.9 ms: bus flood
    m.timestamp_us = sim_us;
    m.can_id = 0x000;
    m.dlc = 8;
    for (uint8_t k = 0; k < 8; k++) m.data[k] = rnd() % 40;
    window_push(&m);
}

static void push_fuzzy() {
    can_msg_t m;
    sim_us += 10000 + (rnd() % 8000) - 4000;
    m.timestamp_us = sim_us;
    m.can_id = ((uint16_t)rnd() << 3) | (rnd() & 0x7);   // arbitrary ID
    m.dlc = rnd() % 9;
    for (uint8_t k = 0; k < 8; k++)
        m.data[k] = (rnd() % 10 < 4) ? rnd() : (uint8_t)(90 + (rnd() % 60) - 30);
    window_push(&m);
}

static void push_fake_rpm() {
    can_msg_t m;
    sim_us += 10000 + (rnd() % 300) - 150;
    m.timestamp_us = sim_us;
    m.can_id = 0x316;                             // engine ECU, spoofed payload
    m.dlc = 8;
    for (uint8_t k = 0; k < 8; k++) m.data[k] = 70 + (rnd() % 40) - 20;
    uint8_t rpm = (rnd() & 1) ? (200 + rnd() % 55) : (rnd() % 20);
    m.data[0] = rpm;
    m.data[1] = rpm;
    window_push(&m);
}

static void push_fake_gear() {
    can_msg_t m;
    sim_us += 10000 + (rnd() % 300) - 150;
    m.timestamp_us = sim_us;
    m.can_id = 0x43F;                             // transmission ECU
    m.dlc = 8;
    for (uint8_t k = 0; k < 8; k++) m.data[k] = 63 + (rnd() % 40) - 20;
    m.data[2] = 0x0D + (rnd() % 3);               // invalid gear nibble
    m.data[3] = 0xD0 + ((rnd() % 3) << 4);
    window_push(&m);
}

// Fill a window: benign background with `attack_pct` of injected frames.
static void build_window(void (*attack)(), uint8_t attack_pct) {
    window_reset();
    while (!window_full()) {
        if (attack && (rnd() % 100) < attack_pct) attack();
        else push_normal();
    }
}

static void run_scenario(const char *label, void (*attack)(), uint8_t pct) {
    build_window(attack, pct);

    float feats[N_FEATURES];
    uint32_t t0 = micros();
    extract_features(feats);
    uint8_t cls = classify(feats);
    uint32_t dt = micros() - t0;

    n_windows++;
    lat_sum_us += dt;
    if (cls != 0) n_alerts++;

    Serial.printf("%-18s -> %-10s | %4lu us | rate %7.0f msg/s | ids %2.0f | ent %.2f\n",
                  label, class_name(cls), (unsigned long)dt,
                  feats[0], feats[4], feats[11]);
}

void setup() {
    Serial.begin(115200);
    delay(1500);

    Serial.println("\n==============================================");
    Serial.println("  SecureCAN AI - CAN Bus Intrusion Detector");
    Serial.println("==============================================");
    Serial.printf("features : %d\n", N_FEATURES);
    Serial.printf("forest   : %d trees / %d nodes\n", N_TREES, N_NODES);
    Serial.printf("model RAM: %u bytes (int8 quantized)\n",
                  (unsigned)(N_NODES * (sizeof(int8_t) * 3 + sizeof(int16_t) * 2)));
    Serial.printf("free heap: %u bytes\n", (unsigned)ESP.getFreeHeap());
    Serial.println("----------------------------------------------");

    run_scenario("baseline",   nullptr,        0);
    run_scenario("baseline",   nullptr,        0);
    run_scenario("DoS flood",  push_dos,      70);
    run_scenario("DoS flood",  push_dos,      90);
    run_scenario("fuzzy inj",  push_fuzzy,    60);
    run_scenario("fake RPM",   push_fake_rpm, 60);
    run_scenario("fake gear",  push_fake_gear,60);
    run_scenario("baseline",   nullptr,        0);

    Serial.println("----------------------------------------------");
    Serial.printf("windows %lu | alerts %lu | avg latency %lu us\n",
                  (unsigned long)n_windows, (unsigned long)n_alerts,
                  (unsigned long)(lat_sum_us / n_windows));
    Serial.println("==============================================");
}

void loop() {
    delay(1000);
}
