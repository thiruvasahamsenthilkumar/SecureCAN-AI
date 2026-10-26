#!/usr/bin/env python3
"""Train the detector: two baselines, then the MLP that ships to the ESP32.

We train three things on purpose:

  1. A hand-written rule (bus_rate and period_ratio thresholds). This is the
     "did you try something simple first?" answer. If the rule already catches
     DoS perfectly -- and it should, DoS is loud -- then the model's DoS score
     proves nothing, and the honest claim is that the model earns its keep on
     the spoofing attacks.

  2. A random forest. Strong tabular baseline, tells us roughly what accuracy is
     achievable from these 16 features at all. Too big for the microcontroller.

  3. A 16 -> 32 -> 16 -> 5 MLP. Small enough to quantise to INT8 and run under
     TensorFlow Lite Micro, which is the actual deliverable.

Thresholds for the rule are tuned on VALIDATION, never on test.

    python -m securecan.train              # full run
    python -m securecan.train --rebuild    # re-extract features first
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # mute TF's startup banner

from . import dataset, features as F  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS = ROOT / "results"
CLASS_NAMES = ["Normal", "DoS", "Fuzzy", "Gear", "RPM"]

FI = {name: i for i, name in enumerate(F.FEATURE_NAMES)}


# ---------------------------------------------------------------------------
# baseline 1: hand-written rule
# ---------------------------------------------------------------------------

def rule_predict(X, bus_rate_thr, period_ratio_thr):
    """Binary attack/normal from two thresholds plus the known-ID check.

    Deliberately crude, and deliberately binary: the rule has no way to tell a
    gear spoof from an RPM spoof without being handed a list of which ID means
    what, which would be hand-coding the answer rather than detecting it.

      high bus rate            -> flooding      (DoS)
      ID never seen in normal  -> unknown ID    (Fuzzy)
      ID arriving far too soon -> replay/spoof  (Gear, RPM)
    """
    flood = X[:, FI["bus_rate"]] > bus_rate_thr
    unknown = X[:, FI["is_known_id"]] < 0.5
    too_soon = X[:, FI["period_ratio"]] < period_ratio_thr
    return (flood | unknown | too_soon).astype(np.int64)


def tune_rule(Xv, yv):
    """Grid-search the two thresholds on validation, maximising attack F1."""
    from sklearn.metrics import f1_score

    truth = (yv != 0).astype(np.int64)
    best = (0.0, None)

    for br in np.linspace(0.3, 1.0, 15):
        for pr in np.linspace(0.005, 0.25, 25):
            f1 = f1_score(truth, rule_predict(Xv, br, pr), zero_division=0)
            if f1 > best[0]:
                best = (f1, (float(br), float(pr)))

    print(f"  tuned on validation: bus_rate>{best[1][0]:.3f}  "
          f"period_ratio<{best[1][1]:.4f}  -> val attack F1 {best[0]:.4f}")
    return best[1]


# ---------------------------------------------------------------------------
# baseline 2: random forest
# ---------------------------------------------------------------------------

def train_forest(Xt, yt, max_rows=600_000, seed=0):
    from sklearn.ensemble import RandomForestClassifier

    # a forest on 3 M rows is slow and adds nothing over a large subsample
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(yt), size=min(max_rows, len(yt)), replace=False)

    clf = RandomForestClassifier(
        n_estimators=60, max_depth=14, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=seed,
    )
    t0 = time.perf_counter()
    clf.fit(Xt[idx], yt[idx])
    print(f"  fitted on {len(idx):,} rows in {time.perf_counter() - t0:.1f}s")
    return clf


# ---------------------------------------------------------------------------
# the real model: a small MLP
# ---------------------------------------------------------------------------

def build_mlp(n_features, n_classes):
    """16 -> 32 -> 16 -> 5.

    Kept tiny because it has to fit in TFLite Micro's arena on an ESP32 with a
    250 us/frame budget. Two hidden layers are enough: the features already do the
    hard work, so the network only has to learn boundaries between them, not
    rediscover timing structure from raw bytes.
    """
    import tensorflow as tf

    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])


def train_mlp(Xt, yt, Xv, yv, epochs=30, batch=4096, seed=0, quiet=False):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)

    # Attack frames are ~13% of the test data and the four classes are uneven
    # among themselves. Without weights the model can score 87% by calling
    # everything normal, which is exactly the failure mode we care least to ship.
    counts = np.bincount(yt, minlength=5).astype(np.float64)
    weights = {i: float(len(yt) / (5 * c)) if c else 0.0
               for i, c in enumerate(counts)}
    if not quiet:
        print("  class weights: " +
              "  ".join(f"{CLASS_NAMES[i]}={w:.2f}" for i, w in weights.items()))

    model = build_mlp(Xt.shape[1], 5)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    t0 = time.perf_counter()
    model.fit(
        Xt, yt,
        validation_data=(Xv, yv),
        epochs=epochs, batch_size=batch, class_weight=weights,
        verbose=0 if quiet else 2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
        ],
    )
    if not quiet:
        print(f"  trained in {time.perf_counter() - t0:.1f}s")
    return model


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="re-extract features instead of using the cache")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--normal-keep", type=float, default=0.15,
                    help="fraction of normal frames kept in train/val")
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)

    print("\n=== constants (training split, normal frames only) ===")
    const = dataset.fit_constants()
    const.save(MODELS / "constants.json")

    print("\n=== dataset ===")
    splits = dataset.load_or_build(const, rebuild=args.rebuild,
                                   normal_keep=args.normal_keep)
    (Xt, yt), (Xv, yv), (Xs, ys) = splits["train"], splits["val"], splits["test"]

    results = {}

    print("\n=== baseline 1: hand-written rule ===")
    thr = tune_rule(Xv, yv)
    results["rule_thresholds"] = {"bus_rate": thr[0], "period_ratio": thr[1]}
    np.save(MODELS / "_rule_test_pred.npy", rule_predict(Xs, *thr))

    print("\n=== baseline 2: random forest ===")
    forest = train_forest(Xt, yt)
    np.save(MODELS / "_forest_test_pred.npy", forest.predict(Xs))

    print("\n=== model: MLP 16-32-16-5 ===")
    model = train_mlp(Xt, yt, Xv, yv, epochs=args.epochs)
    model.save(MODELS / "mlp.keras")

    probs = model.predict(Xs, batch_size=8192, verbose=0)
    np.save(MODELS / "_mlp_test_pred.npy", probs.argmax(1))
    np.save(MODELS / "_mlp_test_prob.npy", probs.astype(np.float32))
    np.save(MODELS / "_test_y.npy", ys)

    params = int(sum(np.prod(w.shape) for w in model.get_weights()))
    results["mlp_params"] = params
    results["n_features"] = int(Xt.shape[1])
    results["split_sizes"] = {k: int(len(v[1])) for k, v in splits.items()}
    print(f"\n  MLP parameters: {params:,}  (float32 = {params * 4 / 1024:.1f} KB)")

    (MODELS / "train_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n-> results/mlp.keras, results/constants.json")
    print("   now run:  python -m securecan.evaluate")


if __name__ == "__main__":
    main()
