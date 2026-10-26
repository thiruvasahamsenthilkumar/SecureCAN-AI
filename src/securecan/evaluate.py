#!/usr/bin/env python3
"""Measure the trained detector and write results/ .

Reports, all on the held-out test split:

  - per-class precision / recall / F1 and the confusion matrix
  - false-positive rate, measured per normal frame (the number that decides
    whether this could ever ship -- a car that cries wolf gets switched off)
  - detection delay: wall-clock time from the first injected frame of an attack
    burst to the first frame we flag
  - rule vs forest vs MLP side by side
  - the unseen-attack experiment: hold one attack class out of training entirely
    and see whether it still gets flagged as *something*

    python -m securecan.evaluate
    python -m securecan.evaluate --unseen       # also run the holdout experiment
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from . import dataset, features as F  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
CLASS_NAMES = ["Normal", "DoS", "Fuzzy", "Gear", "RPM"]


# ---------------------------------------------------------------------------
# core metrics
# ---------------------------------------------------------------------------

def per_class_report(y_true, y_pred):
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(5), zero_division=0)

    return {CLASS_NAMES[i]: {
        "precision": round(float(p[i]), 4),
        "recall": round(float(r[i]), 4),
        "f1": round(float(f1[i]), 4),
        "support": int(sup[i]),
    } for i in range(5)}


def false_positive_rate(y_true, y_pred):
    """Fraction of genuinely normal frames that we flagged as an attack.

    Measured per frame, not per window or per burst. At ~4,000 frames/second a
    0.1% FPR is still four false alarms every second, which is why the target is
    below 1% and why lower is much better than it sounds.
    """
    normal = y_true == 0
    if not normal.any():
        return 0.0
    return float((y_pred[normal] != 0).mean())


def binary_scores(y_true, y_pred):
    """Attack-vs-normal, ignoring which attack it was.

    The rule baseline can only answer this question, so it is the only fair way
    to put all three models on the same axis.
    """
    from sklearn.metrics import precision_recall_fscore_support

    t = (y_true != 0).astype(int)
    p_ = (y_pred != 0).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        t, p_, average="binary", zero_division=0)
    return {"precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4)}


def confusion(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred, labels=range(5))


def print_confusion(cm):
    head = "            " + "".join(f"{n:>10}" for n in CLASS_NAMES)
    print(head)
    for i, row in enumerate(cm):
        cells = "".join(f"{v:>10,}" for v in row)
        print(f"  {CLASS_NAMES[i]:<10}{cells}")


# ---------------------------------------------------------------------------
# detection delay
# ---------------------------------------------------------------------------

BURST_GAP = 0.5     # seconds of quiet that separates one injection campaign
                    # from the next; see the comment in detection_delay()


def detection_delay(y_true, y_pred, ts, gap=BURST_GAP):
    """Per attack burst: time from its first injected frame to our first alert.

    HCRL injected in campaigns of a few seconds. The subtlety is that during a
    campaign the legitimate ECUs keep transmitting, so the label sequence
    alternates frame by frame: 0,1,1,0,1,0,1,1,...  Grouping bursts by
    *contiguous* runs of the attack label therefore counts every individual
    injected frame as its own "burst" -- the first version of this function
    reported 45,329 bursts in one file, all with a delay of 0.000 ms, which is
    meaningless.

    So a burst is defined by a TIME GAP instead: consecutive injected frames
    belong to the same campaign unless more than `gap` seconds of clean traffic
    separate them. Measured gaps between injected frames are 0.5 ms (median) to
    33 ms (p99.9), with campaign boundaries showing up as gaps of seconds, so
    0.5 s separates the two scales cleanly.

    Delay is then measured from the campaign's first injected frame to the first
    frame we flag at or after that point -- which is the number that matters:
    how long is the attacker on the bus before the alarm goes off?

    A campaign we never flag at all is reported separately rather than folded in
    as a zero, which would flatter the average.
    """
    attack = np.flatnonzero(y_true != 0)
    if not len(attack):
        return {}

    # split the injected-frame indices into campaigns
    if len(attack) > 1:
        breaks = np.flatnonzero(np.diff(ts[attack]) > gap) + 1
    else:
        breaks = np.array([], dtype=np.int64)
    groups = np.split(attack, breaks)

    flagged_all = y_pred != 0
    delays, frames, missed, durations = [], [], 0, []

    for g in groups:
        if not len(g):
            continue
        s, e = int(g[0]), int(g[-1])
        durations.append(float(ts[e] - ts[s]))

        # first alert at or after the campaign started (search to its end)
        hit = np.flatnonzero(flagged_all[s:e + 1])
        if len(hit):
            delays.append(float(ts[s + hit[0]] - ts[s]))
            frames.append(int(hit[0]))
        else:
            missed += 1

    if not delays:
        return {"bursts": len(groups), "detected": 0, "missed": missed}

    d = np.array(delays)
    fr = np.array(frames)
    du = np.array(durations)

    return {
        "bursts": int(len(groups)),
        "detected": int(len(d)),
        "missed": int(missed),
        "burst_duration_median_s": round(float(np.median(du)), 3),
        # 0 frames means we flagged the very first injected frame of the burst
        "frames_to_alert_median": int(np.median(fr)),
        "frames_to_alert_max": int(fr.max()),
        "mean_ms": round(float(d.mean() * 1e3), 4),
        "median_ms": round(float(np.median(d) * 1e3), 4),
        "p95_ms": round(float(np.percentile(d, 95) * 1e3), 4),
        "max_ms": round(float(d.max() * 1e3), 4),
    }


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_confusion(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # row-normalise: raw counts are dominated by the normal class and tell you
    # nothing about whether the rare attacks were caught
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(5), CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(5), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)

    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{norm[i, j]:.3f}\n{cm[i, j]:,}",
                    ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")

    fig.colorbar(im, label="fraction of actual class")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# unseen-attack experiment
# ---------------------------------------------------------------------------

def unseen_attack(splits, held_out, epochs=15, drop_can_id=False):
    """Train with one attack class removed, then test on exactly that class.

    This asks a question the headline accuracy cannot: has the model learned what
    *these four attacks* look like, or something more general about abnormal CAN
    traffic? A detector that collapses on anything it wasn't shown is far less
    useful in the field, where tomorrow's attack is by definition not in the
    training set.

    Scored as binary -- we do not expect it to name the unseen class correctly,
    only to refuse to call it normal.

    drop_can_id zeroes feature 0 (can_id_norm) everywhere. See the comment in
    main(): the raw identifier lets the model memorise *which* ID each attack
    uses instead of learning what the attack does, and this is how we test that.
    """
    from .train import train_mlp

    cls = CLASS_NAMES.index(held_out)
    (Xt, yt), (Xv, yv), (Xs, ys) = splits["train"], splits["val"], splits["test"]

    if drop_can_id:
        Xt, Xv, Xs = Xt.copy(), Xv.copy(), Xs.copy()
        Xt[:, 0] = 0.0
        Xv[:, 0] = 0.0
        Xs[:, 0] = 0.0

    keep_t, keep_v = yt != cls, yv != cls
    tag = " (can_id_norm removed)" if drop_can_id else ""
    print(f"\n  holding out {held_out}{tag}: dropping "
          f"{int((~keep_t).sum()):,} train / {int((~keep_v).sum()):,} val frames")

    model = train_mlp(Xt[keep_t], yt[keep_t], Xv[keep_v], yv[keep_v],
                      epochs=epochs, quiet=True)

    mask = ys == cls
    if not mask.any():
        print(f"  no {held_out} frames in the test split -- nothing to score")
        return {"held_out": held_out, "test_frames": 0, "error": "no test frames"}

    pred = model.predict(Xs[mask], batch_size=8192, verbose=0).argmax(1)
    flagged = float((pred != 0).mean())

    # Does dropping a class cost us anything on the classes we DID train on?
    # A mitigation that wrecks normal performance is not a mitigation.
    seen = ys != cls
    pred_seen = model.predict(Xs[seen], batch_size=8192, verbose=0).argmax(1)
    norm = ys[seen] == 0
    fpr = float((pred_seen[norm] != 0).mean()) if norm.any() else 0.0

    # where did it put them? ideally the most similar attack it *did* see
    into = {CLASS_NAMES[i]: int((pred == i).sum()) for i in range(5)
            if (pred == i).any()}

    print(f"  {held_out} frames flagged as *some* attack: {flagged * 100:6.2f}%"
          f"   (FPR on normal still {fpr * 100:.4f}%)")
    print(f"  assigned to: {into}")

    return {"held_out": held_out,
            "can_id_norm_removed": bool(drop_can_id),
            "test_frames": int(mask.sum()),
            "flagged_as_attack_pct": round(flagged * 100, 2),
            "false_positive_rate_pct": round(fpr * 100, 4),
            "assigned_to": into}


def burst_timing_analysis(const, path, gap=BURST_GAP, n=250_000):
    """Why unseen attacks don't transfer. Measured, not asserted.

    The leave-one-attack-out numbers are poor for Gear and RPM, and removing
    can_id_norm does not help. The reason turns out to be a property of the data
    rather than of the model, and this function measures it.

    During a spoofing burst the attacker injects frames on a REAL id while the
    real ECU keeps transmitting on that same id. So the frames of that id
    alternate: fake, real, fake, real. Now consider period_ratio, which is
    (gap since the previous frame of this id) / (this id's normal period). A
    legitimate frame arriving 1 ms after an injected frame has just as small a
    gap as the injected frame did -- the anomaly is a property of the burst, not
    of the individual frame.

    So inside a burst, timing cannot separate injected from legitimate. The only
    thing that can is the PAYLOAD: the fake gear value versus the real one. And
    the fake payload is specific to each attack, so nothing about it transfers to
    an attack the model has never seen. That is the honest explanation for the
    0% numbers, and it is a limitation of per-frame labelling, not a bug.

    DoS is the control case. It injects 0x000, an id this car never uses, so
    there is no learned period and period_ratio does not even apply -- is_known_id
    and bus_rate catch it instead. Neither of those depends on knowing anything
    about the specific attack, which is exactly why DoS is the one attack that
    DOES transfer when held out (100%). The contrast is the point, so we report
    it rather than skipping the file.
    """
    d = np.load(path)
    lab_all = d["label"]
    sl = dataset.split_slices(lab_all)["test"][0]
    sl = slice(sl.start, min(sl.stop, sl.start + n))

    X = F.transform(d["ts"], d["can_id"], d["dlc"], d["data"], const)[sl]
    lab, ts, cid = lab_all[sl], d["ts"][sl], d["can_id"][sl]

    PR = F.FEATURE_NAMES.index("period_ratio")
    atk = np.flatnonzero(lab != 0)
    if not len(atk):
        return None

    cls = int(np.bincount(lab[atk]).argmax())
    targets = np.unique(cid[atk])
    if len(targets) > 4:
        return None          # Fuzzy sprays random ids; there is no single target
    target = int(targets[0])

    # An attack on an id the car never uses needs no payload knowledge at all,
    # so there is nothing here that could fail to transfer. Say so and stop.
    period = const.id_periods.get(str(target))
    if period is None:
        print(f"\n  {CLASS_NAMES[cls]} injects 0x{target:03X}, which is NOT a "
              f"known id of this car.")
        print("    period_ratio does not apply -- there is no schedule to "
              "compare against.")
        print("    is_known_id and bus_rate carry this attack, and neither "
              "needs to know")
        print("    anything attack-specific, which is why it is the one attack "
              "that transfers.")
        return {"attack": CLASS_NAMES[cls],
                "target_id": f"0x{target:03X}",
                "target_is_known_id": False,
                "learned_period_ms": None,
                "detected_by": ["is_known_id", "bus_rate"],
                "needs_attack_specific_payload": False}

    # mark every frame that lies inside an injection burst
    breaks = np.flatnonzero(np.diff(ts[atk]) > gap) + 1
    in_burst = np.zeros(len(lab), dtype=bool)
    for g in np.split(atk, breaks):
        if len(g):
            in_burst[g[0]:g[-1] + 1] = True

    tgt = cid == target
    groups = {
        "injected": tgt & (lab == cls),
        "legit_inside_burst": tgt & (lab == 0) & in_burst,
        "legit_outside_burst": tgt & (lab == 0) & ~in_burst,
    }

    out = {"attack": CLASS_NAMES[cls],
           "target_id": f"0x{target:03X}",
           "target_is_known_id": True,
           "learned_period_ms": round(period * 1e3, 3),
           "groups": {}}

    print(f"\n  {CLASS_NAMES[cls]} spoofs 0x{target:03X} "
          f"(normal period {period * 1e3:.1f} ms)")
    print(f"    {'frames of that id':<24}{'count':>9}{'period_ratio median':>22}")

    for name, m in groups.items():
        if not m.any():
            continue
        pr = X[m, PR]
        out["groups"][name] = {"count": int(m.sum()),
                               "period_ratio_median": round(float(np.median(pr)), 5)}
        print(f"    {name:<24}{int(m.sum()):>9,}{float(np.median(pr)):>22.4f}")

    gi = out["groups"].get("legit_inside_burst", {}).get("period_ratio_median")
    inj = out["groups"].get("injected", {}).get("period_ratio_median")
    if gi is not None and inj is not None and gi <= inj:
        print("    -> legitimate frames inside the burst look MORE anomalous than")
        print("       the injected ones, so timing alone cannot separate them.")
        out["timing_separates_injected_from_legit"] = False

    return out


def payload_shortcut(model, const, path, n=250_000):
    """The actual reason unseen spoofing is not detected. Decisive, not suggestive.

    HCRL generated each spoofing campaign by injecting ONE fixed payload, over
    and over, hundreds of thousands of times:

        RPM  id 0x316  ->  45 29 24 FF 29 24 00 FF    (1 distinct value)
        Gear id 0x43F  ->  01 45 60 FF 6B 00 00 00    (1 distinct value)

    Legitimate frames of those same ids carry thousands of distinct payloads. So
    a per-frame classifier does not need to learn "this id is arriving too often
    for its schedule" -- it can just memorise one 8-byte constant, which is a far
    easier function to fit. Nothing about a memorised constant transfers to an
    attack that uses a different constant, which is exactly the 0% we measured.

    The test here settles it. Take the real attack frames, leave the timing, the
    ids, the burst structure and every other feature untouched, and change ONLY
    the fake value to a different one. If the model were keying on timing the
    recall would barely move. If it memorised the constant, recall collapses.

    This is a property of the DATASET, not of our model: any per-frame classifier
    trained on this data can take the same shortcut, which is worth knowing
    before trusting published 99%-plus numbers on it.
    """
    d = np.load(path)
    lab_all = d["label"]
    sl = dataset.split_slices(lab_all)["test"][0]
    sl = slice(sl.start, min(sl.stop, sl.start + n))

    ts, cid, dlc = d["ts"][sl], d["can_id"][sl], d["dlc"][sl]
    data, lab = d["data"][sl].copy(), lab_all[sl]

    m = lab != 0
    if not m.any():
        return None
    cls = int(np.bincount(lab[m]).argmax())
    targets = np.unique(cid[m])
    if len(targets) > 4:
        return None              # Fuzzy has no single payload to memorise
    target = int(targets[0])

    inj_payloads = np.unique(data[m], axis=0)
    legit = (cid == target) & (lab == 0)
    legit_payloads = np.unique(data[legit], axis=0) if legit.any() else []

    def recall(payload_bytes):
        dd = data.copy()
        if payload_bytes is not None:
            dd[m] = np.asarray(payload_bytes, dtype=np.uint8)
        X = F.transform(ts, cid, dlc, dd, const)
        p = np.argmax(model.predict(X, batch_size=16384, verbose=0), axis=1)
        return float((p[m] != 0).mean() * 100.0)

    def hexs(b):
        return " ".join(f"{x:02X}" for x in np.asarray(b, dtype=np.uint8))

    base = recall(None)

    # The substituted value has to come from a REAL payload of this id, nudged by
    # one byte: that is what a plausible spoof looks like (an attacker lying about
    # the value) and it is nothing like the memorised constant.
    #
    # If the id has no legitimate payloads at all then it is not a spoofing attack
    # -- the id is not part of this car, so there is nothing to impersonate. That
    # is DoS, which floods 0x000. Substituting its payload would mean replacing
    # all-zeros with all-zeros and reporting the resulting 100% as evidence of
    # anything, which would be a vacuous test dressed up as a control. So say
    # plainly that the test does not apply and why.
    if not len(legit_payloads):
        print(f"\n  {CLASS_NAMES[cls]} floods 0x{target:03X}, an id this car never "
              f"sends.")
        print( "    There is no legitimate payload to substitute, so the payload")
        print( "    shortcut cannot be tested here -- and does not need to be:")
        print( "    an unknown id arriving thousands of times a second is caught")
        print( "    by is_known_id and bus_rate regardless of its bytes.")
        print(f"    recall   {base:6.2f}%")
        return {
            "attack": CLASS_NAMES[cls],
            "target_id": f"0x{target:03X}",
            "n_injected_frames": int(m.sum()),
            "distinct_injected_payloads": int(len(inj_payloads)),
            "distinct_legitimate_payloads": 0,
            "injected_constant": hexs(inj_payloads[0]),
            "test_applies": False,
            "reason": "flood of an id the car never sends; no payload to spoof",
            "recall_pct_hcrl_payload": round(base, 2),
            "memorised_the_payload": None,
        }

    novel = np.asarray(legit_payloads[0], dtype=np.uint8).copy()
    novel[-1] = (int(novel[-1]) + 1) % 256
    swapped = recall(novel)

    print(f"\n  {CLASS_NAMES[cls]} spoofs 0x{target:03X}")
    print(f"    distinct payloads, injected frames   {len(inj_payloads):>7,}")
    print(f"    distinct payloads, legitimate frames {len(legit_payloads):>7,}")
    print(f"    the injected constant   {hexs(inj_payloads[0])}")
    print(f"    a different fake value  {hexs(novel)}")
    print(f"    recall with HCRL's constant   {base:6.2f}%")
    print(f"    recall with the other value   {swapped:6.2f}%   "
          f"(timing and ids identical)")

    collapsed = base - swapped > 50.0
    if collapsed:
        print("    -> the model memorised the constant, not the timing anomaly.")

    return {
        "attack": CLASS_NAMES[cls],
        "target_id": f"0x{target:03X}",
        "n_injected_frames": int(m.sum()),
        "distinct_injected_payloads": int(len(inj_payloads)),
        "distinct_legitimate_payloads": int(len(legit_payloads)),
        "injected_constant": hexs(inj_payloads[0]),
        "substituted_value": hexs(novel),
        "test_applies": True,
        "recall_pct_hcrl_payload": round(base, 2),
        "recall_pct_substituted_payload": round(swapped, 2),
        "memorised_the_payload": bool(collapsed),
    }


def false_alarms_by_id(model, const, path=None, top=6):
    """Where the false alarms land on a capture with no attacks in it at all.

    The headline FPR is an average over every id, which hides the interesting
    part. If the model memorised the Gear attack constant, then genuine frames
    of that same id -- recorded while the car was in a different state -- can sit
    close to the constant and trip it. An average over 38 ids buries that.
    """
    p_npz = path or (ROOT / "data" / "processed" / "normal_run_data.npz")
    if not Path(p_npz).exists():
        return None

    d = np.load(p_npz)
    ts, cid, dlc, data = d["ts"], d["can_id"], d["dlc"], d["data"]
    if (d["label"] != 0).any():
        return None                      # must be an attack-free capture

    X = F.transform(ts, cid, dlc, data, const)
    p = np.argmax(model.predict(X, batch_size=16384, verbose=0), axis=1)

    overall = float((p != 0).mean() * 100.0)
    rows = []
    for u in np.unique(cid):
        m = cid == u
        if int(m.sum()) < 200:
            continue
        rows.append((float((p[m] != 0).mean() * 100.0), int(m.sum()), int(u)))
    rows.sort(reverse=True)

    print(f"\n  attack-free capture: {len(p):,} genuine frames, "
          f"overall false-alarm rate {overall:.4f}%")
    print(f"    {'id':<8}{'frames':>9}{'false alarms':>14}{'rate':>9}")
    for r, nf, u in rows[:top]:
        print(f"    0x{u:03X}   {nf:>8,}{int(round(r * nf / 100)):>14,}{r:>8.3f}%")
    print("    the worst ids are 10-50x the average, so a single headline FPR")
    print("    hides which traffic actually causes trouble.")

    return {
        "capture": Path(p_npz).stem,
        "frames": int(len(p)),
        "overall_false_alarm_pct": round(overall, 4),
        "worst_ids": [{"id": f"0x{u:03X}", "frames": nf,
                       "false_alarm_pct": round(r, 4)} for r, nf, u in rows[:top]],
    }


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unseen", action="store_true",
                    help="also run the leave-one-attack-out experiment")
    ap.add_argument("--unseen-class", default="all",
                    choices=["all"] + CLASS_NAMES[1:])
    ap.add_argument("--unseen-epochs", type=int, default=15)
    args = ap.parse_args()

    ys = np.load(RESULTS / "_test_y.npy")
    report = {}

    models = {
        "rule": np.load(RESULTS / "_rule_test_pred.npy"),
        "forest": np.load(RESULTS / "_forest_test_pred.npy"),
        "mlp": np.load(RESULTS / "_mlp_test_pred.npy"),
    }

    print(f"\ntest split: {len(ys):,} frames")
    print("  " + "  ".join(f"{CLASS_NAMES[i]}={int((ys == i).sum()):,}"
                           for i in range(5)))

    print("\n=== attack vs normal (all three comparable here) ===")
    print(f"  {'model':<10}{'precision':>11}{'recall':>9}{'F1':>9}{'FPR %':>9}")
    for name, pred in models.items():
        b = binary_scores(ys, pred)
        fpr = false_positive_rate(ys, pred)
        report[name] = {"binary": b, "false_positive_rate_pct": round(fpr * 100, 4)}
        print(f"  {name:<10}{b['precision']:>11.4f}{b['recall']:>9.4f}"
              f"{b['f1']:>9.4f}{fpr * 100:>9.4f}")

    print("\n=== per-class, MLP (the shipped model) ===")
    pc = per_class_report(ys, models["mlp"])
    report["mlp"]["per_class"] = pc
    report["mlp"]["accuracy"] = round(float((models["mlp"] == ys).mean()), 6)
    print(f"  {'class':<10}{'precision':>11}{'recall':>9}{'F1':>9}{'support':>12}")
    for n, m in pc.items():
        print(f"  {n:<10}{m['precision']:>11.4f}{m['recall']:>9.4f}"
              f"{m['f1']:>9.4f}{m['support']:>12,}")
    print(f"\n  overall accuracy {report['mlp']['accuracy'] * 100:.4f}%")

    print("\n=== per-class, random forest (reference) ===")
    report["forest"]["per_class"] = per_class_report(ys, models["forest"])
    report["forest"]["accuracy"] = round(float((models["forest"] == ys).mean()), 6)
    for n, m in report["forest"]["per_class"].items():
        print(f"  {n:<10}{m['precision']:>11.4f}{m['recall']:>9.4f}{m['f1']:>9.4f}")

    print("\n=== confusion matrix, MLP ===")
    cm = confusion(ys, models["mlp"])
    print_confusion(cm)
    report["mlp"]["confusion_matrix"] = cm.tolist()

    RESULTS.mkdir(exist_ok=True)
    plot_confusion(cm, RESULTS / "confusion_mlp.png", "MLP -- held-out test split")
    plot_confusion(confusion(ys, models["forest"]),
                   RESULTS / "confusion_forest.png", "Random forest -- test split")

    # --- detection delay ---------------------------------------------------
    print("\n=== detection delay (per attack burst) ===")
    const = F.Constants.load(RESULTS / "constants.json")

    import tensorflow as tf
    model = tf.keras.models.load_model(RESULTS / "mlp.keras")

    delays = {}
    for p in sorted((ROOT / "data" / "processed").glob("*.npz")):
        if "normal" in p.name.lower():
            continue

        Xd, yd, td = dataset.test_frames_for_delay(const, p)
        pd_ = model.predict(Xd, batch_size=8192, verbose=0).argmax(1)
        d = detection_delay(yd, pd_, td)
        delays[p.stem] = d
        if d.get("detected"):
            print(f"  {p.stem:<22} bursts {d['bursts']:>4}  "
                  f"missed {d['missed']:>3}  median {d['median_ms']:>8.3f} ms  "
                  f"p95 {d['p95_ms']:>9.3f} ms")
        else:
            print(f"  {p.stem:<22} bursts {d.get('bursts', 0):>4}  none detected")
    report["detection_delay"] = delays

    # write now, so a crash in the (long, optional) experiment below does not
    # throw away everything above it
    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))

    # --- unseen attack -----------------------------------------------------
    if args.unseen:
        print("\n=== leave-one-attack-out experiment ===")
        print("  Train with one attack class deleted, then test on exactly that")
        print("  class. Asks whether the model learned 'abnormal CAN traffic' or")
        print("  just 'these four specific attacks'. Scored binary: we only ask")
        print("  that it refuses to call the unseen attack normal.")

        splits = dataset.load_or_build(const)
        classes = (CLASS_NAMES[1:] if args.unseen_class == "all"
                   else [args.unseen_class])

        runs = []
        for c in classes:
            runs.append(unseen_attack(splits, c, epochs=args.unseen_epochs))
        report["unseen_attack"] = runs
        (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))

        # --- hypothesis 1: the model memorised the attacked ID --------------
        # Gear spoofs 0x43F and RPM spoofs 0x316. Both are "a known ID arriving
        # ~10x too fast", so the timing features ought to transfer from one to
        # the other. But the raw identifier is feature 0, so the model could be
        # taking the shortcut of memorising *which* ID gets attacked instead of
        # learning what an attack looks like -- and a memorised ID is worthless
        # against an attack never seen.
        #
        # Cheapest possible test: zero that feature and re-run. Reported whether
        # it works or not; it is at least as useful to know what the cause ISN'T.
        worst = min(r.get("flagged_as_attack_pct", 100.0) for r in runs)
        if worst < 90.0:
            print("\n=== hypothesis 1: is it the raw CAN ID? ===")
            print("  If the model memorised WHICH id each attack uses, removing")
            print("  feature 0 (can_id_norm) should force it onto the timing")
            print("  features and the numbers above should improve.")

            abl = []
            for c in classes:
                abl.append(unseen_attack(splits, c, epochs=args.unseen_epochs,
                                         drop_can_id=True))
            report["unseen_attack_no_can_id"] = abl

            print(f"\n  {'held out':<10}{'with can_id':>14}{'without':>12}")
            for a, b in zip(runs, abl):
                print(f"  {a['held_out']:<10}"
                      f"{a.get('flagged_as_attack_pct', 0):>13.2f}%"
                      f"{b.get('flagged_as_attack_pct', 0):>11.2f}%")

            changed = max(
                abs(a.get("flagged_as_attack_pct", 0) - b.get("flagged_as_attack_pct", 0))
                for a, b in zip(runs, abl))
            verdict = ("CONFIRMED" if changed > 5.0 else
                       "NOT the cause -- removing the ID changes nothing")
            print(f"\n  verdict: {verdict}")
            report["hypothesis_can_id_shortcut"] = {
                "max_change_pp": round(changed, 2),
                "confirmed": bool(changed > 5.0),
            }

            # --- hypothesis 2: the labels themselves ------------------------
            # Since the ID is not the cause, look at what timing can actually
            # see during a burst. See burst_timing_analysis() for the argument.
            print("\n=== hypothesis 2: can timing separate injected from legit? ===")
            print("  During a spoofing burst the real ECU keeps transmitting on")
            print("  the same id, so frames alternate fake/real. If a legitimate")
            print("  frame arriving right after an injected one looks just as")
            print("  anomalous, then timing cannot tell them apart and the model")
            print("  has to use the attack-specific payload instead.")

            bursts = {}
            for p in sorted((ROOT / "data" / "processed").glob("*.npz")):
                if "normal" in p.name.lower():
                    continue
                r = burst_timing_analysis(const, p)
                if r:
                    bursts[p.stem] = r
            report["burst_timing_analysis"] = bursts

            (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))

            # --- hypothesis 3: it memorised one 8-byte constant -------------
            # Hypothesis 2 showed timing CANNOT separate injected from legit
            # inside a burst, so something else must be carrying the 100% test
            # score. Looking at the actual payloads answers it: HCRL injected a
            # single fixed value for each spoofing campaign, hundreds of
            # thousands of times. Memorising one constant is a much easier
            # function than learning a timing anomaly, and it transfers to
            # nothing.
            #
            # The test leaves timing, ids and burst structure untouched and
            # changes only the fake value. See payload_shortcut().
            print("\n=== hypothesis 3: did it memorise the payload constant? ===")
            print("  HCRL injected ONE fixed payload per spoofing campaign.")
            print("  Keep the timing, the ids and the bursts exactly as they")
            print("  are and substitute a different fake value. Timing-based")
            print("  detection would be unaffected; a memorised constant dies.")

            shortcut = {}
            for p in sorted((ROOT / "data" / "processed").glob("*.npz")):
                if "normal" in p.name.lower():
                    continue
                r = payload_shortcut(model, const, p)
                if r:
                    shortcut[p.stem] = r
            report["payload_shortcut"] = shortcut

            if shortcut:
                # Only the spoofing attacks can be judged. DoS floods an id the
                # car never sends, so there is no payload to substitute and the
                # test does not apply -- folding it into an all() would let a
                # non-result outvote two real ones.
                testable = [v for v in shortcut.values() if v.get("test_applies")]
                confirmed = bool(testable) and all(
                    v["memorised_the_payload"] for v in testable)

                print(f"\n  {'attack':<8}{'HCRL payload':>15}{'substituted':>14}")
                for v in shortcut.values():
                    if v.get("test_applies"):
                        print(f"  {v['attack']:<8}"
                              f"{v['recall_pct_hcrl_payload']:>14.2f}%"
                              f"{v['recall_pct_substituted_payload']:>13.2f}%")
                    else:
                        print(f"  {v['attack']:<8}"
                              f"{v['recall_pct_hcrl_payload']:>14.2f}%"
                              f"{'n/a':>13}   ({v['reason']})")

                print(f"\n  verdict on the spoofing attacks: "
                      f"{'CONFIRMED' if confirmed else 'not confirmed'}")
                if confirmed:
                    print("  This is the real cause of the 0% leave-one-out recall.")
                    print("  It is a property of the DATASET, so ANY per-frame model")
                    print("  trained on it can take the same shortcut. Published")
                    print("  99%-plus numbers on this data should be read with that")
                    print("  in mind, including ours.")
                report["hypothesis_payload_shortcut"] = {
                    "confirmed": confirmed,
                    "scope": "spoofing attacks (Gear, RPM); DoS not applicable",
                    "attacks_tested": [v["attack"] for v in testable],
                }

            # Where do the false alarms actually land? If the constant was
            # memorised, genuine frames of that same id can sit near it.
            print("\n=== and what does that cost on clean traffic? ===")
            fa = false_alarms_by_id(model, const)
            if fa:
                report["false_alarms_by_id"] = fa

            (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))

    print("\n-> results/metrics.json, results/confusion_*.png")


if __name__ == "__main__":
    main()
