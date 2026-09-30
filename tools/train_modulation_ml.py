"""Train an experimental modulation model and evaluate it on separate sources.

Synthetic impairment-diverse windows train the model. A source-documented real
BPSK recording, when available, is used only as a separately reported transfer
check. No real multi-class accuracy is claimed from this single labeled capture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib


CLASSES = ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK")
WINDOW = 1024


def synth_window(label: str, rng: np.random.Generator) -> np.ndarray:
    sps = int(rng.integers(4, 9))
    n_symbols = int(np.ceil(WINDOW / sps)) + 2
    if label.endswith("PSK") or label == "BPSK":
        order = {"BPSK": 2, "QPSK": 4, "8PSK": 8}[label]
        ids = rng.integers(0, order, n_symbols)
        symbols = np.exp(2j * np.pi * ids / order + (np.pi / order if order == 2 else 0))
    elif label.endswith("QAM"):
        side = {"16QAM": 4, "64QAM": 8}[label]
        levels = np.arange(-(side - 1), side, 2, dtype=float)
        symbols = (rng.choice(levels, n_symbols) + 1j * rng.choice(levels, n_symbols))
        symbols /= np.sqrt(np.mean(np.abs(symbols) ** 2))
    else:
        symbols = np.exp(1j * np.pi / 2 * rng.integers(0, 4, n_symbols))
    up = np.repeat(symbols, sps)[: WINDOW + sps]
    if label == "2FSK":
        # Continuous-phase binary FSK with randomized modulation index.
        bits = rng.integers(0, 2, len(up))
        freq = np.where(bits, 1.0, -1.0) * rng.uniform(0.035, 0.12)
        up = np.exp(1j * np.cumsum(freq))
    snr_db = rng.uniform(-2, 24)
    noise_power = np.mean(np.abs(up) ** 2) / (10 ** (snr_db / 10))
    up = up + np.sqrt(noise_power / 2) * (rng.normal(size=len(up)) + 1j * rng.normal(size=len(up)))
    cfo = rng.uniform(-0.012, 0.012)
    phase = rng.uniform(-np.pi, np.pi)
    up *= np.exp(1j * (2 * np.pi * cfo * np.arange(len(up)) + phase))
    return np.asarray(up[:WINDOW], dtype=np.complex64)


def features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    if len(x) != WINDOW:
        raise ValueError(f"expected {WINDOW} IQ samples, got {len(x)}")
    x = x - np.mean(x)
    scale = np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12
    z = x / scale
    amp = np.abs(z)
    dphi = np.angle(z[1:] * np.conj(z[:-1]))
    # Remove the median phase slope so CFO does not dominate modulation features.
    dphi = np.angle(np.exp(1j * (dphi - np.median(dphi))))
    spec = np.abs(np.fft.fftshift(np.fft.fft(z * np.hanning(len(z)))))
    spec /= np.sum(spec) + 1e-12
    moments = [np.mean(amp ** k) for k in (1, 2, 3, 4, 6, 8)]
    phase = [np.mean(np.cos(k * dphi)) for k in (1, 2, 3, 4)] + [np.mean(np.sin(k * dphi)) for k in (1, 2, 3, 4)]
    quantiles = np.quantile(amp, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
    spectral = np.quantile(spec, [0.5, 0.9, 0.99]).tolist() + [np.max(spec), np.sum(spec ** 2)]
    cumulants = [abs(np.mean(z ** k)) for k in (2, 4, 6, 8)]
    return np.nan_to_num(np.asarray(moments + phase + quantiles + spectral + cumulants, dtype=np.float32))


def load_real_windows(path: Path) -> np.ndarray:
    iq = np.memmap(path, mode="r", dtype="<c8")
    # Centered windows, spaced apart to reduce near-duplicate adjacent crops.
    starts = np.linspace(0, len(iq) - WINDOW, min(128, max(1, len(iq) // (WINDOW * 8))), dtype=int)
    return np.stack([np.array(iq[i:i + WINDOW], copy=True) for i in starts])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("data/dataset_batches/ml"))
    ap.add_argument("--per-class", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=26147)
    ap.add_argument("--real-iq", type=Path, default=Path("data/dataset_batches/indoor-jamming/selected/w1_nojamming_70000000.iq"))
    args = ap.parse_args()
    if args.per_class < 20:
        ap.error("--per-class must be at least 20")
    rng = np.random.default_rng(args.seed)
    X, y = [], []
    for label in CLASSES:
        for _ in range(args.per_class):
            X.append(features(synth_window(label, rng)))
            y.append(label)
    X = np.stack(X)
    y = np.asarray(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=args.seed)
    model = make_pipeline(StandardScaler(), ExtraTreesClassifier(
        n_estimators=400, min_samples_leaf=2, max_features=0.9,
        class_weight="balanced", n_jobs=-1, random_state=args.seed,
    ))
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    report = {
        "task": "synthetic multi-class modulation recognition",
        "training_source": "generated signals with randomized SNR, CFO, phase, and samples-per-symbol",
        "real_capture_used_for_training": False,
        "classes": list(CLASSES), "window_samples": WINDOW,
        "synthetic_examples_per_class": args.per_class,
        "split": {"train": int(len(ytr)), "test": int(len(yte)), "stratified": True, "seed": args.seed},
        "synthetic_holdout_balanced_accuracy": float(balanced_accuracy_score(yte, yp)),
        "synthetic_holdout_confusion_matrix": confusion_matrix(yte, yp, labels=CLASSES).tolist(),
        "synthetic_holdout_classification_report": classification_report(yte, yp, labels=CLASSES, output_dict=True, zero_division=0),
        "real_transfer_check": None,
        "limitations": ["Synthetic holdout measures only generated-data generalization.", "The available real labeled capture documents BPSK only; this is not real multi-class validation.", "Windows come from one capture/session and do not establish independent field performance."],
    }
    if args.real_iq.exists():
        rw = load_real_windows(args.real_iq)
        rp = model.predict(np.stack([features(w) for w in rw]))
        report["real_transfer_check"] = {
            "capture": str(args.real_iq), "known_label": "BPSK", "label_basis": "source README / Zenodo record",
            "windows": len(rp), "predicted_class_counts": {c: int(np.sum(rp == c)) for c in CLASSES},
            "bpsk_prediction_fraction": float(np.mean(rp == "BPSK")),
            "interpretation": "single-capture transfer check only; windows are not independent recordings",
        }
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output / "synthetic_modulation_extratrees.joblib", compress=3)
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("synthetic_holdout_balanced_accuracy", "real_transfer_check")}, indent=2))


if __name__ == "__main__":
    main()
