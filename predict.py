from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
import wfdb

from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks


def load_signal(path):
    path = Path(path)

    if path.suffix.lower() == ".hea":
        path = path.with_suffix("")

    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path, header=None).apply(
            pd.to_numeric, errors="coerce"
        ).dropna()

        signal = data.iloc[:, 0].to_numpy(dtype=float)

    else:
        record = wfdb.rdrecord(str(path), physical=True)
        signal = record.p_signal

        if record.n_sig == 3:
            signal = signal[:, 0]
        elif record.n_sig == 1:
            signal = signal[:, 0]
        else:
            raise ValueError("Expected a one-channel or three-channel recording.")

    signal = np.asarray(signal).flatten()

    if len(signal) != 300:
        raise ValueError(
            f"Expected 300 samples, but received {len(signal)}."
        )

    if not np.isfinite(signal).all():
        raise ValueError("Signal contains invalid values.")

    return signal


def extract_features(signal):
    signal_std = np.std(signal)
    differences = np.diff(signal)

    if signal_std <= 1e-10:
        signal_skewness = 0.0
        signal_kurtosis = 0.0
    else:
        signal_skewness = skew(signal)
        signal_kurtosis = kurtosis(signal)

    peaks, peak_properties = find_peaks(
        signal,
        distance=9,
        prominence=max(signal_std * 0.1, 1e-10)
    )

    troughs, _ = find_peaks(
        -signal,
        distance=9,
        prominence=max(signal_std * 0.1, 1e-10)
    )

    peak_intervals = np.diff(peaks) / 30.0
    peak_prominences = peak_properties["prominences"]

    return {
        "mean": np.mean(signal),
        "std": signal_std,
        "minimum": np.min(signal),
        "maximum": np.max(signal),
        "range": np.ptp(signal),
        "median": np.median(signal),
        "rms": np.sqrt(np.mean(signal ** 2)),
        "skewness": signal_skewness,
        "kurtosis": signal_kurtosis,
        "mean_absolute_difference": np.mean(np.abs(differences)),
        "difference_std": np.std(differences),
        "peak_count": len(peaks),
        "trough_count": len(troughs),
        "mean_peak_prominence": (
            np.mean(peak_prominences) if len(peak_prominences) else 0
        ),
        "std_peak_prominence": (
            np.std(peak_prominences) if len(peak_prominences) else 0
        ),
        "mean_peak_interval": (
            np.mean(peak_intervals) if len(peak_intervals) else 0
        ),
        "std_peak_interval": (
            np.std(peak_intervals) if len(peak_intervals) else 0
        )
    }


def main():
    parser = argparse.ArgumentParser(
        description="Classify a 10-second smartphone PPG recording."
    )
    parser.add_argument("recording", help="Path to a WFDB record or raw CSV.")
    args = parser.parse_args()

    bundle = joblib.load("ppg_quality_model.joblib")

    signal = load_signal(args.recording)
    features = extract_features(signal)

    feature_frame = pd.DataFrame(
        [features],
        columns=bundle["features"]
    )

    model = bundle["model"]
    threshold = bundle["threshold"]

    probability = model.predict_proba(feature_frame)[0, 1]
    prediction = int(probability >= threshold)

    label = "GOOD QUALITY" if prediction == 1 else "POOR QUALITY"

    print("\nPPG SIGNAL QUALITY PREDICTION")
    print("-----------------------------")
    print("Recording:", args.recording)
    print("Prediction:", label)
    print(f"Good-quality probability: {probability:.1%}")
    print(f"Classification threshold: {threshold:.2f}")
    print("\nResearch prototype only — not for clinical diagnosis.")


if __name__ == "__main__":
    main()
