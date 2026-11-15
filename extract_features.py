
from pathlib import Path
import pandas as pd
import wfdb
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

# Dataset location
root = Path(
    "data/raw/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0"
)

# Load labels
labels = pd.read_csv(root / "quality-hr-ann.csv")

results = []
failed = []

for _, row in labels.iterrows():

    record_id = str(int(row["ID"])).zfill(6)

    try:
        # Find and read recording
        record_path = next(
            root.rglob(f"{record_id}_PPG.hea")
        ).with_suffix("")

        record = wfdb.rdrecord(
            str(record_path),
            physical=True
        )

        signal = record.p_signal

        # Handle older single-channel recordings
        if record.n_sig == 300:
            signal = signal.flatten()

        # Handle newer RGB recordings using red channel
        elif record.n_sig == 3:
            signal = signal[:, 0]

        else:
            raise ValueError("Unexpected signal format")

        # Validate signal
        if len(signal) != 300:
            raise ValueError("Incorrect signal length")

        if not np.isfinite(signal).all():
            raise ValueError("Signal contains invalid values")

        # Calculate basic statistics
        signal_std = np.std(signal)
        differences = np.diff(signal)

        # Handle effectively constant signals
        if signal_std <= 1e-10:
            skewness = 0.0
            kurt = 0.0
        else:
            skewness = skew(signal)
            kurt = kurtosis(signal)

       # Detect pulse peaks and troughs
        peaks, peak_properties = find_peaks(
            signal,
            distance=9,
            prominence=max(signal_std * 0.1, 1e-10)
        )

        troughs, trough_properties = find_peaks(
            -signal,
            distance=9,
            prominence=max(signal_std * 0.1, 1e-10)
        )

        # Calculate peak intervals in seconds (30 Hz sampling rate)
        peak_intervals = np.diff(peaks) / 30.0

        # Extract peak prominence values
        peak_prominences = peak_properties["prominences"]

        # Extract 11 features
        features = {
            "record_id": record_id,
            "subject_id": record_id[:3],
            "Quality": int(row["Quality"]),

            "mean": np.mean(signal),
            "std": signal_std,
            "minimum": np.min(signal),
            "maximum": np.max(signal),
            "range": np.ptp(signal),
            "median": np.median(signal),
            "rms": np.sqrt(np.mean(signal ** 2)),
            "skewness": skewness,
            "kurtosis": kurt,
            "mean_absolute_difference": np.mean(
                np.abs(differences)
            ),
   "difference_std": np.std(differences),

# Waveform-shape features
"peak_count": len(peaks),
"trough_count": len(troughs),
"mean_peak_prominence": (
    np.mean(peak_prominences) if len(peak_prominences) > 0 else 0
),
"std_peak_prominence": (
    np.std(peak_prominences) if len(peak_prominences) > 0 else 0
),
"mean_peak_interval": (
    np.mean(peak_intervals) if len(peak_intervals) > 0 else 0
),
"std_peak_interval": (
    np.std(peak_intervals) if len(peak_intervals) > 0 else 0
)
        }

        # Check extracted features
        feature_values = [
            value for key, value in features.items()
            if key not in ["record_id", "subject_id", "Quality"]
        ]

        if not np.isfinite(feature_values).all():
            raise ValueError("Extracted features contain invalid values")

        results.append(features)

    except Exception as e:
        failed.append((record_id, str(e)))


# Create processed data folder
output = Path("data/processed")
output.mkdir(parents=True, exist_ok=True)

# Save extracted features
df = pd.DataFrame(results)

df.to_csv(
    output / "ppg_features.csv",
    index=False
)

# Display results
print("\nFEATURE EXTRACTION RESULTS")
print("--------------------------")
print("Total recordings:", len(labels))
print("Successfully processed:", len(results))
print("Failed:", len(failed))
print("Features per recording:", len(df.columns) - 3)
print("Saved to:", output / "ppg_features.csv")

if failed:
    print("\nFirst 10 failures:")
    for item in failed[:10]:
        print(item)
