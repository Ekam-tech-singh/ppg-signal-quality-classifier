
from pathlib import Path
import pandas as pd
import wfdb
import numpy as np

# Dataset location
root = Path(
    "data/raw/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0"
)

# Load labels
labels = pd.read_csv(root / "quality-hr-ann.csv")

signals = []
targets = []
subject_ids = []
failed = []

for _, row in labels.iterrows():

    record_id = str(int(row["ID"])).zfill(6)

    try:
        # Find recording
        record_path = next(
            root.rglob(f"{record_id}_PPG.hea")
        ).with_suffix("")

        record = wfdb.rdrecord(
            str(record_path),
            physical=True
        )

        signal = record.p_signal

        # Older single-channel format
        if record.n_sig == 300:
            signal = signal.flatten()

        # Newer RGB format: use red channel
        elif record.n_sig == 3:
            signal = signal[:, 0]

        else:
            raise ValueError("Unexpected signal format")

        # Validate signal
        if len(signal) != 300:
            raise ValueError("Incorrect signal length")

        if not np.isfinite(signal).all():
            raise ValueError("Signal contains invalid values")

        signals.append(signal)
        targets.append(int(row["Quality"]))
        subject_ids.append(record_id[:3])

    except Exception as e:
        failed.append((record_id, str(e)))


# Display results
print("\nDATASET LOADING RESULTS")
print("-----------------------")
print("Total labels:", len(labels))
print("Successfully loaded:", len(signals))
print("Failed:", len(failed))
print("Good quality:", targets.count(1))
print("Poor quality:", targets.count(0))
print("Unique subjects:", len(set(subject_ids)))

if failed:
    print("\nFirst 10 failures:")
    for item in failed[:10]:
        print(item)
