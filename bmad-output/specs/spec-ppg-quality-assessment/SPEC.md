---
id: SPEC-ppg-quality-assessment
companions: []
sources: ["extract_features.py", "inspect_features.py", "load_dataset.py", "train_model.py", "data/processed/ppg_features_rgb.csv"]
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete, preservation-validated contract for what to build, test, and validate. Source documents listed in frontmatter are for traceability — consult them only if you need narrative rationale or prose color this contract intentionally omits.

# PPG Quality Assessment

## Why

This project exists to distinguish usable smartphone PPG recordings from low-quality recordings before downstream physiological analysis. The underlying problem is that the raw signal can be noisy, inconsistent, or distorted by motion, sensor quality, or recording artifacts, and those defects can corrupt downstream heart-rate and signal-processing tasks. The work matters because a reliable quality gate is a prerequisite for trustworthy cardiovascular signal interpretation, especially when data is collected in real-world mobile settings.

## Capabilities

- **CAP-1**
  - **intent:** System can classify a PPG recording as good or poor quality using signal-derived features extracted from the waveform.
  - **success:** The classifier produces a binary Quality label with test-set performance that is measured and reported using a subject-aware split and balanced accuracy.

- **CAP-2**
  - **intent:** System can summarize and inspect waveform statistics to identify which features are most indicative of quality variations.
  - **success:** A feature dataset is generated with consistent numeric statistics, missing values are checked, and quality distributions are reported by subject and recording format.

- **CAP-3**
  - **intent:** System can prevent subject leakage by evaluating on held-out subjects rather than random recordings.
  - **success:** Model training and evaluation use GroupShuffleSplit so recordings from the same subject remain in either train or test, and the report documents the resulting subject counts.

## Constraints

- Data comes from the BRNO smartphone PPG database and must be interpreted as a binary labeling problem using the provided Quality field.
- The dataset used for modeling is reduced to the newer RGB recordings that follow the 3-channel red-channel format; older single-channel recordings are treated as a separate analysis path rather than mixed into the modeling dataset.
- Subject-wise separation is required to avoid data leakage across samples from the same individual.
- The feature set is intentionally restricted to 11 statistical descriptors derived from the signal waveform rather than full waveform models or temporal deep-learning features.
- The project must operate with an interpretable, tabular ML pipeline rather than opaque end-to-end signal models.

## Non-goals

- Building a full end-to-end HR estimation or PPG waveform reconstruction system.
- Training a multi-class quality taxonomy beyond the existing binary quality label.
- Publishing a production deployment pipeline or API wrapper for real-time inference.
- Resolving all data-quality edge cases in the raw database beyond the project’s processed quality classification task.

## Success signal

The project succeeds when it produces a consistent, subject-aware quality classifier that can separate poor and good smartphone PPG recordings with measurable performance on held-out subjects, backed by a clear feature set and reproducible data pipeline. In practical terms, this means the generated model report includes balanced accuracy and a classification summary for the held-out subject split without leakage between recordings from the same subject.

## Assumptions

- The Quality label in the database is the authoritative target for binary quality assessment.
- The red-channel waveform is the meaningful signal representation for the RGB recordings used in the modeling dataset.
- The 11 highlighted feature statistics are sufficient to distinguish quality differences in this dataset for a baseline ML approach.
- The current project is a research and modeling workflow rather than a product-grade clinical monitoring deployment.

## Open Questions

- Should this baseline quality filter be expanded to include additional waveform-temporal features or a second model family before being considered operationally reliable?
- Is the project intended to stop at a classifier baseline, or should it progress toward a deployment-ready quality gate for real-time screening?
- Are there downstream tasks, such as HR estimation, that require stricter recall for poor-quality rejection than the current binary classification objective provides?
