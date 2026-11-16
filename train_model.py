import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

# Load the RGB feature dataset
df = pd.read_csv("data/processed/ppg_features_rgb.csv")

# Select the 17 signal features
feature_columns = [
    "mean",
    "std",
    "minimum",
    "maximum",
    "range",
    "median",
    "rms",
    "skewness",
    "kurtosis",
    "mean_absolute_difference",
    "difference_std",
    "peak_count",
    "trough_count",
    "mean_peak_prominence",
    "std_peak_prominence",
    "mean_peak_interval",
    "std_peak_interval"
]

# Inputs and target labels
X = df[feature_columns]
y = df["Quality"]

# Subject IDs for subject-wise splitting
groups = df["subject_id"]

print("Dataset loaded successfully!")
print("Recordings:", len(df))
print("Features:", len(feature_columns))
print("Subjects:", groups.nunique())

# Split data by subject to prevent data leakage
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nTRAIN/TEST SPLIT")
print("----------------")
print("Training recordings:", len(X_train))
print("Testing recordings:", len(X_test))
print("Training subjects:", groups.iloc[train_idx].nunique())
print("Testing subjects:", groups.iloc[test_idx].nunique())

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)


# Train the model
model.fit(X_train, y_train)

# Analyze feature importance
importances = pd.Series(
    model.feature_importances_,
    index=feature_columns
).sort_values(ascending=False)

print("\nFEATURE IMPORTANCE")
print("------------------")
print(importances)

print("\nMODEL TRAINING COMPLETE!")
print("Random Forest trained on", len(X_train), "recordings.")

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate model performance
print("\nMODEL EVALUATION")
print("----------------")

print("\nBalanced Accuracy:")
print(balanced_accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Poor Quality", "Good Quality"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# EXPERIMENT 1: SUBJECT-AWARE THRESHOLD TUNING
# ==========================================

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

# Compare thresholds using subject-aware cross-validation
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

comparison_models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

        "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42
    ),

   "Gradient Boosting": GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    )
}

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

thresholds = np.arange(0.10, 0.91, 0.01)

print("\n\nSUBJECT-AWARE THRESHOLD COMPARISON")
print("----------------------------------")

for name, candidate in comparison_models.items():

    oof_probabilities = np.full(len(X_train), np.nan)

    for fold, (fit_idx, val_idx) in enumerate(
        cv.split(X_train, y_train, groups.iloc[train_idx]), 1
    ):
        fold_model = clone(candidate)

        if name == "Gradient Boosting":
            weights = compute_sample_weight(
                "balanced", y_train.iloc[fit_idx]
            )
            fold_model.fit(
                X_train.iloc[fit_idx],
                y_train.iloc[fit_idx],
                sample_weight=weights
            )
        else:
            fold_model.fit(
                X_train.iloc[fit_idx],
                y_train.iloc[fit_idx]
            )

        oof_probabilities[val_idx] = fold_model.predict_proba(
            X_train.iloc[val_idx]
        )[:, 1]

    best_threshold = 0.50
    best_score = 0

    for threshold in thresholds:
        predictions = (oof_probabilities >= threshold).astype(int)
        score = balanced_accuracy_score(y_train, predictions)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    if name == "Gradient Boosting":
        weights = compute_sample_weight("balanced", y_train)
        candidate.fit(X_train, y_train, sample_weight=weights)
    else:
        candidate.fit(X_train, y_train)

    test_probabilities = candidate.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= best_threshold).astype(int)

    print(f"\n{name}")
    print("Best threshold:", round(best_threshold, 2))
    print("Training CV balanced accuracy:", round(best_score, 4))
    print("Test balanced accuracy:", round(
        balanced_accuracy_score(y_test, test_predictions), 4
    ))
    print("Test confusion matrix:")
    print(confusion_matrix(y_test, test_predictions))
print("--------------------------")

# Create 5 subject-aware folds using training data only
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Store out-of-fold probabilities for training recordings
oof_probabilities = np.full(len(X_train), np.nan)

for fold, (fit_idx, val_idx) in enumerate(
    cv.split(X_train, y_train, groups.iloc[train_idx]), 1
):
    fold_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )

    fold_model.fit(
        X_train.iloc[fit_idx],
        y_train.iloc[fit_idx]
    )

    oof_probabilities[val_idx] = fold_model.predict_proba(
        X_train.iloc[val_idx]
    )[:, 1]

    print(
        f"Fold {fold}: "
        f"{len(fit_idx)} training recordings, "
        f"{len(val_idx)} validation recordings"
    )

# Find the threshold with the best balanced accuracy
thresholds = np.arange(0.10, 0.91, 0.01)

best_threshold = 0.50
best_score = 0

for threshold in thresholds:
    oof_predictions = (oof_probabilities >= threshold).astype(int)
    score = balanced_accuracy_score(y_train, oof_predictions)

    if score > best_score:
        best_score = score
        best_threshold = threshold

print("\nBEST TRAINING THRESHOLD")
print("-----------------------")
print("Threshold:", round(best_threshold, 2))
print("Training cross-validation balanced accuracy:", round(best_score, 4))

# Evaluate the selected threshold on untouched test subjects
test_probabilities = model.predict_proba(X_test)[:, 1]
tuned_predictions = (test_probabilities >= best_threshold).astype(int)

print("\nHELD-OUT TEST RESULTS")
print("---------------------")
print("Balanced Accuracy:",
      round(balanced_accuracy_score(y_test, tuned_predictions), 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    tuned_predictions,
    target_names=["Poor Quality", "Good Quality"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tuned_predictions))

# Compare three models on the same held-out subjects
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    )
}

print("\n\nTHREE-MODEL COMPARISON")
print("----------------------")

for name, comparison_model in models.items():

    if name == "Gradient Boosting":
        weights = compute_sample_weight("balanced", y_train)
        comparison_model.fit(X_train, y_train, sample_weight=weights)
    else:
        comparison_model.fit(X_train, y_train)

    predictions = comparison_model.predict(X_test)

    print(f"\n{name}")
    print("Balanced accuracy:", round(
        balanced_accuracy_score(y_test, predictions), 4
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))


# ==========================================
# FINAL MODEL: GRADIENT BOOSTING
# ==========================================

import joblib
from sklearn.model_selection import StratifiedGroupKFold

print("\nFINAL GRADIENT BOOSTING MODEL")
print("-----------------------------")

final_model = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)

final_cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

final_oof_probabilities = np.full(len(X_train), np.nan)

for fit_idx, val_idx in final_cv.split(
    X_train, y_train, groups.iloc[train_idx]
):
    fold_model = clone(final_model)

    weights = compute_sample_weight(
        "balanced", y_train.iloc[fit_idx]
    )

    fold_model.fit(
        X_train.iloc[fit_idx],
        y_train.iloc[fit_idx],
        sample_weight=weights
    )

    final_oof_probabilities[val_idx] = (
        fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]
    )

best_final_threshold = 0.50
best_final_score = 0

for threshold in thresholds:
    predictions = (
        final_oof_probabilities >= threshold
    ).astype(int)

    score = balanced_accuracy_score(y_train, predictions)

    if score > best_final_score:
        best_final_score = score
        best_final_threshold = threshold

final_weights = compute_sample_weight("balanced", y_train)

final_model.fit(
    X_train,
    y_train,
    sample_weight=final_weights
)

final_test_probabilities = final_model.predict_proba(X_test)[:, 1]

final_test_predictions = (
    final_test_probabilities >= best_final_threshold
).astype(int)

final_test_score = balanced_accuracy_score(
    y_test, final_test_predictions
)

print("Selected threshold:", round(best_final_threshold, 2))
print("CV balanced accuracy:", round(best_final_score, 4))
print("Held-out balanced accuracy:", round(final_test_score, 4))

model_bundle = {
    "model": final_model,
    "threshold": best_final_threshold,
    "features": feature_columns,
    "sampling_frequency": 30,
    "recording_duration_seconds": 10,
    "model_name": "Gradient Boosting"
}

joblib.dump(model_bundle, "ppg_quality_model.joblib")

print("\nSaved model to ppg_quality_model.joblib")
