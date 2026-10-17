import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ==============================================================================
# 1. LOAD YOUR DATASET (MULTICLASS CLASSIFICATION)
# ==============================================================================
DATA_PATH = "train.csv"
SUBMISSION_PATH = "test.csv"
TARGET_COLUMN = "target"

train_df = pd.read_csv(DATA_PATH)
test_df = pd.read_csv(SUBMISSION_PATH)

print(f"Train dataset shape: {train_df.shape}")
print(f"Test dataset shape:  {test_df.shape}")

# ==============================================================================
# 2. FEATURE ENGINEERING & DATA PREPARATION
# ==============================================================================
def prepare_features(df_input):
    """
    Enhanced feature engineering reflecting custom INSAT coefficients
    and allocation dynamics (cohort rankings & score margins).
    """
    df = df_input.copy()

    # Keep what we need from columns that are dropped / overwritten below
    cohort = df["cohort_year"] if "cohort_year" in df.columns else pd.Series(0, index=df.index)
    s1_official = df["S1_rounded"].copy() if "S1_rounded" in df.columns else None

    # 1. Drop metadata
    drop_cols = ["id", "gender", "knows_the_hood", "birth_day_month", "zodiac", "cohort_year", "exam_room"]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # 2. Categorical & Binary Cleanings
    if "bac_section" in df.columns:
        bac_map = {"technique": "tech", "sciences": "sc", "informatique": "info"}
        df["bac_section"] = df["bac_section"].astype(str).str.lower().str.strip().replace(bac_map)

    binary_map = {"yes": 1, "no": 0, "true": 1, "false": 0, "oui": 1, "non": 0, "1": 1, "0": 0, True: 1, False: 0}
    for col in ["club", "redoublant", "transfer", "laptop_owned", "has_github"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip().map(binary_map).fillna(0).astype(int)

    # 3. Compute Sub-Averages (Expose raw drivers so model learns custom coefficients)
    s1_cols = ["an1", "alg1", "circ", "opt", "estat", "meca", "algo1", "prog1", "fr1", "en1", "droit"]
    s2_cols = ["an2", "alg2", "syslog", "electronique", "thermo", "emag", "algo2", "prog2", "fr2", "en2", "eco"]
    
    all_marks = list(set(s1_cols + s2_cols))
    if all(c in df.columns for c in all_marks):
        marks = df[all_marks].apply(lambda row: row.fillna(row.mean()), axis=1)

        # Core subject averages
        df["moy_math"] = (marks["an1"] + marks["alg1"] + marks["an2"] + marks["alg2"]) / 4.0
        df["moy_inf"] = (2.0 * (marks["algo1"] + marks["algo2"]) + marks["prog1"] + marks["prog2"]) / 6.0
        df["moy_en"] = (marks["en1"] + marks["en2"]) / 2.0
        df["moy_fr"] = (marks["fr1"] + marks["fr2"]) / 2.0
        df["moy_physique"] = (marks["circ"] + marks["opt"] + marks["estat"] + marks["meca"] + marks["thermo"] + marks["emag"]) / 6.0

        # Semester averages
        # Use the official S1 average from the data (not a plain mean of the marks);
        # only fall back to the marks when it is missing
        s1_fallback = marks[s1_cols].mean(axis=1)
        df["S1_rounded"] = (s1_official.fillna(s1_fallback) if s1_official is not None else s1_fallback).round(2)
        s2_weights = pd.Series({"an2": 3.0, "alg2": 3.0, "algo2": 3.0, "en2": 1.5, "eco": 1.5, "emag": 3.0, "electronique": 3.5, "fr2": 1.5, "syslog": 3.5, "thermo": 4.0, "prog2": 2.5})
        df["S2_rounded"] = (marks[s2_weights.index].dot(s2_weights) / s2_weights.sum()).round(2)
        df["MG"] = (df["S1_rounded"] + df["S2_rounded"]) / 2.0

        # 4. Approximated Track Scores (Base reference points)
        df["score_gl"] = (2.0 * df["MG"]) + df["moy_math"] + (2.0 * df["moy_inf"]) + marks["syslog"]
        df["score_rt"] = (2.0 * df["MG"]) + df["moy_math"] + df["moy_inf"] + marks["syslog"] + df["moy_en"]
        df["score_iia"] = (2.0 * df["MG"]) + df["moy_inf"] + marks["syslog"] + ((df["moy_en"] + marks["electronique"]) / 2.0)
        df["score_imi"] = (2.0 * df["MG"]) + df["moy_math"] + (df["moy_en"] + (marks["syslog"] + marks["electronique"]) / 2.0) + ((marks["meca"] + marks["thermo"]) / 2.0)

        # 5. HINT #1: Allocation Dynamics (Margins & Dominance)
        scores_df = df[["score_gl", "score_rt", "score_iia", "score_imi"]]
        df["max_score"] = scores_df.max(axis=1)
        df["min_score"] = scores_df.min(axis=1)
        df["score_spread"] = df["max_score"] - df["min_score"]
        
        # Margins between GL and other tracks
        df["margin_gl_rt"] = df["score_gl"] - df["score_rt"]
        df["margin_gl_iia"] = df["score_gl"] - df["score_iia"]
        df["margin_gl_imi"] = df["score_gl"] - df["score_imi"]

        # Percentile of each student INSIDE their own cohort (seats are allocated per cohort)
        for col in ["score_gl", "score_rt", "score_iia", "score_imi", "MG"]:
            df[f"{col}_percentile"] = df[col].groupby(cohort).rank(pct=True)

    return df

# Preprocess training and submission datasets consistently
train_df_clean = prepare_features(train_df)
test_df_clean = prepare_features(test_df)

# Separate features and target
X = train_df_clean.drop(columns=[TARGET_COLUMN], errors="ignore")
y = train_df_clean[TARGET_COLUMN]

# Remove invalid target rows if any exist
valid_target_idx = y.dropna().index
X, y = X.loc[valid_target_idx], y.loc[valid_target_idx]

# Stratified split into Training (80%) and Test Holdout (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================================================================
# 3. TRANSFORMATION PIPELINES & COLUMN ROUTING
# ==============================================================================
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ],
    remainder="drop"
)

# ==============================================================================
# 4. GBDT MODEL PIPELINE & HYPERPARAMETER TUNING
# ==============================================================================
master_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    # Categorical columns come last after the ColumnTransformer -> let the GBDT treat them as categories
    ("model", HistGradientBoostingClassifier(
        random_state=42,
        categorical_features=list(range(len(numeric_cols), len(numeric_cols) + len(categorical_cols))),
    )),
])

# Hyperparameter grid suited for Gradient Boosted Trees
# Shallow trees + low learning rate generalised best to unseen cohorts
param_grid = {
    "model__max_depth": [2, 3],
    "model__learning_rate": [0.02, 0.03, 0.05],
    "model__max_iter": [200, 300],
    "model__l2_regularization": [0.0, 1.0],
}

# Validate by cohort (like the real test: unseen years) instead of random folds
groups_train = train_df.loc[X_train.index, "cohort_year"]

grid_search = GridSearchCV(
    master_pipeline,
    param_grid,
    cv=GroupKFold(n_splits=train_df["cohort_year"].nunique()),
    scoring="f1_macro",
    n_jobs=-1
)

print("Training HistGradientBoosting models with leave-one-cohort-out cross-validation...")
grid_search.fit(X_train, y_train, groups=groups_train)
print(f"Best cross-validated macro-F1 (unseen cohorts): {grid_search.best_score_:.4f}")

# ==============================================================================
# 5. EVALUATION METRICS ON HOLDOUT TEST SET
# ==============================================================================
best_model = grid_search.best_estimator_

test_probs = best_model.predict_proba(X_test)
test_preds = best_model.predict(X_test)

metrics_summary = pd.DataFrame(
    {
        "Accuracy": [accuracy_score(y_test, test_preds)],
        "Precision": [precision_score(y_test, test_preds, average="macro", zero_division=0)],
        "Recall": [recall_score(y_test, test_preds, average="macro", zero_division=0)],
        "Macro F1": [f1_score(y_test, test_preds, average="macro", zero_division=0)],
        "ROC-AUC": [roc_auc_score(y_test, test_probs, multi_class="ovr", average="macro", labels=best_model.classes_)],
    },
    index=["Test Set Holdout"],
)

print("\n" + "=" * 65)
print("             BEST MODEL HYPERPARAMETERS")
print("=" * 65)
for key, value in grid_search.best_params_.items():
    print(f" {key.replace('model__', ''):<20} : {value}")

print("\n" + "=" * 65)
print("     CLASSIFICATION PERFORMANCE METRICS (MACRO AVG)")
print("=" * 65)
print(
    metrics_summary.to_string(
        formatters={
            "Accuracy": "{:,.4f}".format,
            "Precision": "{:,.4f}".format,
            "Recall": "{:,.4f}".format,
            "Macro F1": "{:,.4f}".format,
            "ROC-AUC": "{:,.4f}".format,
        }
    )
)

print("\n" + "=" * 65)
print("              TEST SET CONFUSION MATRIX")
print("=" * 65)
class_labels = best_model.classes_
cm = confusion_matrix(y_test, test_preds, labels=class_labels)
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {c}" for c in class_labels],
    columns=[f"Pred {c}" for c in class_labels],
)
print(cm_df)

# ==============================================================================
# 6. FEATURE IMPORTANCE (PERMUTATION IMPORTANCE FOR GBDT)
# ==============================================================================
print("\nCalculating permutation feature importances on test set...")
perm_importance = permutation_importance(best_model, X_test, y_test, scoring="f1_macro", n_repeats=5, random_state=42)

feature_names = X_test.columns.tolist()
importance_df = (
    pd.DataFrame({
        "Feature": feature_names,
        "Importance_Mean": perm_importance.importances_mean,
        "Importance_Std": perm_importance.importances_std
    })
    .sort_values("Importance_Mean", ascending=False)
)

print("\n" + "=" * 65)
print("  TOP 15 MOST IMPORTANT FEATURES (BY PERMUTATION IMPACT ON MACRO F1)")
print("=" * 65)
print(importance_df.head(15).to_string(index=False, formatters={"Importance_Mean": "{:,.4f}".format, "Importance_Std": "{:,.4f}".format}))
print("=" * 65 + "\n")

# ==============================================================================
# 7. GENERATE SUBMISSION FILE FOR KAGGLE
# ==============================================================================
print("Generating submission predictions...")
submission_cols = [col for col in X.columns if col in test_df_clean.columns]
submission_preds = best_model.predict(test_df_clean[submission_cols])

submission = pd.DataFrame({
    "id": test_df["id"] if "id" in test_df.columns else test_df.index,
    TARGET_COLUMN: submission_preds,
})
submission.to_csv("submission.csv", index=False)
print("Submission successfully exported to 'submission.csv'.")