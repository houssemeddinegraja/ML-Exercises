import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# Check for external specialized GBDT libraries
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

print(f"Package Status -> LightGBM: {HAS_LGBM} | CatBoost: {HAS_CATBOOST}")
if not HAS_LGBM or not HAS_CATBOOST:
    print("Tip: Run 'pip install lightgbm catboost' in your terminal for maximum accuracy.\n")

# ==============================================================================
# 1. LOAD DATASETS
# ==============================================================================
DATA_PATH = "train.csv"
SUBMISSION_PATH = "test.csv"
TARGET_COLUMN = "target"

train_df = pd.read_csv(DATA_PATH)
test_df = pd.read_csv(SUBMISSION_PATH)

print(f"Train dataset shape: {train_df.shape}")
print(f"Test dataset shape:  {test_df.shape}")

# ==============================================================================
# 2. FEATURE ENGINEERING (HINT COLUMNS + NON-ACADEMIC METADATA)
# ==============================================================================
def prepare_features(df_input):
    df = df_input.copy()

    cohort = df["cohort_year"] if "cohort_year" in df.columns else pd.Series(0, index=df.index)
    s1_official = df["S1_rounded"].copy() if "S1_rounded" in df.columns else None

    # Binary Flags Normalization
    binary_map = {
        "yes": 1, "no": 0, "true": 1, "false": 0, "oui": 1, "non": 0,
        "1": 1, "0": 0, True: 1, False: 0, "m": 1, "f": 0, "male": 1,
        "female": 0, "homme": 1, "femme": 0
    }
    for col in ["club", "redoublant", "transfer", "laptop_owned", "has_github", "knows_the_hood", "gender"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip().map(binary_map).fillna(0).astype(int)

    # Bac Section Normalization
    bac_map = {
        "technique": "tech", "tech": "tech", "Tech": "tech",
        "sciences": "sc", "sc": "sc", "Sc": "sc", "SC": "sc",
        "informatique": "info", "info": "info", "Info": "info", "INFO": "info",
        "math": "math", "Math": "math", "MATH": "math"
    }
    if "bac_section" in df.columns:
        cleaned_bac = df["bac_section"].astype(str).str.lower().str.strip()
        df["bac_clean"] = cleaned_bac.map(bac_map).fillna("other")
    else:
        df["bac_clean"] = "other"

    df["is_bac_math"] = (df["bac_clean"] == "math").astype(int)
    df["is_bac_tech"] = (df["bac_clean"] == "tech").astype(int)
    df["is_bac_sc"] = (df["bac_clean"] == "sc").astype(int)
    df["is_bac_info"] = (df["bac_clean"] == "info").astype(int)

    # Governorates & Capital Residency
    capital_govs = ["tunis", "ariana", "ben arous", "manouba"]
    if "governorate" in df.columns:
        df["gov_clean"] = df["governorate"].astype(str).str.lower().str.strip()
        df["is_capital"] = df["gov_clean"].isin(capital_govs).astype(int)
    else:
        df["is_capital"] = 0

    # Behavioral Engagement & Risk Features
    att = df["attendance"].fillna(df["attendance"].mean()) if "attendance" in df.columns else pd.Series(0.8, index=df.index)
    df["attendance_clean"] = att

    df["engagement_score"] = df.get("club", 0) + df.get("has_github", 0) + df.get("laptop_owned", 0)
    df["club_x_attendance"] = df.get("club", 0) * att
    df["club_x_bac_math"] = df.get("club", 0) * df["is_bac_math"]
    df["club_x_bac_info"] = df.get("club", 0) * df["is_bac_info"]
    df["non_club_low_att_risk"] = ((df.get("club", 0) == 0) & (att < 0.75)).astype(int)
    df["redoublant_risk"] = df.get("redoublant", 0) * (2.0 - att)

    df["jawa7a_stereotype"] = df["is_capital"] * df["is_bac_math"] * df.get("knows_the_hood", 0)

    # Room Parsing & Peer Context
    if "exam_room" in df.columns:
        df["exam_room_num"] = df["exam_room"].astype(str).str.extract(r'(\d+)').fillna(-1).astype(int)

    # Marks & Sub-Averages
    s1_cols = ["an1", "alg1", "circ", "opt", "estat", "meca", "algo1", "prog1", "fr1", "en1", "droit"]
    s2_cols = ["an2", "alg2", "syslog", "electronique", "thermo", "emag", "algo2", "prog2", "fr2", "en2", "eco"]
    all_marks = list(set(s1_cols + s2_cols))
    
    if all(c in df.columns for c in all_marks):
        marks = df[all_marks].apply(lambda row: row.fillna(row.mean()), axis=1)

        df["moy_math"] = (marks["an1"] + marks["alg1"] + marks["an2"] + marks["alg2"]) / 4.0
        df["moy_inf"] = (2.0 * (marks["algo1"] + marks["algo2"]) + marks["prog1"] + marks["prog2"]) / 6.0
        df["moy_en"] = (marks["en1"] + marks["en2"]) / 2.0
        df["moy_fr"] = (marks["fr1"] + marks["fr2"]) / 2.0
        df["moy_physique"] = (marks["circ"] + marks["opt"] + marks["estat"] + marks["meca"] + marks["thermo"] + marks["emag"]) / 6.0
        df["hardware_score"] = (marks["electronique"] + marks["syslog"] + marks["thermo"] + marks["meca"]) / 4.0

        s1_fallback = marks[s1_cols].mean(axis=1)
        df["S1_rounded"] = (s1_official.fillna(s1_fallback) if s1_official is not None else s1_fallback).round(2)
        s2_weights = pd.Series({"an2": 3.0, "alg2": 3.0, "algo2": 3.0, "en2": 1.5, "eco": 1.5, "emag": 3.0, "electronique": 3.5, "fr2": 1.5, "syslog": 3.5, "thermo": 4.0, "prog2": 2.5})
        df["S2_rounded"] = (marks[s2_weights.index].dot(s2_weights) / s2_weights.sum()).round(2)
        df["MG"] = (df["S1_rounded"] + df["S2_rounded"]) / 2.0

        if "bac_avg" in df.columns:
            df["bac_avg"] = df["bac_avg"].fillna(df["bac_avg"].mean())
            df["insat_vs_bac_delta"] = df["MG"] - df["bac_avg"]

        df["score_gl"] = (2.0 * df["MG"]) + df["moy_math"] + (2.0 * df["moy_inf"]) + marks["syslog"]
        df["score_rt"] = (2.0 * df["MG"]) + df["moy_math"] + df["moy_inf"] + marks["syslog"] + df["moy_en"]
        df["score_iia"] = (2.0 * df["MG"]) + df["moy_inf"] + marks["syslog"] + ((df["moy_en"] + marks["electronique"]) / 2.0)
        df["score_imi"] = (2.0 * df["MG"]) + df["moy_math"] + (df["moy_en"] + (marks["syslog"] + marks["electronique"]) / 2.0) + ((marks["meca"] + marks["thermo"]) / 2.0)

        df["margin_gl_rt"] = df["score_gl"] - df["score_rt"]
        df["margin_gl_iia"] = df["score_gl"] - df["score_iia"]
        df["margin_gl_imi"] = df["score_gl"] - df["score_imi"]

        for col in ["score_gl", "score_rt", "score_iia", "score_imi", "MG"]:
            df[f"{col}_percentile"] = df[col].groupby(cohort).rank(pct=True)

        if "exam_room_num" in df.columns:
            df["room_mean_MG"] = df.groupby("exam_room_num")["MG"].transform("mean")
            df["room_relative_MG"] = df["MG"] - df["room_mean_MG"]

    # Drop unparsed high-cardinality string columns
    drop_cols = ["id", "cohort_year", "bac_section", "governorate", "birth_day_month", "exam_room"]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    return df

# Preprocess datasets
train_df_clean = prepare_features(train_df)
test_df_clean = prepare_features(test_df)

X = train_df_clean.drop(columns=[TARGET_COLUMN], errors="ignore")
y = train_df_clean[TARGET_COLUMN]

valid_idx = y.dropna().index
X, y = X.loc[valid_idx], y.loc[valid_idx]
groups = train_df.loc[valid_idx, "cohort_year"]

# ==============================================================================
# 3. PREPROCESSOR SETUP
# ==============================================================================
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categorical_cols)
])

# ==============================================================================
# 4. OUT-OF-FOLD (OOF) ENSEMBLE TRAINING
# ==============================================================================
gkf = GroupKFold(n_splits=train_df["cohort_year"].nunique())
classes = np.sort(y.unique())
num_classes = len(classes)

oof_probs = np.zeros((len(X), num_classes))
submission_cols = [c for c in X.columns if c in test_df_clean.columns]
test_probs_folds = np.zeros((len(test_df_clean), num_classes))

print(f"\nTraining Multi-Model Ensemble across {gkf.n_splits} Cohort Folds...")

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

    # Model 1: HistGradientBoosting
    m_hgb = Pipeline([
        ("pre", preprocessor),
        ("clf", HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.03, max_iter=300, l2_regularization=0.5, random_state=42 + fold,
            categorical_features=list(range(len(numeric_cols), len(numeric_cols) + len(categorical_cols))) if len(categorical_cols) > 0 else None
        ))
    ])
    m_hgb.fit(X_tr, y_tr)
    probs_hgb_va = m_hgb.predict_proba(X_va)
    probs_hgb_te = m_hgb.predict_proba(test_df_clean[submission_cols])

    # Model 2: ExtraTreesClassifier
    m_et = Pipeline([
        ("pre", preprocessor),
        ("clf", ExtraTreesClassifier(n_estimators=300, max_depth=12, random_state=42 + fold, n_jobs=-1))
    ])
    m_et.fit(X_tr, y_tr)
    probs_et_va = m_et.predict_proba(X_va)
    probs_et_te = m_et.predict_proba(test_df_clean[submission_cols])

    # Model 3: LightGBM (if installed)
    if HAS_LGBM:
        m_lgbm = Pipeline([
            ("pre", preprocessor),
            ("clf", LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, num_leaves=15, random_state=42 + fold, verbose=-1))
        ])
        m_lgbm.fit(X_tr, y_tr)
        probs_lgbm_va = m_lgbm.predict_proba(X_va)
        probs_lgbm_te = m_lgbm.predict_proba(test_df_clean[submission_cols])
    else:
        probs_lgbm_va, probs_lgbm_te = probs_hgb_va, probs_hgb_te

    # Model 4: CatBoost (if installed)
    if HAS_CATBOOST:
        m_cat = Pipeline([
            ("pre", preprocessor),
            ("clf", CatBoostClassifier(iterations=350, learning_rate=0.03, depth=5, random_seed=42 + fold, verbose=0))
        ])
        m_cat.fit(X_tr, y_tr)
        probs_cat_va = m_cat.predict_proba(X_va)
        probs_cat_te = m_cat.predict_proba(test_df_clean[submission_cols])
    else:
        probs_cat_va, probs_cat_te = probs_hgb_va, probs_hgb_te

    # Weighted Fold Blend
    if HAS_LGBM and HAS_CATBOOST:
        val_fold_blend = 0.35 * probs_cat_va + 0.30 * probs_lgbm_va + 0.20 * probs_hgb_va + 0.15 * probs_et_va
        test_fold_blend = 0.35 * probs_cat_te + 0.30 * probs_lgbm_te + 0.20 * probs_hgb_te + 0.15 * probs_et_te
    else:
        val_fold_blend = 0.60 * probs_hgb_va + 0.40 * probs_et_va
        test_fold_blend = 0.60 * probs_hgb_te + 0.40 * probs_et_te

    oof_probs[val_idx] = val_fold_blend
    test_probs_folds += test_fold_blend / gkf.n_splits

    fold_f1 = f1_score(y_va, classes[np.argmax(val_fold_blend, axis=1)], average="macro")
    print(f" Fold {fold + 1} Blend Macro F1: {fold_f1:.4f}")

# Overall OOF Score
default_preds = classes[np.argmax(oof_probs, axis=1)]
default_f1 = f1_score(y, default_preds, average="macro")
print(f"\nOverall Raw Ensemble OOF Macro F1: {default_f1:.5f}")

# ==============================================================================
# 5. POST-PROCESSING THRESHOLD OPTIMIZATION
# ==============================================================================
print("\nOptimizing Class Multipliers via Nelder-Mead for Macro F1...")

def macro_f1_loss(weights):
    scaled_probs = oof_probs * weights
    preds = classes[np.argmax(scaled_probs, axis=1)]
    return -f1_score(y, preds, average="macro")

opt_res = minimize(macro_f1_loss, [1.0] * num_classes, method="Nelder-Mead")
best_weights = opt_res.x

opt_preds = classes[np.argmax(oof_probs * best_weights, axis=1)]
opt_f1 = f1_score(y, opt_preds, average="macro")

print(f"Optimized Multipliers: {best_weights.round(4)}")
print(f"OPTIMIZED ENSEMBLE OOF MACRO F1: {opt_f1:.5f} (Gain: +{opt_f1 - default_f1:.5f})")

# ==============================================================================
# 6. GENERATE SUBMISSION
# ==============================================================================
final_test_preds = classes[np.argmax(test_probs_folds * best_weights, axis=1)]

submission = pd.DataFrame({
    "id": test_df["id"] if "id" in test_df.columns else test_df.index,
    TARGET_COLUMN: final_test_preds,
})
submission.to_csv("submission10.csv", index=False)
print("\nFinal Ensemble submission exported to 'submission10.csv'.")