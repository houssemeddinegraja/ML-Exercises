"""
Improved version of the user's script (same structure: 1 load, 2 features, 3 pipeline, 4 models,
5 evaluation, 6 permutation importance, 7 submission) -- every change below was validated with
YEAR-GROUPED cross-validation (whole cohort years held out, like the 2025/2026 test).

Usage:   python improved_code.py                (data folder = folder of this script)
         python improved_code.py --cv           (also print grouped-CV metrics, confusion matrix, importances)
         python improved_code.py --tune         (also grid-search HistGradientBoosting with grouped CV; slow)
Needs:   numpy pandas scikit-learn lightgbm xgboost catboost
"""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_COLUMN = "target"
CLASSES = ["GL", "RT", "IIA", "IMI", "fail"]
S1 = ["an1", "alg1", "circ", "opt", "estat", "meca", "algo1", "prog1", "fr1", "en1", "droit"]
S2 = ["an2", "alg2", "syslog", "electronique", "thermo", "emag", "algo2", "prog2", "fr2", "en2", "eco"]
GRADES = S1 + S2
# Dataset weights: the 8 core subjects count 1.5, languages/law count 1 (reproduces the given S1_rounded exactly)
W = np.array([1.5] * 8 + [1] * 3)
# Official INSAT coefficients (only used to build track-score *rankings*; the dataset's own formulas differ slightly)
W1R = np.array([3, 3, 3.5, 3.5, 3.5, 3.5, 3, 2.5, 1.5, 1.5, 1.5])
W2R = np.array([3, 3, 3.5, 3.5, 4, 3, 3, 2.5, 1.5, 1.5, 1.5])
GRADE_CAP = 17.0   # outliers: nobody scores 18-19 across the board -> cap every grade at 17


# ==============================================================================
# 2. FEATURE ENGINEERING & DATA PREPARATION
# ==============================================================================
def _wavg(x, w):
    m = ~np.isnan(x)
    return (np.where(m, x, 0) * w).sum(1) / np.maximum((m * w).sum(1), 1e-9)


def _clean(d):
    d = d.copy()
    d["gender"] = d["gender"].astype(str).str.upper().str.strip()
    b = d["bac_section"].astype(str).str.lower().str.strip()
    d["bac_section"] = b.replace({"technique": "tech", "sciences": "sc", "informatique": "info"})
    bmap = {"yes": 1, "no": 0, "true": 1, "false": 0, "oui": 1, "non": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
    for c in ["club", "redoublant", "transfer", "laptop_owned", "has_github", "knows_the_hood"]:
        if c in d.columns:    # missing stays missing (NOT "no"); the median imputer handles it later
            d[c] = d[c].astype(str).str.lower().str.strip().map(bmap)
    return d


def _allocation_features(X, d):
    """HINT #1 (allocation dynamics): compare every student with the others of the SAME cohort and
    the same bac section (best proxy for the intended track). Official-style formulas are used only
    as a ranking device -- the models learn the rest."""
    X = X.copy()
    cy = d["cohort_year"].values
    sec = X["bac_section"].values
    mg = (_wavg(d[S1].values.astype(float), W1R) + _wavg(d[S2].values.astype(float), W2R)) / 2
    math = (d.an1 + d.alg1 + d.an2 + d.alg2) / 4
    inf = (2 * d.algo1 + 2 * d.algo2 + d.prog1 + d.prog2) / 6
    common = (d.electronique + d.syslog + d.circ) / 3 + (d.meca + d.thermo) / 2
    sc = {"GL": 2 * mg + math + 2 * inf + d.syslog, "RT": 2 * mg + math + inf + d.syslog,
          "IIA": 2 * mg + math + inf + common, "IMI": 2 * mg + math + common}
    sc = {k: np.asarray(v, float) for k, v in sc.items()}
    nat = np.where(sec == "math", "GL", np.where(sec == "tech", "IIA", "RT"))   # natural track by bac section
    ns = np.array([sc[n][i] for i, n in enumerate(nat)])
    s = pd.Series(ns)
    X["P_nat_sc"] = ns
    X["P_nat_rk_cohort"] = s.groupby(cy).rank(pct=True).values
    X["P_nat_rk_sec"] = s.groupby([cy, sec]).rank(pct=True).values
    R = []
    for t in ("GL", "RT", "IIA", "IMI"):
        st = pd.Series(sc[t])
        X["P_rk_sec_" + t] = st.groupby([cy, sec]).rank(pct=True).values
        X["P_rk_all_" + t] = st.groupby(cy).rank(pct=True).values
        R.append(X["P_rk_all_" + t].values)
    for t in ("GL", "RT", "IIA"):
        X["P_ahead_" + t] = pd.Series(sc[t]).groupby(cy).rank(ascending=False).values   # seat pressure
    R = np.column_stack(R)
    X["P_best_track"] = np.nan_to_num(R, nan=0).argmax(1)
    X["P_rk_spread"] = np.nanmax(R, 1) - np.nanmin(R, 1)
    return X


def prepare_features(df_input):
    """Returns a purely numeric feature frame. Every cohort statistic is computed PER COHORT YEAR
    and uses no labels, so it is valid for the unseen test cohorts."""
    raw = df_input.copy()
    # grade-format regime (quarter-point vs two-decimal grades): the pass/fail rule differs between them.
    g0 = raw[GRADES].values.astype(float)
    fmt = ((~np.isnan(g0)) & (np.abs(g0 * 4 - np.round(g0 * 4)) > 1e-6)).any(axis=1).astype(float)
    d = _clean(raw)
    d[GRADES] = d[GRADES].astype(float).clip(upper=GRADE_CAP)           # outlier cap (on a copy)
    X = d.copy()
    g1, g2 = d[S1].values.astype(float), d[S2].values.astype(float)
    X["S1"], X["S2"] = _wavg(g1, W), _wavg(g2, W)                        # replaces the unweighted S1 mean
    X["A"] = (X.S1 + X.S2) / 2
    X["dS"] = X.S2 - X.S1
    X["nz"] = (np.nan_to_num(g1, nan=1) == 0).sum(1) + (np.nan_to_num(g2, nan=1) == 0).sum(1)   # zero grades
    X["nnan"] = np.isnan(g1).sum(1) + np.isnan(g2).sum(1)
    g = np.concatenate([g1, g2], 1)
    X["below10"] = (g < 10).sum(1)
    X["min_g"], X["max_g"], X["std_g"] = np.nanmin(g, 1), np.nanmax(g, 1), np.nanstd(g, 1)
    X["S1_ok"], X["S2_ok"] = (X.S1 >= 10).astype(int), (X.S2 >= 10).astype(int)
    X["dropout"] = (X.nz >= 10).astype(int)
    gz = np.where(g == 0, np.nan, g)
    X["A_nz"] = _wavg(gz, np.tile(W, 2))
    X["pm"] = X[["prog1", "prog2", "algo1", "algo2"]].mean(axis=1)
    X["math"] = X[["an1", "alg1", "an2", "alg2"]].mean(axis=1)
    X["elec"] = X[["circ", "electronique", "syslog", "emag"]].mean(axis=1)
    X["phys"] = X[["meca", "thermo", "estat", "opt"]].mean(axis=1)
    X["lang"] = X[["fr1", "en1", "fr2", "en2", "droit", "eco"]].mean(axis=1)
    for c in ["A", "S1", "S2", "bac_avg", "pm", "math", "elec", "phys"]:      # PER-COHORT percentiles
        X[c + "_rk"] = X.groupby("cohort_year")[c].rank(pct=True)
    X["A_z"] = X.groupby("cohort_year")["A"].transform(lambda s: (s - s.mean()) / s.std())
    X = _allocation_features(X, d)
    X["fmt"] = fmt
    drop_cols = ["id", TARGET_COLUMN, "birth_day_month", "zodiac", "exam_room", "S1_rounded", "governorate", "cohort_year"]
    X = X.drop(columns=drop_cols, errors="ignore")
    return pd.get_dummies(X, columns=["bac_section", "gender"], dtype=float)   # one-hot, no fake ordering


# ==============================================================================
# 4. MODELS
# ==============================================================================
class SeedAvg:
    """Average predict_proba over several seeds (lower variance for tree models)."""
    def __init__(self, factory, seeds):
        self.factory, self.seeds = factory, seeds

    def fit(self, X, y):
        self.ms = [self.factory(sd).fit(X, y) for sd in self.seeds]
        self.factory = None
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.ms], 0)


SEEDS = (0, 1, 2)
HGB_PARAMS = dict(learning_rate=0.05, max_iter=200, max_depth=3, l2_regularization=5.0, min_samples_leaf=30)


def make_models(hgb_params=None):
    hp = dict(HGB_PARAMS, **(hgb_params or {}))
    return {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)),
        "hgb": lambda: HistGradientBoostingClassifier(random_state=42, **hp),
        "extratrees": lambda: SeedAvg(lambda sd: ExtraTreesClassifier(500, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=sd), SEEDS),
        "lgbm": lambda: SeedAvg(lambda sd: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=25, subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5, verbose=-1, random_state=sd), SEEDS),
        "xgb": lambda: SeedAvg(lambda sd: xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.6, reg_lambda=5, min_child_weight=3, random_state=sd, verbosity=0), SEEDS),
        "catboost": lambda: CatBoostClassifier(iterations=600, learning_rate=0.04, depth=5, l2_leaf_reg=6, random_seed=0, verbose=0, thread_count=1, allow_writing_files=False),
    }


def build_preprocessor(columns):
    return ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), list(columns))], remainder="drop")


def main():
    data_dir = next((a for a in sys.argv[1:] if not a.startswith("--")), HERE)
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    print(f"Train dataset shape: {train_df.shape}\nTest dataset shape:  {test_df.shape}")

    X = prepare_features(train_df)
    X_test = prepare_features(test_df).reindex(columns=X.columns, fill_value=0.0)
    y = train_df[TARGET_COLUMN].map({c: i for i, c in enumerate(CLASSES)}).values
    groups = train_df["cohort_year"].values
    pre = build_preprocessor(X.columns).fit(X)                        # median imputation (fit on train)
    Xp = pd.DataFrame(pre.transform(X), columns=X.columns)
    Xtp = pd.DataFrame(pre.transform(X_test), columns=X.columns)
    print("Features:", X.shape[1])

    hgb_params = None
    if "--tune" in sys.argv:
        grid = {"learning_rate": [0.03, 0.06], "max_depth": [3, 4, 6], "l2_regularization": [1.0, 5.0], "max_iter": [150, 300]}
        gs = GridSearchCV(HistGradientBoostingClassifier(random_state=42, min_samples_leaf=30), grid,
                          cv=GroupKFold(3), scoring="f1_macro", n_jobs=-1)
        gs.fit(Xp, y, groups=groups)
        hgb_params = gs.best_params_
        print("Best HGB params (grouped CV):", hgb_params)
    models = make_models(hgb_params)

    if "--cv" in sys.argv:
        oof = {k: np.zeros((len(Xp), 5)) for k in models}
        for a, b in GroupKFold(6).split(Xp, y, groups):                # hold out WHOLE cohort years
            for k, mk in models.items():
                oof[k][b] = mk().fit(Xp.iloc[a], y[a]).predict_proba(Xp.iloc[b])
        print("\n" + "=" * 65 + "\n  YEAR-GROUPED CROSS-VALIDATION (macro-F1)\n" + "=" * 65)
        for k in oof:
            print(f" {k:<11s} {f1_score(y, oof[k].argmax(1), average='macro'):.4f}")
        P = np.mean(list(oof.values()), 0)
        p = P.argmax(1)
        print(f" {'ENSEMBLE':<11s} {f1_score(y, p, average='macro'):.4f}   accuracy={accuracy_score(y, p):.4f}")
        print(classification_report(y, p, target_names=CLASSES, digits=3))
        cm = pd.DataFrame(confusion_matrix(y, p), index=[f"Actual {c}" for c in CLASSES], columns=[f"Pred {c}" for c in CLASSES])
        print(cm)
        # permutation importance on one held-out cohort year (2024) with the HGB model
        tr_i, te_i = np.where(groups != 2024)[0], np.where(groups == 2024)[0]
        m = models["hgb"]().fit(Xp.iloc[tr_i], y[tr_i])
        pi = permutation_importance(m, Xp.iloc[te_i], y[te_i], scoring="f1_macro", n_repeats=3, random_state=42)
        imp = pd.DataFrame({"Feature": X.columns, "Importance_Mean": pi.importances_mean}).sort_values("Importance_Mean", ascending=False)
        print("\nTOP 15 FEATURES (permutation impact on macro-F1, held-out year 2024)")
        print(imp.head(15).to_string(index=False))

    # ==========================================================================
    # 7. FINAL FIT ON ALL LABELLED DATA + SUBMISSION
    # ==========================================================================
    Pt = np.zeros((len(Xtp), 5))
    for k, mk in models.items():
        Pt += mk().fit(Xp, y).predict_proba(Xtp) / len(models)
    sub = pd.DataFrame({"id": test_df["id"], TARGET_COLUMN: [CLASSES[i] for i in Pt.argmax(1)]})
    assert len(sub) == len(test_df) and sub["id"].is_unique and sub[TARGET_COLUMN].isin(CLASSES).all()
    out = os.path.join(data_dir, "submission.csv")
    sub.to_csv(out, index=False)
    print(sub[TARGET_COLUMN].value_counts())
    print("Submission successfully exported to", out)


if __name__ == "__main__":
    main()
