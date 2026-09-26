#!/usr/bin/env python3
"""
predict_submission.py -- score Kaggle's unlabeled test.csv with the model.pkl saved by
mlcc_workflow.py, and write a submission.csv in Kaggle's (Id, SalePrice) format.

mlcc_workflow.py only trains and evaluates on one labeled file (it makes its own train/
validation/test split), so it never touches test.csv. This script is the missing second
half: it reuses the exact fitted recipe (imputer medians, scalers, one-hot vocabularies)
from model.pkl so test.csv is transformed identically to how training data was, then
applies the trained weights.

Usage:
    python mlcc_workflow.py train.csv --label SalePrice --categorical mssubclass,mosold \\
        --filter "grlivarea < 4000" --log-label      # writes model.pkl
    python predict_submission.py test.csv model.pkl submission.csv
"""
import importlib.util
import pickle
import sys

import numpy as np
import pandas as pd


def load_workflow_module(path="mlcc_workflow.py"):
    spec = importlib.util.spec_from_file_location("mlcc_workflow", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # safe: the script only runs main() under __main__
    return mod


def main():
    test_path = sys.argv[1] if len(sys.argv) > 1 else "test.csv"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "model.pkl"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "submission.csv"

    wf = load_workflow_module()
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    model, recipe, task = saved["model"], saved["recipe"], saved["task"]
    y_mu, y_sd, log_label = saved["y_mu"], saved["y_sd"], saved["log_label"]

    raw = pd.read_csv(test_path)
    id_col = next((c for c in raw.columns if c.lower() == "id"), None)
    ids = raw[id_col].copy() if id_col else pd.Series(range(len(raw)), name="Id")

    df = raw.copy()
    df.columns = [wf.clean_name(c) for c in df.columns]   # match training's column names

    missing = [c for c in recipe["num_cols"] + recipe["raw_cat_cols"] if c not in df.columns]
    if missing:
        raise SystemExit(f"test.csv is missing columns the model was trained on: {missing}")

    X, _ = wf.step15_build_features(df, recipe)            # the SAME fitted recipe as training
    pred = wf.predict(X, model["w"], model["b"], task) * y_sd + y_mu   # back to the label's scale
    if log_label:
        pred = np.expm1(pred)                              # undo log1p from training

    out = pd.DataFrame({id_col or "Id": ids, "SalePrice": pred})
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(out)} rows | predicted price range "
          f"${out['SalePrice'].min():,.0f} - ${out['SalePrice'].max():,.0f}, "
          f"median ${out['SalePrice'].median():,.0f}")


if __name__ == "__main__":
    main()
