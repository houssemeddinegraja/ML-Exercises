import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from ydata_profiling import ProfileReport

# ------------------------------------------------------------------
# CONFIG — edit these two lines per dataset
# ------------------------------------------------------------------
DATA_PATH = "train.csv"
SUBMISSION_PATH = "test.csv"
TARGET_COLUMN = "price_range"

def clean_and_engineer(df):
    """Same cleaning/engineering logic applied identically to train and test data."""
    df = df.copy()
    df.replace('?', np.nan, inplace=True)
    df = df.infer_objects()

    zero_as_missing_cols = ['pc', 'sc_w', 'fc']
    df[zero_as_missing_cols] = df[zero_as_missing_cols].replace(0, np.nan)

    df['px_count'] = df['px_height'] * df['px_width']
    df.drop(columns=['px_height', 'px_width'], inplace=True)

    df['sc_area'] = df['sc_w'] * df['sc_h']
    df.drop(columns=['sc_w', 'sc_h'], inplace=True)

    df['inconsistent_4g_3g'] = ((df['four_g'] == 1) & (df['three_g'] == 0)).astype(int)
    df.drop(columns=['four_g', 'three_g'], inplace=True)

    return df

# 1. LOAD
df_train_full = pd.read_csv(DATA_PATH)
df_submission_raw = pd.read_csv(SUBMISSION_PATH)

# test.csv has no target column — pull its id column aside before cleaning so it
# doesn't get treated as a feature, but keep it to label the final submission
submission_ids = df_submission_raw['id'] if 'id' in df_submission_raw.columns else None
df_submission_raw = df_submission_raw.drop(columns=['id'], errors='ignore')

# 2. CLEAN & ENGINEER — identical function on both, so columns line up exactly
df_train_full = clean_and_engineer(df_train_full)
df_submission = clean_and_engineer(df_submission_raw)

print(df_train_full.shape)
print(df_train_full.head())

# 3. SEPARATE FEATURES & TARGET
X_full = df_train_full.drop(columns=[TARGET_COLUMN])
y_full = df_train_full[TARGET_COLUMN]

print("Class balance:\n", y_full.value_counts(normalize=True), "\n")

# 4. TRAIN/TEST SPLIT — carved out of train.csv, since test.csv has no labels to evaluate against
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)
# 4. IDENTIFY COLUMNS
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# 5. PREPROCESSING ("Mini-Pipelines")
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# 6. COLUMN TRANSFORMER
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_cols),
    ('cat', cat_transformer, categorical_cols)
])

# 7. BASELINE — what's the floor we need to beat?
baseline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DummyClassifier(strategy='most_frequent'))
])
baseline.fit(X_train, y_train)
baseline_f1 = f1_score(y_test, baseline.predict(X_test), average='weighted')
print(f"Baseline (most-frequent) weighted F1: {baseline_f1:.4f}\n")

# 8. MASTER PIPELINE (Preprocessing + Model)
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# 9. HYPERPARAMETER TUNING (Grid Search)
param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__C': [0.1, 1, 10],
    'classifier__class_weight': [None, 'balanced'],  # try if classes are imbalanced
}

grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='f1_weighted')

# 10. EXECUTE
grid_search.fit(X_train, y_train)

# 11. INSPECT THE SEARCH ITSELF, not just the final score
print("Best params:", grid_search.best_params_)
print(f"Best CV f1_weighted: {grid_search.best_score_:.4f}\n")
cv_results = pd.DataFrame(grid_search.cv_results_).sort_values('rank_test_score')
print("Top 5 parameter combinations:")
print(cv_results[['params', 'mean_test_score', 'std_test_score']].head(5).to_string(index=False))
print()

# 12. RESULTS & FORMATTED EVALUATION
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
test_accuracy = best_model.score(X_test, y_test)

print("\n" + "=" * 55)
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"(Baseline was : {baseline.score(X_test, y_test):.4f})")
print("=" * 55)
print(classification_report(y_test, predictions))
print("=" * 55)
print("CONFUSION MATRIX:")
print("=" * 55)
print(confusion_matrix(y_test, predictions))
print("=" * 55 + "\n")

# 13. CONFUSION MATRIX PLOT
ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# 14. FEATURE COEFFICIENTS (linear model — directly interpretable)
if hasattr(best_model.named_steps['classifier'], 'coef_'):
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    coefs = best_model.named_steps['classifier'].coef_[0]
    coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefs})
    coef_df = coef_df.reindex(coef_df.coefficient.abs().sort_values(ascending=False).index)
    print("Top features by |coefficient|:")
    print(coef_df.head(15).to_string(index=False))

# 15. PERSIST THE MODEL
joblib.dump(best_model, "classification_model.pkl")
print("\nModel saved to classification_model.pkl")

# 16. RETRAIN ON ALL LABELED DATA, THEN PREDICT THE REAL TEST SET
final_model = grid_search.best_estimator_
final_model.fit(X_full, y_full)
submission_predictions = final_model.predict(df_submission)

submission_df = pd.DataFrame({
    'id': submission_ids,
    TARGET_COLUMN: submission_predictions
})
submission_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")