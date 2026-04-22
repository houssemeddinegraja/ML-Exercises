import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. IMPORT REGRESSION SPECIFIC TOOLS
# We use Ridge (Regularized Linear Regression) and regression metrics
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

# 2. LOAD & PRE-CLEAN
df = pd.read_csv("insurance.csv")
df.drop_duplicates(inplace=True)

df["smoker"] = df["smoker"].replace({"yes": 1, "no": 0}).astype(int)
df["bmi*smoker"] = df["smoker"]*df["bmi"]
print(df.head())
print(df.info())

X = df.drop('expenses', axis=1)
y = df['expenses']

# 4. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. IDENTIFY COLUMNS
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

# 6. PREPROCESSING (The "Mini-Pipelines")
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Median is often safer for predicting money/prices
    ('scaler', StandardScaler())                   # Scaling is CRITICAL for Ridge Regression
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# 7. THE COLUMN TRANSFORMER
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_cols),
    ('cat', cat_transformer, categorical_cols)
])

# 8. THE MASTER PIPELINE (Preprocessing + Regression Model)
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge())  # <--- The engine is now a Regressor!
])

# 9. HYPERPARAMETER TUNING (Grid Search)
# Tuning the 'alpha' penalty term in Ridge Regression
param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'regressor__alpha': [0.1, 1.0, 10.0, 100.0]  # <--- Tuning the Regression model
}

# Notice we changed the scoring metric! 'r2' is standard for regression
grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='r2')

# 10. EXECUTE EVERYTHING
grid_search.fit(X_train, y_train)

# 11. RESULTS & EVALUATION
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

# For Regression, we look at R-squared and Root Mean Squared Error (RMSE)
r2 = r2_score(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"R-squared Score (Closer to 1 is better): {r2:.4f}")
print(f"RMSE (Average error in the units of the target): {rmse:.2f}")
# =====================================================================
# 12. RANDOM FOREST COMPARISON
# =====================================================================
print("\n" + "="*50)
print("TRAINING RANDOM FOREST REGRESSOR...")
print("="*50)

# 12a. Build the Random Forest Pipeline (Reusing the exact same preprocessor!)
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# 12b. Hyperparameters for the Forest
# We keep the grid relatively small so it doesn't take 10 minutes to run
rf_param_grid = {
    'regressor__n_estimators': [100, 200],   # Number of trees
    'regressor__max_depth': [None, 5, 10],   # How deep the trees can go
    'regressor__min_samples_split': [2, 5]   # When to stop splitting nodes
}

# 12c. Run the Grid Search (n_jobs=-1 uses all your computer's cores to speed it up)
rf_grid_search = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='r2', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

# 12d. Evaluate Random Forest
rf_best_model = rf_grid_search.best_estimator_
rf_predictions = rf_best_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_predictions)
rf_rmse = root_mean_squared_error(y_test, rf_predictions)

print(f"RF Best Parameters: {rf_grid_search.best_params_}")
print(f"RF R-squared Score: {rf_r2:.4f}")
print(f"RF RMSE:            {rf_rmse:.2f}")
"""

print(df.values[1])
print(df.head())
print(df.info())

print(df["children"].value_counts())

"""