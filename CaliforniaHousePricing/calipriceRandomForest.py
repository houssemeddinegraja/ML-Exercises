import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
housing_data = fetch_california_housing(as_frame=True)

df = housing_data.frame

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_cols),
    ('cat', cat_transformer, categorical_cols)
])

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(42))
])

param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'regressor__n_estimators': [50, 100],
    'regressor__max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='r2')

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

r2 = r2_score(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"Best Parameters: {grid_search.best_params_}")
print(f"R-squared Score (Closer to 1 is better): {r2:.4f}")
print(f"RMSE (Average error in the units of your target): {rmse:.2f}")
