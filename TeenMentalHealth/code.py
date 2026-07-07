import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("Teen_Mental_Health_Dataset.csv")

mapping = {"low": 0, "medium": 1, "high": 2}
df["social_interaction_level"] = df["social_interaction_level"].str.lower().map(mapping)

X = df.drop(columns=['depression_label'])
y = df['depression_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

num_transformer = SklearnPipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_transformer = SklearnPipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_cols),
    ('cat', cat_transformer, categorical_cols)
])

full_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 5, 10]
}

grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

probabilities = best_model.predict_proba(X_test)[:, 1]
custom_threshold = 0.25
y_pred_custom = (probabilities >= custom_threshold).astype(int)

custom_accuracy = accuracy_score(y_test, y_pred_custom)

print("\n" + "="*55)
print(f"Test Accuracy : {custom_accuracy:.4f}")
print("="*55)
print(classification_report(y_test, y_pred_custom, digits=4))
print("="*55)
print("CONFUSION MATRIX:")
print("="*55)
print(confusion_matrix(y_test, y_pred_custom))
print("="*55 + "\n")