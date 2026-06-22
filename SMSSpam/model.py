import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("spam.csv", encoding="latin-1")
df.replace('?', np.nan, inplace=True)
df = df.infer_objects()
df2 = df[["v1","v2"]]

df2["v2"] = df2["v2"].str.lower()
df2["char_cnt"] = df2["v2"].str.len()
df2["word_cnt"] = df2["v2"].str.split().str.len()
df2["avg_word_len"] = df2["char_cnt"] / df2["word_cnt"]
print(df2.head())

X = df2[["v2", "char_cnt", "word_cnt", "avg_word_len"]]
y = df2["v1"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tv = TfidfVectorizer(max_features=5000, stop_words='english')
tv.fit(X_train)
train_tv_transformed = tv.transform(X_train)
test_tv_transformed = tv.transform(X_test)

train_tv_df = pd.DataFrame(
    train_tv_transformed.toarray(),
    columns=tv.get_feature_names_out()
).add_prefix("tv_")

examine_row = train_tv_df.iloc[0]
print(examine_row.sort_values(ascending=False).head(15))

text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english'))
])

num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('text', text_transformer, 'v2'),
    ('num', num_transformer, ["char_cnt", "word_cnt", "avg_word_len"])
])

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

param_grid = {
    'preprocessor__text__tfidf__max_features': [2000, 3000, 5000],
    'classifier__C': [0.1, 1, 10]
}

grid_search = GridSearchCV(full_pipeline, param_grid, cv=5)

grid_search.fit(X_train, y_train)

best_score = grid_search.best_score_
predictions = grid_search.predict(X_test)
print(f"Best Accuracy: {best_score}")