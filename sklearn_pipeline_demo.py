import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder ,StandardScaler

#----------------------------------------------
# 1. Example dataset
#----------------------------------------------

data = pd.DataFrame({
    "age": [22, 25, 47, 52, 46, 56, 23, 35],
    "income": [2500, 3000, 7000, 9000, 8500, None, 2800, 5000],
    "city": ["Dubai", "Abu Dhabi", "Dubai", "Sharjah", "Dubai", "Abu Dhabi", None, "Sharjah"],
    "plan": ["basic", "basic", "premium", "premium", "premium", "basic", "basic", "premium"],
    "defaulted": [1, 1, 0, 0, 0, 0, 1, 0]
})

#----------------------------------------------
# 2. Separate features and target
#----------------------------------------------

x = data.drop(columns="defaulted")
y = data["defaulted"]

#----------------------------------------------
# 3. Define column types
#----------------------------------------------

numeric_features = ["age", "income"]
categorical_features = ["city", "plan"]

#----------------------------------------------
# 4. Numeric preprocessing
#----------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

#----------------------------------------------
# 5. Categorial preprocessing
#----------------------------------------------

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

#----------------------------------------------
# 6. Combine preprocessing for different column types
#----------------------------------------------

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

#----------------------------------------------
# 7. Build final full pipeline
#----------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )),
])

#----------------------------------------------
# 8. Train-test split
#----------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#----------------------------------------------
# 9. Train the full pipeline
#----------------------------------------------

model.fit(x_train, y_train)

#----------------------------------------------
# 10. Predict on test data
#----------------------------------------------

y_pred = model.predict(x_test)

#----------------------------------------------
# 11. Evaluate
#----------------------------------------------

print("Accuracy:", accuracy_score(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))

#----------------------------------------------
# 12. Cross-validation
#----------------------------------------------

scores = cross_val_score(
    model,
    x,
    y,
    cv=3,
    scoring="accuracy"
)

print("Cross-validation scores:", scores)
print("Mean CV accuraacy:", scores.mean())