## scikit-learn Pipeline with Preprocessing

This project builds a proper scikit-learn Pipeline with preprocessing.

In real machine learning projects, raw data usually contains different types of columns.

Examples:

- numeric columns
- categorical columns
- missing values

A Pipeline helps combine preprocessing and modeling into one reliable workflow.

### Why Pipelines Are Important

Preprocessing should not be done separately before model training.

If preprocessing is fitted on the full dataset before the train-test split, information from the test data can leak into the training process.

This is called data leakage.

Data leakage can make model performance look better than it really is.

A proper Pipeline prevents this by fitting preprocessing only on the training data.

### Numeric Preprocessing

Numeric columns may contain missing values and different scales.

Example numeric columns:

- age
- income

The numeric preprocessing pipeline uses:

- SimpleImputer
- StandardScaler

SimpleImputer fills missing numeric values.

Example:

missing income → median income

StandardScaler scales numeric values using:

z = (x - mean) / standard deviation

This is useful for models that are sensitive to feature scale.

### Categorical Preprocessing

Categorical columns contain text or category values.

Example categorical columns:

- city
- plan

The categorical preprocessing pipeline uses:

- SimpleImputer
- OneHotEncoder

SimpleImputer fills missing categorical values using the most frequent category.

OneHotEncoder converts categories into numeric binary columns.

Example:

city = Dubai

becomes:

city_Dubai = 1

### ColumnTransformer

ColumnTransformer applies different preprocessing steps to different columns.

Example:

numeric columns → imputer + scaler

categorical columns → imputer + one-hot encoder

This is useful because numeric and categorical columns cannot be processed in the same way.

### Full Pipeline

The final Pipeline contains:

1. Preprocessor
2. Classifier

Example:

raw data
→ preprocessing
→ model
→ prediction

The model is trained by calling:

model.fit(X_train, y_train)

The model predicts by calling:

model.predict(X_test)

The Pipeline automatically applies the same preprocessing steps during prediction.

### Cross-Validation

Pipelines are especially important during cross-validation.

During each fold, preprocessing is fitted only on the training part of that fold.

The validation fold remains unseen.

This gives a more honest estimate of model performance.

### Why This Workflow Works

This workflow works because it keeps the full machine learning process organized, reproducible, and safe.

All transformations are stored inside one object.

The same preprocessing used during training is also used during testing and future prediction.

This reduces mistakes and prevents data leakage.