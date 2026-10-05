import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# -------------------------
# LOAD DATA
# -------------------------

df = pd.read_csv("cleaned_accident.csv")


# -------------------------
# DEFINE FEATURES AND TARGET
# -------------------------

X = df.drop("Survived", axis=1)

y = df["Survived"]


# -------------------------
# TRAIN / VALIDATION / TEST SPLIT
# -------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# -------------------------
# COLUMN TYPES
# -------------------------

numeric_features = [
    "Age",
    "Speed_of_Impact",
    "Safety_Score"
]

categorical_features = [
    "Gender",
    "Helmet_Used",
    "Seatbelt_Used",
    "Age_Group",
    "Speed_Category"
]


# -------------------------
# PREPROCESSING
# -------------------------

numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# -------------------------
# MODELS
# -------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Random Forest": RandomForestClassifier(
        random_state=42
    ),

    "KNN": KNeighborsClassifier()
}


# -------------------------
# TRAIN AND COMPARE MODELS
# -------------------------

results = {}

best_model_name = None
best_score = 0
best_pipeline = None


for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    results[name] = accuracy

    print(f"\n{name}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    if accuracy > best_score:

        best_score = accuracy
        best_model_name = name
        best_pipeline = pipeline


# -------------------------
# SHOW MODEL COMPARISON
# -------------------------

print("\nMODEL COMPARISON")

for model, score in results.items():

    print(
        f"{model}: {score:.4f}"
    )


print(
    f"\nBest Model: {best_model_name}"
)


# -------------------------
# TUNE RANDOM FOREST IF IT WINS
# -------------------------

if best_model_name == "Random Forest":

    parameters = {

        "model__n_estimators": [
            50,
            100,
            200
        ],

        "model__max_depth": [
            None,
            5,
            10
        ]
    }

    grid_search = GridSearchCV(
        best_pipeline,
        parameters,
        cv=5,
        scoring="accuracy"
    )

    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_

    print(
        "\nBest Parameters:"
    )

    print(
        grid_search.best_params_
    )


# -------------------------
# FINAL TEST EVALUATION
# -------------------------

test_predictions = best_pipeline.predict(
    X_test
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print(
    f"\nFinal Test Accuracy: {test_accuracy:.4f}"
)

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        test_predictions
    )
)


# -------------------------
# CONFUSION MATRIX
# -------------------------

cm = confusion_matrix(
    y_test,
    test_predictions
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

display.plot()

plt.title(
    "Confusion Matrix"
)

plt.show()


# -------------------------
# SAVE MODEL
# -------------------------

joblib.dump(
    best_pipeline,
    "accident_survival_model.pkl"
)

print(
    "\nModel saved successfully!"
)