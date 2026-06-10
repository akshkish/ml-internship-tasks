import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

feature_names = data.feature_names

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# Hyperparameter Tuning
# -------------------------
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

# -------------------------
# Cross-validation score
# -------------------------
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)

print("\nCross-validation Accuracy:")
print("Mean:", cv_scores.mean())
print("Std:", cv_scores.std())

# -------------------------
# Model Evaluation
# -------------------------
y_pred = best_model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------
# Feature Importance
# -------------------------
importances = best_model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feature_df = feature_df.sort_values(by="Importance", ascending=False)

print("\nTop Features:")
print(feature_df.head(10))

# Plot feature importance
plt.figure(figsize=(8,5))
plt.barh(feature_df["Feature"][:10][::-1], feature_df["Importance"][:10][::-1])
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.show()