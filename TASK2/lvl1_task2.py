import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Try different K values
k_values = [1, 3, 5, 7, 9]
results = []

best_k = None
best_acc = 0

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results.append((k, acc))

    print(f"\n================ K = {k} ================")
    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Track best model
    if acc > best_acc:
        best_acc = acc
        best_k = k
        best_model = model
        best_pred = y_pred

# Results table
results_df = pd.DataFrame(results, columns=["K Value", "Accuracy"])
print("\n📊 Accuracy Comparison:")
print(results_df)

# Plot accuracy vs K
plt.figure(figsize=(6,4))
sns.lineplot(data=results_df, x="K Value", y="Accuracy", marker="o")
plt.title("K Value vs Accuracy")
plt.show()

# Confusion Matrix (Best Model)
cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title(f"Confusion Matrix (Best K = {best_k})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\n🏆 Best K Value:", best_k)
print("🏆 Best Accuracy:", best_acc)