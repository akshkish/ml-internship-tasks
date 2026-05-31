import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Read space-separated data
df = pd.read_csv("housing.csv", header=None, sep=r"\s+")

print("Dataset Shape:", df.shape)
print(df.head())

# Features = all columns except last
X = df.iloc[:, :-1]

# Target = last column
y = df.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nIntercept:")
print(model.intercept_)

print("\nCoefficients:")
for i, coef in enumerate(model.coef_):
    print(f"Feature {i+1}: {coef}")

print("\nR-squared:", r2)
print("Mean Squared Error:", mse)