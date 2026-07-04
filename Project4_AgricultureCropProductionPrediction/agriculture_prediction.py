import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("datafile (1).csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# -------------------------
# Remove Missing Values
# -------------------------
df = df.dropna()

# -------------------------
# Encode Categorical Columns
# -------------------------
le_crop = LabelEncoder()
le_state = LabelEncoder()

df["Crop"] = le_crop.fit_transform(df["Crop"])
df["State"] = le_state.fit_transform(df["State"])

# -------------------------
# Features and Target
# -------------------------
X = df.drop(columns=["Yield (Quintal/ Hectare) "])
y = df["Yield (Quintal/ Hectare) "]

# -------------------------
# Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# Train Model
# -------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------
# Prediction
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# Evaluation
# -------------------------
print("\nModel Performance")

print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R2 Score :", r2_score(y_test, y_pred))

# -------------------------
# Feature Importance
# -------------------------
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance.sort_values().plot(
    kind="barh",
    figsize=(8,5)
)

plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# -------------------------
# Actual vs Predicted
# -------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Yield")

plt.show()