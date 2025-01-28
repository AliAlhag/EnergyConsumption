# Disable warnings
import warnings
warnings.filterwarnings("ignore")

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Preprocessing
# Assume df is the DataFrame containing the dataset
# Copy the dataset to avoid modifying the original
df_model = df.copy()

# Encode categorical variables
label_encoders = {}
for col in ["BuildingType", "HVACSystem"]:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le  # Store encoders for later use

# Define features and target variable
X = df_model.drop(columns=["EnergyConsumption"])
y = df_model["EnergyConsumption"]

# Step 2: Split into Train, Validation, and Test Sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Step 3: Train LightGBM Model
model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

# Train on the training set
model.fit(X_train, y_train)

# Step 4: Evaluate on Validation Set
y_val_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

print("\nValidation Set Performance:")
print(f"MAE: {val_mae:.4f}")
print(f"MSE: {val_mse:.4f}")
print(f"R² Score: {val_r2:.4f}")

# Step 5: Evaluate on Test Set
y_test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTest Set Performance:")
print(f"MAE: {test_mae:.4f}")
print(f"MSE: {test_mse:.4f}")
print(f"R² Score: {test_r2:.4f}")

# Step 6: Visualize Model Performance
results_df = pd.DataFrame({
    "Dataset": ["Validation", "Test"],
    "MAE": [val_mae, test_mae],
    "MSE": [val_mse, test_mse],
    "R² Score": [val_r2, test_r2]
})

# Plot R² Score for Validation and Test Sets
plt.figure(figsize=(8, 5))
sns.barplot(x="Dataset", y="R² Score", data=results_df)
plt.title("Validation vs Test Set Performance (R² Score)")
plt.ylabel("R² Score")
plt.show()
