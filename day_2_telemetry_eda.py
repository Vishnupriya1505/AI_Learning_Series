import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: Create illustrative spacecraft telemetry data
data = {
    "Time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Temperature": [25.2, 25.5, 25.8, np.nan, 26.1,
                    26.4, 26.7, 45.8, 27.2, 27.5],
    "Voltage": [5.1, 5.2, 5.1, 5.3, np.nan,
                5.2, 5.1, 5.3, 5.2, 5.1],
    "Pressure": [101, 102, 100, 103, 101,
                 102, 101, 104, 102, 101],
    "Vibration": [0.12, 0.14, 0.13, 0.15, 0.14,
                  0.13, 0.15, 0.65, 0.14, 0.13]
}
df = pd.DataFrame(data)
print("----- ORIGINAL TELEMETRY DATA -----")
print(df)

# STEP 2: Inspect the dataset
print("\nDataset shape:", df.shape)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

# STEP 3: Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# STEP 4: Handle missing numerical values
# Use median as a simple demonstration
clean_df = df.copy()
sensor_columns = [
    "Temperature", "Voltage", "Pressure", "Vibration"
]
for column in sensor_columns:
    clean_df[column] = clean_df[column].fillna(
        clean_df[column].median()
    )
print("\nData after missing-value treatment:")
print(clean_df)

# STEP 5: Detect potential outliers using IQR
for column in sensor_columns:
    Q1 = clean_df[column].quantile(0.25)
    Q3 = clean_df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_rows = clean_df[
        (clean_df[column] < lower) |
        (clean_df[column] > upper)
    ]
    print(f"\nPotential outliers in {column}:")
    print(outlier_rows[["Time", column]])

# STEP 6: Correlation analysis
correlation = clean_df.corr(numeric_only=True)
print("\nCorrelation matrix:")
print(correlation)
plt.figure(figsize=(8, 5))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Telemetry Correlation Heatmap")
plt.tight_layout()
plt.show()

# STEP 7: Feature engineering
clean_df["Temperature_Change"] = (
    clean_df["Temperature"].diff()
)
clean_df["Vibration_Change"] = (
    clean_df["Vibration"].diff()
)
clean_df["Temperature_Rolling_Mean"] = (
    clean_df["Temperature"].rolling(window=3).mean()
)
print("\nDataset with engineered features:")
print(clean_df)

# STEP 8: Visualize temperature readings
plt.figure(figsize=(8, 5))
plt.plot(
    clean_df["Time"],
    clean_df["Temperature"],
    marker="o",
    label="Temperature"
)
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.title("Spacecraft Temperature Monitoring")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# STEP 9: Save the processed dataset
clean_df.to_csv("cleaned_telemetry.csv", index=False)
print("\nCleaned telemetry data saved successfully!")