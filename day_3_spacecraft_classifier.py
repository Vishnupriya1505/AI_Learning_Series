import numpy as np
import pandas as pd
np.random.seed(42)

# Generate normal spacecraft readings
n_normal = 300
normal = pd.DataFrame({
    "Temperature": np.random.normal(27, 2, n_normal),
    "Voltage": np.random.normal(5.1, 0.15, n_normal),
    "Pressure": np.random.normal(101, 2, n_normal),
    "Vibration": np.random.normal(0.14, 0.04, n_normal),
    "Status": 0
})

# Generate illustrative anomalous readings
n_anomaly = 100
anomaly = pd.DataFrame({
    "Temperature": np.random.normal(40, 4, n_anomaly),
    "Voltage": np.random.normal(4.5, 0.3, n_anomaly),
    "Pressure": np.random.normal(108, 4, n_anomaly),
    "Vibration": np.random.normal(0.55, 0.12, n_anomaly),
    "Status": 1
})

# Combine both classes
df = pd.concat(
    [normal, anomaly],
    ignore_index=True
)

# Shuffle the observations
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print(df.head())
print("\nDataset shape:", df.shape)
print("\nClass distribution:")
print(df["Status"].value_counts())
X = df[
    ["Temperature", "Voltage", "Pressure", "Vibration"]
]
y = df["Status"]
print("Features:")
print(X.head())
print("\nTarget:")
print(y.head())

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)
print("\nTraining class distribution:")
print(y_train.value_counts())
print("\nTesting class distribution:")
print(y_test.value_counts())

# Feature scaling for Logistic Regression and KNN
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_predictions = log_model.predict(X_test_scaled)
print("Logistic Regression predictions:")
print(log_predictions)

# Train KNN
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_predictions = knn_model.predict(X_test_scaled)
print("KNN predictions:")
print(knn_predictions)

#Train Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)
tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)
print("Decision Tree predictions:")
print(tree_predictions)

#Evaluate models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
models = {
    "Logistic Regression": log_predictions,
    "KNN": knn_predictions,
    "Decision Tree": tree_predictions
}
for name, predictions in models.items():

    print("\nModel:", name)

    print("Accuracy:",
          accuracy_score(y_test, predictions))

    print("Precision:",
          precision_score(y_test, predictions,
                          zero_division=0))

    print("Recall:",
          recall_score(y_test, predictions,
                       zero_division=0))

    print("F1 Score:",
          f1_score(y_test, predictions,
                   zero_division=0))

#Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(
    y_test,
    log_predictions,
    labels=[0, 1]
)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Anomaly"],
    yticklabels=["Normal", "Anomaly"]
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(
    classification_report(
        y_test,
        log_predictions,
        target_names=["Normal", "Anomaly"]
    )
)