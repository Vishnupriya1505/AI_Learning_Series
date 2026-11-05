import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import( mean_squared_error,r2_score,mean_absolute_error)
data = {
    "Time_Minutes" : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    "Temperature" : [20.5,21.0,21.2, 22.1, 22.4,
                    23.0, 23.3, 24.1, 24.4, 25.0,
                    25.2, 26.1, 26.4, 27.0, 27.5]
}
df = pd.DataFrame(data)
print(df)
print("Dataset shape:", df.shape)
print("dataset information:")
print(df.describe())
plt.scatter(df["Time_Minutes"], df["Temperature"])
plt.xlabel("Time (minutes)")
plt.ylabel("Temperature (°C)")
plt.title("Spacecraft Telemetry: Time vs Temperature")
plt.show()
# Separate features and target
X = df[["Time_Minutes"]]
y = df["Temperature"]
#  Split the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("Training rows:",len(X_train))
print("Testing rows:",len(X_test))
# Create the model
model = LinearRegression()
# Train using only training data
model.fit(X_train, y_train)
#  Predict the testing data
y_pred = model.predict(X_test)
print("\nActual temperatures:")
print(y_test.to_numpy())
print("\nPredicted temperatures:")
print(y_pred)
#  Understand the learned equation
print("\nSlope:", model.coef_[0])
print("Intercept:", model.intercept_)
# 8. Evaluate the predictions
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\n--- Model Evaluation ---")
print("MAE:", round(mae, 3))
print("MSE:", round(mse, 3))
print("RMSE:", round(rmse, 3))
print("R² Score:", round(r2, 3))
new_data = pd.DataFrame({"Time_Minutes": [16]})
prediction = model.predict(new_data)
print("Predicted temperature:", round(prediction[0], 2), "°C")