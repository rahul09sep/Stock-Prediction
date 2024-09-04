import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('D:\Desktop\Python\Stock priction\Google_test_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()
data = df['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
X = []
y = []

for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = DecisionTreeRegressor()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse}")

mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse}")

r2 = r2_score(y_test, predictions)
print(f"R²: {r2}")

train = df[:len(X_train) + 60] 
valid = df[len(X_train) + 60:]  
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'], color='green', label='Train')
plt.plot(valid['Close'], color='blue', label='Actual')
plt.plot(valid['Predictions'], color='red', label='Predictions')
plt.legend(loc='lower right')
plt.show()
