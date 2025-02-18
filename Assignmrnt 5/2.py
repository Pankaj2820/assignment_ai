import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('50_Startups.csv')

# Drop the categorical 'State' column
df = data.drop(columns='State')

# Compute and visualize correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Select features based on correlation analysis
X = df[['R&D Spend', 'Marketing Spend']]
y = df['Profit']

# Scatter plots for selected features
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(df['R&D Spend'], df['Profit'], color='blue', alpha=0.5)
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.title('R&D Spend vs Profit')

plt.subplot(1, 2, 2)
plt.scatter(df['Marketing Spend'], df['Profit'], color='red', alpha=0.5)
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.title('Marketing Spend vs Profit')

plt.tight_layout()
plt.show()

# Split dataset into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict values for training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Compute RMSE and R² scores
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)

rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

# Print results
print(f"Training Data  -> RMSE: {rmse_train:.2f}, R²: {r2_train:.2%}")
print(f"Testing Data   -> RMSE: {rmse_test:.2f}, R²: {r2_test:.2%}")
