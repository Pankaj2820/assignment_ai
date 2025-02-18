import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge, Lasso

# Load dataset
data = pd.read_csv('Auto.csv')

# Convert 'horsepower' to numeric and handle missing values
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
data.dropna(inplace=True)

# Select independent variables and target variable
X = data[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']]
y = data['mpg']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a range of alpha values
alphas = np.logspace(-4, 2, 500)

# Store R² scores for Ridge and Lasso
r2_ridge, r2_lasso = [], []

# Train Ridge and Lasso models for each alpha and store R² scores
for alpha in alphas:
    ridge_model = Ridge(alpha=alpha).fit(X_train, y_train)
    lasso_model = Lasso(alpha=alpha).fit(X_train, y_train)

    r2_ridge.append(r2_score(y_test, ridge_model.predict(X_test)))
    r2_lasso.append(r2_score(y_test, lasso_model.predict(X_test)))

# Get the best alpha values
best_alpha_ridge = alphas[np.argmax(r2_ridge)]
best_alpha_lasso = alphas[np.argmax(r2_lasso)]

# Train Ridge and Lasso with the best alpha values
final_ridge = Ridge(alpha=best_alpha_ridge).fit(X_train, y_train)
final_lasso = Lasso(alpha=best_alpha_lasso).fit(X_train, y_train)

# Predict and compute R² and RMSE scores
ridge_r2, ridge_rmse = r2_score(y_test, final_ridge.predict(X_test)), np.sqrt(mean_squared_error(y_test, final_ridge.predict(X_test)))
lasso_r2, lasso_rmse = r2_score(y_test, final_lasso.predict(X_test)), np.sqrt(mean_squared_error(y_test, final_lasso.predict(X_test)))

# Print the best alpha values and evaluation metrics
print(f"Ridge -> Best Alpha: {best_alpha_ridge:.4f}, R²: {ridge_r2:.4f}, RMSE: {ridge_rmse:.4f}")
print(f"Lasso -> Best Alpha: {best_alpha_lasso:.4f}, R²: {lasso_r2:.4f}, RMSE: {lasso_rmse:.4f}")

# Plot R² scores for Ridge and Lasso
sns.set_style("whitegrid")
plt.figure(figsize=(10, 5))

plt.plot(alphas, r2_ridge, label='Ridge Regression', color='blue')
plt.plot(alphas, r2_lasso, label='Lasso Regression', color='red')

plt.xscale('log')  # Log scale for better visualization
plt.xlabel('Alpha')
plt.ylabel('R² Score')
plt.title('R² Scores for Ridge and Lasso Regression')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
