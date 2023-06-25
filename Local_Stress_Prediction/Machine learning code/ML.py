import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Read the dataset, specify the encoding as GBK
data = pd.read_csv('date-2.csv', encoding='GBK')

# Extract features and target variable
X = data[['T', 't', 'R', 'r', 'a', 'P']]
y = data['stress']

# Data standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Construct the Gradient Boosting Regressor model
model = GradientBoostingRegressor(random_state=42)

# Define an expanded range of hyperparameters
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.2, 0.1, 0.05, 0.01],
    'max_depth': [1, 3, 5, 7, 9],
    'min_samples_split': [2, 4, 6,8],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Output the best hyperparameter combination
print("Best Hyperparameters:", grid_search.best_params_)

# Train the model with the best hyperparameters
model = GradientBoostingRegressor(random_state=42, **grid_search.best_params_)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared:", r2)

# Evaluate model performance using cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
mean_cv_score = np.mean(cv_scores)
print("Cross-validated R-squared:", mean_cv_score)

# Save the model as a .pkl file
with open('model-2.pkl', 'wb') as file:
    pickle.dump(model, file)

# Plotting predicted vs. true values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Gradient Boosting Regressor - True vs. Predicted Values')
plt.show()