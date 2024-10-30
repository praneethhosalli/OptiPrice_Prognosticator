import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
# Load dataset
data = pd.read_csv('dynamic_pricing.csv')
# Set up features and target variable
X = data[['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 'Expected_Ride_Duration']]
Y = data['Historical_Cost_of_Ride']
# Add some synthetic features for experimentation
X['Driver_to_Rider_Ratio'] = X['Number_of_Drivers'] / (X['Number_of_Riders'] + 1)
X['Ride_Duration_Rating_Product'] = X['Expected_Ride_Duration'] * X['Average_Ratings']
# Split data into train, validation, and test sets (60-20-20)
X_temp, X_test, y_temp, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
# Model hyperparameters
lasso_params = {'alpha': [0.01, 0.1, 1, 10]}
ridge_params = {'alpha': [0.01, 0.1, 1, 10]}
svr_params = {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1, 0.5]}
# Initialize models
linear_model = LinearRegression()
lasso_model = GridSearchCV(Lasso(), lasso_params, scoring='neg_mean_squared_error', cv=5)
ridge_model = GridSearchCV(Ridge(), ridge_params, scoring='neg_mean_squared_error', cv=5)
svr_model = GridSearchCV(SVR(kernel='linear'), svr_params, scoring='neg_mean_squared_error', cv=5)
# Fit models and perform cross-validation
models = {
    "Linear Regression": linear_model,
    "Lasso Regression": lasso_model,
    "Ridge Regression": ridge_model,
    "SVR": svr_model
}
validation_results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    validation_mse = mean_squared_error(y_val, y_val_pred)
    validation_results[model_name] = validation_mse
    print(f"{model_name} Validation MSE: {validation_mse:.2f}")
# Display best parameters for models with tuning
if hasattr(lasso_model, 'best_params_'):
    print("Best Lasso Params:", lasso_model.best_params_)
if hasattr(ridge_model, 'best_params_'):
    print("Best Ridge Params:", ridge_model.best_params_)
if hasattr(svr_model, 'best_params_'):
    print("Best SVR Params:", svr_model.best_params_)
# Evaluate models on the test set
test_results = {}
for model_name, model in models.items():
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_results[model_name] = {
        "Predictions": y_test_pred,
        "MSE": test_mse
    }
# Save results to a DataFrame
results_df = pd.DataFrame({'Actual': y_test})
for model_name, result in test_results.items():
    results_df[f'Predicted_{model_name}'] = result['Predictions']
    results_df[f'Error_{model_name}'] = results_df['Actual'] - result['Predictions']
# Save the results DataFrame to a CSV file
results_df.to_csv('updated_ride_cost_predictions.csv', index=False)  
# Display the first few rows of the results
print(results_df.head())  