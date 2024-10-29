from sklearn.metrics import mean_squared_error, root_mean_squared_error  # Make sure to import root_mean_squared_error


def test_model(model, X_test, y_test):
    # Make predictions
    y_pred_test = model.predict(X_test)

    # Calculate RMSE
    test_error = root_mean_squared_error(y_test, y_pred_test)  # Call without 'squared' argument

    results = X_test.copy()
    results['Historical_Cost_of_Ride'] = y_test
    results['Predicted_Cost_of_Ride'] = y_pred_test
    results['Error'] = results['Historical_Cost_of_Ride'] - results['Predicted_Cost_of_Ride']

    # Save results to CSV
    results.to_csv('test_results_with_error.csv', index=False)

    return test_error
