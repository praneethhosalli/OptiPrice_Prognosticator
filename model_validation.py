from sklearn.metrics import root_mean_squared_error

def validate_model(model, X_val, y_val):
    y_pred_val = model.predict(X_val)
    val_error = root_mean_squared_error(y_val, y_pred_val)
    return val_error
