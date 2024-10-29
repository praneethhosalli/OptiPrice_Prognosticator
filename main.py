from data_preprocessing import load_and_preprocess_data
from model_training import train_model
from model_validation import validate_model
from model_testing import test_model

# Load and preprocess data
file_path = 'Dataset/dynamic_pricing.csv'
X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(file_path)

# Train the model
model = train_model(X_train, y_train)

# Validate the model
validation_error = validate_model(model, X_val, y_val)
print(f'Validation RMSE: {validation_error}')

# Test the model and save results
test_error = test_model(model, X_test, y_test)
print(f'Test RMSE: {test_error}')





