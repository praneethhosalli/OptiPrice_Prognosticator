import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

file_path = 'G:\\Spring board internship\\OptiPrice_Prognosticator_Infosys_Internship_Oct2024\\Data\\Raw\\dynamic_pricing.csv'
data = pd.read_csv(file_path)

# X = data.select_dtypes(include='number')
# X = data.drop(columns=['Historical_Cost_of_Ride'])
X = data.drop(columns=['Historical_Cost_of_Ride']).select_dtypes(include='number')
Y = data['Historical_Cost_of_Ride']

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)


Y_val_pred = model.predict(X_val)
val_mse = mean_squared_error(Y_val, Y_val_pred)
print(f'Validation Mean Squared Error: {val_mse}')


Y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(Y_test, Y_test_pred)
print(f'Test Mean Squared Error: {test_mse}')


output_dir = 'G:\\Spring board internship\\OptiPrice_Prognosticator_Infosys_Internship_Oct2024\\Data\\Processed'
os.makedirs(output_dir, exist_ok=True)  


results = X_test.copy()
results['Actual'] = Y_test
results['Predicted'] = Y_test_pred
results['Error'] = results['Actual'] - results['Predicted']


output_file_path = os.path.join(output_dir, 'dynamic_pricing_results.csv')  
try:
    results.to_csv(output_file_path, index=False)
    print("CSV file saved successfully at:", output_file_path)
except Exception as e:
    print("Error saving CSV:", e)



# import numpy as np

# # Define a custom input with values for each feature
# custom_input = np.array([[70, 28, 55, 4.85, 160,]])  # Replace these with your actual test values

# # Reshape to ensure it has the right dimensions (1 row, number of features)
# custom_input = custom_input.reshape(1, -1)

# # Predict using the trained model
# predicted_cost = model.predict(custom_input)
# print("Predicted Historical Cost of Ride:", predicted_cost[0])

