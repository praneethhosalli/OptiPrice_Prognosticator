# Import necessary libraries
import pandas as pd
import numpy as np
import os
from Utils.constants import RAW_DIR, PROCESSED_DIR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load the raw data file
input_file = os.path.join(RAW_DIR, "dynamic_pricing.csv")
df = pd.read_csv(input_file)

# Calculate demand-supply ratio and classify demand/supply levels
df["d_s_ratio"] = round(df["Number_of_Riders"] / df["Number_of_Drivers"], 2)
df["demand_class"] = np.where(df["Number_of_Riders"] > np.percentile(df["Number_of_Riders"], 75), "high_demand", "low_demand")
df["supply_class"] = np.where(df["Number_of_Drivers"] > np.percentile(df["Number_of_Drivers"], 75), "high_supply", "low_supply")

# Compute demand and supply metrics based on demand/supply classes
df["demand_metric"] = np.where(df["demand_class"] == "high_demand",
                               df["Number_of_Riders"] / np.percentile(df["Number_of_Riders"], 75),
                               df["Number_of_Riders"] / np.percentile(df["Number_of_Riders"], 25))
df["supply_metric"] = np.where(df["supply_class"] == "high_supply",
                               df["Number_of_Drivers"] / np.percentile(df["Number_of_Drivers"], 75),
                               df["Number_of_Drivers"] / np.percentile(df["Number_of_Drivers"], 25))

# Define a function to adjust prices based on demand/supply metrics and other conditions
def adjust_price(row):
    high_demand_cutoff = 1.15
    low_demand_cutoff = 0.85
    high_supply_cutoff = 1.15
    low_supply_cutoff = 0.85
    
    # Adjust price based on demand-supply metrics
    if row['demand_metric'] > high_demand_cutoff and row['supply_metric'] < low_supply_cutoff:
        price = row['Historical_Cost_of_Ride'] * 1.10
    elif row['demand_metric'] < low_demand_cutoff and row['supply_metric'] > high_supply_cutoff:
        price = row['Historical_Cost_of_Ride'] * 0.95
    else:
        price = row['Historical_Cost_of_Ride']
    
    # Additional adjustments based on ride conditions
    if row['Average_Ratings'] >= 4.0:
        price *= 1.03
    if row['Time_of_Booking'] == 'Night':
        price *= 1.03
    if row['Vehicle_Type'] == 'Premium':
        price *= 1.03
    
    return price

# Apply the price adjustment function to each row
df["Updated_Price"] = df.apply(adjust_price, axis=1)

# Scale prices to achieve a target total revenue (35% increase from historical cost)
total_historical_cost = df['Historical_Cost_of_Ride'].sum()
required_total_price = total_historical_cost * 1.35
scaling_factor = required_total_price / df["Updated_Price"].sum()
df["Final_Updated_Price"] = df["Updated_Price"] * scaling_factor

# Select features and target variable for model training
X = df[['Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration', 'demand_metric', 'supply_metric']]
y = df['Historical_Cost_of_Ride']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features to improve model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict prices using the trained model
df['Predicted_Price'] = model.predict(scaler.transform(X))

# Save te processed dataset with predictions
output_file = os.path.join(PROCESSED_DIR, "dynamic_pricing_processed_with_predictions.csv")
df.to_csv(output_file, index=False)

print(f"Processed data saved to {output_file}")
