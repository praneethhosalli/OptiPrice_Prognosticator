# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
df= pd.read_csv(r"C:\Users\sanoj\Downloads\dynamic_pricing (1).csv")


## Calculating Demand_Supply_Ratio and picking demand_supply_threshold = 2.3 arround the mean of Demand_Supply_Ratio
# 1. Higher Demand = when 'Demand_Supply_Ratio' > demand_supply_threshold (2.3) else Low-demand
# 2. Higher supply = when 'Demand_Supply_Ratio' < demand_supply_threshold (2.3) else Low-supply

df['Demand_Supply_Ratio'] = df['Number_of_Riders'] / df['Number_of_Drivers']
demand_supply_threshold = 2.3

df['Demand_class'] = np.where(df['Demand_Supply_Ratio'] > demand_supply_threshold, "Higher_demand", "Lower_demand")
df['Supply_class'] = np.where(df['Demand_Supply_Ratio'] < demand_supply_threshold, "Higher_supply", "Lower_supply")



## calulation Base Price and Surge_charge based on supply demand ratio and demand_supply_factor
# 1. Calculate base historical cost based on expected_Ride_duration
# 2. Calculate rider-to-driver ratio
# 3. Calculate demand-supply factor
# 4. Defining a methode to Calculate supply_demand_surge and Apply the dynamic pricing formula


constant_rate = 3.5  # Define the base rate per unit of duration, this is arround mean of ratio of ('Historical_Cost_of_Ride'/'Expected_Ride_Duration')
demand_hike = 0.35  # This is how much demand increase the pricing

# Calculate base historical cost based on expected_Ride_duration
df['base_cost'] = df['Expected_Ride_Duration'] * constant_rate

# Calculate rider-to-driver ratio
df['rider_driver_ratio'] = df['Number_of_Riders'] / df['Number_of_Drivers']

# Calculate demand-supply factor
df['demand_supply_factor'] = df['rider_driver_ratio'] - 1
df['demand_supply_factor'] = df['demand_supply_factor'].apply(lambda x: min(x, 6))

# defining a methode to Calculate supply_demand_surge and Apply the dynamic pricing formula
def apply_surge(df):
    SD_surge_charge=0
    if (df['Demand_class']=='Higher_demand') & (df['Supply_class']=='Lower_supply'):
        SD_surge_charge = df['base_cost'] * (demand_hike * df['demand_supply_factor'])
    return SD_surge_charge

df['S/D_surge_charge'] = df.apply(apply_surge,axis=1)

# Conditional Surge based on Vehical_Type and Time_of_booking && Location_Category Condition

def cal_surge_charge(df):
    surge_charge = 0
    if df['Vehicle_Type'] == 'Premium':
        if (df['Location_Category'] in ('Urban', 'Suburban')) & (df['Time_of_Booking'] in ('Evening', 'Night')):
            surge_charge = df['base_cost'] * 0.05 + df['base_cost'] * 0.02
    else:
        if (df['Location_Category'] in ('Urban', 'Suburban')) & (df['Time_of_Booking'] in ('Evening', 'Night')):
            surge_charge = df['base_cost'] * 0.025 + df['base_cost'] * 0.01
    return surge_charge

df['Surge_charge'] = df.apply(cal_surge_charge, axis=1)



# Calculating Total cost
df['New_cost']= df['base_cost'] + df['S/D_surge_charge'] + df['Surge_charge']
df.iloc[:,[0,1,9,10,11,12,13,16,17,18]].sample(10)

# Revenue Before and after
print("Revenue before applying Dynamic_pricing -->",round(sum(df['Historical_Cost_of_Ride']),2))
print("Revenue after applying Dynamic_pricing-->",round(sum(df['New_cost']),2))

diff=sum(df['New_cost'])-sum(df['Historical_Cost_of_Ride'])
print("Diffrenece of Revenue--> ", diff)
print("Revenue Percentage --> ", diff/sum(df['Historical_Cost_of_Ride'])*100)

# Conclusion
# Diffrenece of Revenue-->  179795.39
# Revenue Percentage -->  48.26





                                            # Data Spliting and Model training


#Importing libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error

x=df[['Demand_Supply_Ratio','Location_Category','Time_of_Booking','Vehicle_Type','Expected_Ride_Duration']] #Train column
y=df['New_cost'] #Target column

X_train_val, X_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

ohe = OneHotEncoder()
ohe.fit(x[['Location_Category','Time_of_Booking','Vehicle_Type']])
cat=ohe.categories_

column_trans = make_column_transformer((OneHotEncoder(categories=cat),
                                        ['Location_Category','Time_of_Booking','Vehicle_Type']),
                                        remainder='passthrough')    

rf=RandomForestRegressor()                                    
pipe=make_pipeline(column_trans,rf)
pipe.fit(X_train,y_train)

y_pred_rf=pipe.predict(X_test)
print(y_pred_rf)



                                            # Model Evaluation & Check prediction

# 1. Model evaluation on validation Data
y_val_pred_rf = pipe.predict(X_val)
mape = mean_absolute_percentage_error(y_val,y_val_pred_rf)
r2 = r2_score(y_val,y_val_pred_rf)
print("Error of RandomForest Regression Model = %.2f"%(mape*100),'%')
print("Accuracy of RandomForest Regression Model = %.2f"%((1 - mape)*100),'%')
print("R2 score of RandomForest Regression = %.2f"%(r2))


# 2. Model evaluation on Test Data
y_test_pred_rf = pipe.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred_rf)
test_r2 = r2_score(y_test, y_test_pred_rf)
print("Mean square error :", test_mse)
print("R2 score of RandomForestr Regression on test data:", test_r2)


#using Linearregression
lr=LinearRegression()                                    
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)

y_pred_lr=pipe.predict(X_test)
print(y_pred_lr)



                                            # Model Evaluation & Check prediction

# 1. Model evaluation on validation Data
y_val_pred_lr = pipe.predict(X_val)
mape = mean_absolute_percentage_error(y_val,y_val_pred_lr)
r2 = r2_score(y_val,y_val_pred_lr)
print("Error of Linear Regression Model = %.2f"%(mape*100),'%')
print("Accuracy of Linear Regression Model = %.2f"%((1 - mape)*100),'%')
print("R2 score of Linear Regression = %.2f"%(r2))


# 2. Model evaluation on Test Data
y_test_pred_lr = pipe.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred_lr)
test_r2 = r2_score(y_test, y_test_pred_lr)
print("Mean square error :", test_mse)
print("R2 score of Linear Regression on test data:", test_r2)

#pipe.predict(pd.DataFrame([['2.0','Urban','Night','Premium','90']],columns=['Demand_Supply_Ratio','Location_Category','Time_of_Booking','Vehicle_Type','Expected_Ride_Duration']))


                                            
                                            # Save Test data, Prediction and Error
results_df = pd.DataFrame(X_test, columns=x.columns)
results_df['Actual'] = y_test
results_df['Predicted_RandomForest'] = y_pred_rf
results_df['Predicted_LinearRegression'] = y_pred_lr
results_df['Error_RF'] = y_pred_rf-y_test
results_df['Error_LR'] = y_pred_lr-y_test
results_df.to_csv("test_results.csv", index=False)