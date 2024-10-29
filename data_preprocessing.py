# data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Selecting numerical features only
    X = df.select_dtypes(include=['float64', 'int64']).drop(columns=['Historical_Cost_of_Ride'])
    y = df['Historical_Cost_of_Ride']

    # Split data into train, validation, and test sets (70% train, 15% validation, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test
