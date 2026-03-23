import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta

def estimate_gas_price(date_str):
    """
    Estimates the natural gas price for a given date by combining polynomial
    regression for the long-term trend and a seasonal component.
    
    Args:
        date_str (str): A date string in 'YYYY-MM-DD' format.
        
    Returns:
        float: The estimated price of natural gas for the given date.
    """
    try:
        # Step 1: Load and Prepare Data
        # Assume Nat_Gas.csv is in the same directory.
        df = pd.read_csv("Nat_Gas.csv")

        # Clean column names by removing whitespace
        df.columns = df.columns.str.strip()

        # Rename columns for clarity
        df.rename(columns={'Date': 'date', 'Price': 'price'}, inplace=True)

        # Convert the 'date' column to datetime objects
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y')
        df.set_index('date', inplace=True)

        # Step 2: Feature Engineering for Modeling
        # Create a numerical representation of time (days since the first date)
        df['time'] = (df.index - df.index[0]).days

        # Extract month for seasonal analysis
        df['month'] = df.index.month

        # Use one-hot encoding for the month to capture seasonality
        df = pd.get_dummies(df, columns=['month'], prefix='month')

        # Step 3: Train the Model
        # Features: time (for trend), and one-hot encoded months (for seasonality)
        features = ['time'] + [col for col in df.columns if 'month_' in col]
        X = df[features]
        y = df['price']

        # Use a 3rd-degree polynomial for the 'time' feature to capture non-linear trends
        poly = PolynomialFeatures(degree=3, include_bias=False)
        X_poly = poly.fit_transform(X[['time']])

        # Combine polynomial features with seasonal dummies
        X_final = np.hstack((X_poly, X.drop('time', axis=1)))

        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(X_final, y)

        # Step 4: Visualize the Model Fit on Historical Data
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['price'], 'o', label='Actual Monthly Prices', markersize=4)
        ax.plot(df.index, model.predict(X_final), '-', label='Model Fit', color='orange')
        ax.set_title('Natural Gas Prices with Polynomial & Seasonal Fit')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_ylim(bottom=0)
        ax.grid(True)
        ax.legend()
        plt.show()

        # Step 5: Extrapolate and Visualize Future Prices
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=365, freq='D')
        
        future_df = pd.DataFrame(index=future_dates)
        future_df['time'] = (future_df.index - df.index[0]).days
        future_df['month'] = future_df.index.month
        
        future_df = pd.get_dummies(future_df, columns=['month'], prefix='month')
        
        month_cols = [col for col in features if 'month_' in col]
        for col in month_cols:
            if col not in future_df.columns:
                future_df[col] = 0
        
        future_df = future_df.reindex(columns=features, fill_value=0)

        X_future_poly = poly.transform(future_df[['time']])
        X_future_final = np.hstack((X_future_poly, future_df.drop('time', axis=1)))

        future_prices = np.maximum(model.predict(X_future_final), 0)
        future_df['predicted_price'] = future_prices

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['price'], 'o', label='Historical Prices', markersize=4)
        ax.plot(future_df.index, future_df['predicted_price'], '-', label='Extrapolated Prices')
        ax.set_title('Historical & Extrapolated Natural Gas Prices')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_ylim(bottom=0)
        ax.grid(True)
        ax.legend()
        plt.show()

        # Step 6: Get the specific price for the input date
        input_date = pd.to_datetime(date_str)
        
        # If the requested date is an exact date from the original file, return its price.
        if input_date in df.index:
            estimate = df.loc[input_date, 'price']
            print(f"Historical price for {input_date.strftime('%Y-%m-%d')}: ${estimate:.2f}")
            return estimate

        # For any other date (past, future, or in-between), use the model to predict.
        # This handles interpolation and extrapolation consistently.
        input_df = pd.DataFrame(index=[input_date])
        input_df['time'] = (input_df.index - df.index[0]).days
        input_df['month'] = input_df.index.month

        input_df = pd.get_dummies(input_df, columns=['month'], prefix='month')
        
        for col in month_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        # Ensure feature columns are in the same order as during training
        input_df = input_df.reindex(columns=features, fill_value=0)

        # Transform and predict
        X_input_poly = poly.transform(input_df[['time']])
        X_input_final = np.hstack((X_input_poly, input_df.drop('time', axis=1)))
        
        estimate = np.maximum(model.predict(X_input_final)[0], 0)
        print(f"Estimated price for {input_date.strftime('%y-%m-%d')}: ${estimate:.2f}")
        return estimate

    except FileNotFoundError:
        print("Error: The file 'Nat_Gas.csv' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example Usage:
# Estimate a price for a past date
estimate_gas_price('2021-05-15')
# Estimate a price for a future date (within the one-year extrapolation)
estimate_gas_price('2025-05-15')