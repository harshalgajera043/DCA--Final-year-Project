import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tkinter import *
import requests
from bs4 import BeautifulSoup

BG_COLOR = "#9BA4B5"
FONT = ("Arial", 14, "bold")
FONT_1 = ("Arial", 10, "bold")


#----------------------DECLINE CURVE ANALYSIS----------------------#
# # csv_path = "/Users/hgaje/PycharmProjects/DCA_prectice/monthly-production-data.csv"
# # csv_file = "input_data_quiz.csv" #File to be used directly
# csv_file = Generate_DCA()
# df = pd.read_csv(csv_file)
# # print(df.head())


def split_data_set(value):
    if value == 1:
        return 0.3
    else:
        return 0.4

def generate_dca_linear_regression(df):
    # Your existing code for decline curve analysis with Linear Regression
    import warnings
    warnings.filterwarnings("ignore")
    
    df = df.fillna(0)
    print(df.describe().T)
    print(df.columns)
    
    X = df[['LIQ_VOL']]
    y = df['AVG_DOWNHOLE_PRESSURE']
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Return the trained Linear Regression model and test data for evaluation
    return model, X_test, y_test

def generate_dca_random_forest(df):
    # Your existing code for decline curve analysis with Random Forest
    import warnings
    warnings.filterwarnings("ignore")
    
    df = df.fillna(0)
    print(df.describe().T)
    print(df.columns)

    X = df[['LIQ_VOL', 'BORE_GAS_VOL']]
    y = df['BORE_OIL_VOL']
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Return the trained Random Forest model and test data for evaluation
    return model, X_test, y_test

def forecast_future_production(df, linear_model, rf_model):
    time_series_data = df[['DATE', 'BORE_OIL_VOL']]
    
    time_series_data['DATE'] = pd.to_datetime(time_series_data['DATE'])
    time_series_data.set_index('DATE', inplace=True)

    # Generate forecast using Linear Regression model
    linear_forecast = linear_model.predict(time_series_data[['LIQ_VOL']])

    # Generate forecast using Random Forest model
    rf_forecast = rf_model.predict(time_series_data[['LIQ_VOL', 'BORE_GAS_VOL']])
    
    combined_forecast = (linear_forecast + rf_forecast) / 2 

    print("Combined Forecasted Production:")
    print(combined_forecast)
    plt.plot(time_series_data.index, time_series_data['BORE_OIL_VOL'], label='Historical Production')
    plt.plot(time_series_data.index, combined_forecast, label='Combined Forecasted Production', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Oil Production')
    plt.title('Forecasted Oil Production')
    plt.legend()
    plt.show()

def web_scrape_crude_price():
    url = "Add price website here"  # Replace this with the actual URL for scraping crude oil prices
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    price_element = soup.find("div", class_="crude-price")
    if price_element:
        crude_price = price_element.text.strip()
        return crude_price
    else:
        return "Price Not Available"

def estimate_revenue(forecasted_production, current_crude_price):
    # Calculate revenue generated based on forecasted production and current crude oil price
    revenue = forecasted_production * current_crude_price
    if revenue>=0:
        return revenue
    else:
        print('Revenue forecast unable')
    
#----------------------------GUI SET-UP----------------------------#
window = Tk()
# window.minsize(width=500, height=450)
window.title("Decline Curve Analysis")
window.config(bg=BG_COLOR)

# creating labels of
title_label = Label(text="File Path:", bg=BG_COLOR,font=FONT)
title_label.grid(column=0, row=2, padx=(200, 0), pady=(20, 20))

Label_DCA = Label(text="DCA", bg=BG_COLOR, font=("Arial", 100, "bold")).grid(column=0, row=0, columnspan=2)
Label_DCA_full = Label(text="Decline Curve Analysis", bg=BG_COLOR, font=("Arial", 14, "bold")).grid(column=0, row=1, columnspan=2)

# text_box to get user inputs
file_path = Entry(width=56)
file_path.grid(column=1, row=2, padx=(20, 200), pady=(20, 20))
file_path.focus()

Label(window, text="Select Percentage Split of the dataset", bg=BG_COLOR,font=FONT_1).grid(column=0, row=3, columnspan=2, pady=(0, 5))
var = IntVar()
Radiobutton(window, text="80-20 Split", variable=var, value=1, bg=BG_COLOR,font=FONT_1).grid(column=0, row=4, columnspan=2)
Radiobutton(window, text="75-25 Split", variable=var, value=2, bg=BG_COLOR,font=FONT_1).grid(column=0, row=5, columnspan=2)
Radiobutton(window, text="70-30 Split", variable=var, value=3, bg=BG_COLOR,font=FONT_1).grid(column=0, row=6, columnspan=2)
Radiobutton(window, text="65-35 Split", variable=var, value=4, bg=BG_COLOR,font=FONT_1).grid(column=0, row=7, columnspan=2)
Radiobutton(window, text="60-40 Split", variable=var, value=5, bg=BG_COLOR,font=FONT_1).grid(column=0, row=8, columnspan=2)
Radiobutton(window, text="55-45 Split", variable=var, value=6, bg=BG_COLOR,font=FONT_1).grid(column=0, row=9, columnspan=2)

def Generate_DCA():
    global file_path
    # csv_path = "/Users/hgaje/PycharmProjects/DCA_prectice/monthly-production-data.csv"
    # csv_file = "input_data_quiz.csv" #File to be used directly
    csv_file = file_path.get()
    df = pd.read_csv(csv_file)

    # Generate DCA using linear regression
    linear_model, linear_X_test, linear_y_test = generate_dca_linear_regression(df)

    # Generate DCA using random forest
    rf_model, rf_X_test, rf_y_test = generate_dca_random_forest(df)
    
    forecast_future_production(df, linear_model, rf_model)    # Forecast future production
    
    current_crude_price = web_scrape_crude_price()
    combined_forecast = (df['Linear_Forecast'] + df['RF_Forecast']) / 2

    # Add the combined forecast to the dataframe
    df['Combined_Forecast'] = combined_forecast

    # Estimate revenue based on the forecasted production and current crude oil price
    forecasted_production = combined_forecast.iloc[-1]  # Get the last forecasted production value
    revenue = estimate_revenue(forecasted_production, current_crude_price)
    print(f"The estimated revenue generated based on the forecasted production and current crude oil price is ${revenue}.")


# Label().grid(column=0, row=4)


# let's create a generate password button
Button(text="Generate DCA", width=15, font=("Arial", 7, "bold"), command=Generate_DCA).grid(column=0, row=10, columnspan=2, pady=(20, 20))

window.mainloop()
