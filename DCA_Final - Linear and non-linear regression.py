import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tkinter import *

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

def generate_dca(df):
    import warnings
    warnings.filterwarnings("ignore")

    # let's fill empty values with zeros
    df = df.fillna(0)
    print(df.describe().T)
    print(df.columns)

    # Ploted Average downhole Pressure and WI rate in one plot with subplot and twinx()
    plt.rcParams["figure.autolayout"] = True

    ax1 = plt.subplot()
    l1, = ax1.plot(df.index, df["AVG_DOWNHOLE_PRESSURE"], color="green")

    ax2 = ax1.twinx()
    l2, = ax2.plot(df.index, df["BORE_GAS_VOL"], alpha=0.3, color="blue")

    plt.legend([l1, l2], ["AVG_DOWNHOLE_PRESSURE", "BORE_GAS_VOL"])

    plt.show()

    df["AVG_DOWNHOLE_PRESSURE"].replace(0, np.random.randint(240, 260), inplace=True)
    df["BORE_GAS_VOL"].replace(0, np.random.randint(4500, 6000), inplace=True)
    df["BORE_WAT_VOL"].replace(0, df["BORE_WAT_VOL"].median(), inplace=True)
    df["BORE_OIL_VOL"].replace(0, df["BORE_OIL_VOL"].median(), inplace=True)
    # df.replace(0, df.median(), inplace=True)

    # Create ML model to predict GLR

    df["LIQ_VOL"] = df["BORE_OIL_VOL"] + df["BORE_WAT_VOL"]

    sns.scatterplot(x=df["LIQ_VOL"], y=df["AVG_DOWNHOLE_PRESSURE"])
    plt.show()

    length = len(df)
    print(length)

    split = var.get()
    print(split)

    split_at = round(length * (1-split_data_set(split)))
    print(split_at)

    x_train = df[["LIQ_VOL"]][:split_at]
    y_train = df[["AVG_DOWNHOLE_PRESSURE"]][:split_at]
    x_test = df[["LIQ_VOL"]][split_at:]
    y_test = df[["AVG_DOWNHOLE_PRESSURE"]][split_at:]


    plt.scatter(x_train, y_train)
    plt.scatter(x_test, y_test)
    plt.show()

    # Predicting the oil and water production rate using ML algorithms

    fig, ax = plt.subplots(2, 1, sharex="col", sharey="row")

    ax[0].plot(df.index, df["BORE_OIL_VOL"], color="orange")
    ax[1].plot(df.index, df["BORE_WAT_VOL"], color="blue")
    plt.show()


    # split data into training and testing sets
    x1_train = df[["AVG_DOWNHOLE_PRESSURE", "BORE_GAS_VOL"]][:split_at]
    y1_train = df[["BORE_OIL_VOL"]][:split_at]
    x1_test = df[["AVG_DOWNHOLE_PRESSURE", "BORE_GAS_VOL"]][split_at:]
    y1_test = df[["BORE_OIL_VOL"]][split_at:]

    # model = LinearRegression()
    # model.fit(x1_train, y1_train)
    #
    # yp_test = model.predict(x1_test)
    # print(yp_test)

    #Random forest for regression
    clf = RandomForestRegressor()

    # train the model
    clf.fit(x1_train, y1_train)

    # predict on test data
    predict_1 = clf.predict(x1_test)
    print(f"Prediction is {predict_1}")

    plt.plot(df.index[split_at:], predict_1, color="red", label="Predicted")
    plt.plot(df.index[split_at:], y1_test, color="green", label="Actual")
    plt.legend()

    # Label to axis
    plt.xlabel("DATEPRD", color= "blue")
    plt.ylabel("Oil Production Rate", color="blue")

    plt.title("Comparision between Predicted oil Prod rate and Actual Oil Prod rate", size=12, color="blue")
    plt.show()

    # water

    x2_train = df[["AVG_DOWNHOLE_PRESSURE", "BORE_WAT_VOL"]][:split_at]
    y2_train = df[["BORE_WAT_VOL"]][:split_at]
    x2_test = df[["AVG_DOWNHOLE_PRESSURE", "BORE_WAT_VOL"]][split_at:]
    y2_test = df[["BORE_WAT_VOL"]][split_at:]

    # Linear Regression Used
    # model_2 = LinearRegression()
    # model_2.fit(x2_train, y2_train)
    #
    # yp_test_2 = model_2.predict(x2_test)

    # RandomForest Used
    clf = RandomForestRegressor()

    # train the model
    clf.fit(x2_train, y2_train)

    # predict on test data
    predict = clf.predict(x2_test)
    print(f"Prediction is {predict}")


    # plt.plot(df.index[2000:], yp_test_2, color="red", label="Predicted")  # use for linear regression
    plt.plot(df.index[split_at:], predict, color="red", label="Predicted")  # use for non-linear regression using random forest
    plt.plot(df.index[split_at:], y2_test, color="green", label="Actual")
    plt.legend()

    plt.xlabel("DATEPRD", color="blue")
    plt.ylabel("Water Production Rate", color="blue")

    plt.title("Comparision between Predicted water Prod rate and Actual water Prod rate", size=12, color="blue")
    plt.show()

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


    generate_dca(df)



# Label().grid(column=0, row=4)


# let's create a generate password button
Button(text="Generate DCA", width=15, font=("Arial", 7, "bold"), command=Generate_DCA).grid(column=0, row=10, columnspan=2, pady=(20, 20))

window.mainloop()