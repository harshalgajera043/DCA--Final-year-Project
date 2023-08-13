import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import torch

dca_df = pd.read_csv("monthly-production-data.csv", index_col=0, parse_dates=True)
# print(dca_df.head())

df_file = dca_df.copy()
df = df_file[:round(len(df_file)*0.7)]
print(df)
# df.plot(figsize=(12, 4))
# plt.show()

print(df.index)  # will give us the all the date for which data has been recorded in datetime formate

# Time index vs Continuous Days array - what a DCA model needs?


def day_maker(dataframe):
    """ Pass a time series Dataframe to it, and it will return a days' column. Subtract dates and makes days.
    Return is a days (np array)."""
    days = []
    for i in range(len(df)):
        delta = dataframe.index[i] - dataframe.index[0]
        days.append(delta.days)
    days = np.array(days)
    print(days)
    return days

try:
    df["days"] = day_maker(df)
except UserWarning:
    pass
# print(df)
#
# plt.plot(df["days"], df["Oil_(BBL/M)"])
# plt.show()


# exponential model
def expo(t, q, b):
    q0 = q[0]
    q_exp = q0*np.exp(-b*t)
    return q_exp

df["predicted_rate"] = expo(df["days"], df["Oil_(BBL/M)"], b=0.0011)  # changing the value of B manually to get the perfact fit for our prediction
# print(df)

plt.plot(df["days"], df["predicted_rate"], color="red")
plt.title("DCA using exponential decline")
plt.xlabel("Days")
plt.ylabel("Oil Production Rate in BBL/M")
plt.scatter(df["days"], df["Oil_(BBL/M)"])
plt.show()


# Defining hyperbolic # Class and OOP application

# hyperbolic model
def q_hyp(t, qi, b, d):
    qfit = qi*(np.abs((1+b*d*t))**(1/b))
    return qfit


q = df["Oil_(BBL/M)"]
t = df["days"]

# First we have to normalize so that it converges well and quick.
q_n = q / max(q)
t_n = t / max(t)

# curve-fit (Optimization of parameters)
params = curve_fit(q_hyp, t_n, q_n)
print(params)
[qi, b, d] = params[0]

# These are for normalized t and q.
# We must re-adjust for q  and t (non-normalized)
d_f = d / max(t)
qi_f = qi * max(q)

# Now we can use these parameters.
qfit = q_hyp(t, qi_f, b, d_f)
print(qfit)
params_fit = qfit[1]
print(params_fit)
#
# qfit = hyp_fitter(q=df["Oil_(BBL/M)"], t=df["days"])
# params_fit = hyp_fitter(q=df["Oil_(BBL/M)"], t=df["days"])[1]

plt.plot(df["days"], qfit, color="red")
plt.scatter(df["days"], df["Oil_(BBL/M)"])
plt.title("DCA using hyperbolic decline")
plt.xlabel("Days")
plt.ylabel("Oil Production Rate in BBL/M")
plt.show()


# Forcasting
# function for hyperbolic cumulative production
def cumpro(q_forecast, qi, di, b):
    # return q_forecast
    return ((qi ** b) * (((qi ** (1 - b)) - (q_forecast ** (1 - b)))/ ((1 - b) * di)))


# forecast gas rate until 4000 days
t_forecast = np.arange(4000)
q_forecast = q_hyp(t_forecast, qi_f, b, d_f)
print(f"forcast time in days: {t_forecast}")
print(f"qi_f: {qi_f}")
print(f"d_f:{d_f}")
print(f"b : {b}")
# q_forecast = q_hyp(t_forecast, qi_f, d_f, b)
print(f"forcasted production rate: {q_forecast}")

# forecast cumulative production until 4000 days
Qp_forecast = cumpro(q_forecast, qi, d, b)
print(f"forcasted cumulative production rate:{Qp_forecast}")

# plot the production data with the forecasts (rate and cum. production)
plt.figure(figsize=(10, 5))


plt.subplot(1, 2, 1)
plt.plot(t, q, '.', color='red', label='Production Data')
plt.plot(t_forecast, q_forecast, label='Forecast')
plt.title('Oil Production Rate Result of DCA', size=13, pad=15)
plt.xlabel('Days')
plt.ylabel('Rate (BBL/M)')
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(t_forecast, Qp_forecast)
plt.title('Oil Cumulative Production Result of DCA', size=13, pad=15)
plt.xlabel('Days')
plt.ylabel('Production (BBL)')
plt.xlim(xmin=0)
plt.ylim(ymin=0)


plt.show()
