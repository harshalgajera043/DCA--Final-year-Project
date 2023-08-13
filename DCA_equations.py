import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exponential_decline(t, q_i, D):
    return q_i * np.exp(-D * t)

data_file = pd.read_csv("Production_rate_data.csv")
production_rate = []
for i in range(0, len(data_file["Production Rate"])):
    production_rate.append(data_file["Production Rate"][i])
print(production_rate)
print(len(production_rate))

print(type(data_file["Date"][0]))
production_time = []
day = 0
for i in range(0, len(data_file["Date"])):
    d1 = dt.datetime.strptime(data_file["Date"][0], "%d-%m-%Y")
    d2 = dt.datetime.strptime(data_file["Date"][1], "%d-%m-%Y")
    delta_d = (d2-d1).days
    day+=delta_d
    production_time.append(day)
print(production_time)
print(len(production_time))

t = np.array(production_time)
q = np.array(production_rate)

popt, pcov = curve_fit(exponential_decline, t, q)

plt.scatter(t, q, label='Production Data')
plt.plot(t, exponential_decline(t, *popt), 'r-', label='Exponential Fit')
plt.xlabel('Time (days)')
plt.ylabel('Production Rate (bpd)')
plt.legend()
plt.show()
