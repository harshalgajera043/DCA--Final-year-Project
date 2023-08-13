model = LinearRegression()
model.fit(x1_train, y1_train)

yp_test = model.predict(x1_test)
print(yp_test)

plt.plot(df.index[split_at:], predict_1, color="red", label="Predicted")
plt.plot(df.index[split_at:], y1_test, color="green", label="Actual")
plt.legend()

# Label to axis
plt.xlabel("DATEPRD", color="blue")
plt.ylabel("Oil Production Rate", color="blue")

plt.title("Comparision between Predicted oil Prod rate and Actual Oil Prod rate", size=12, color="blue")
plt.show()

# water

x2_train = df[["AVG_DOWNHOLE_PRESSURE", "BORE_WAT_VOL"]][:split_at]
y2_train = df[["BORE_WAT_VOL"]][:split_at]
x2_test = df[["AVG_DOWNHOLE_PRESSURE", "BORE_WAT_VOL"]][split_at:]
y2_test = df[["BORE_WAT_VOL"]][split_at:]

# Linear Regression Used
model_2 = LinearRegression()
model_2.fit(x2_train, y2_train)

yp_test_2 = model_2.predict(x2_test)

plt.plot(df.index[2000:], yp_test_2, color="red", label="Predicted")  # use for linear regression
# plt.plot(df.index[split_at:], predict, color="red", label="Predicted")  # use for non-linear regression using random forest
plt.plot(df.index[split_at:], y2_test, color="green", label="Actual")
plt.legend()

plt.xlabel("DATEPRD", color="blue")
plt.ylabel("Water Production Rate", color="blue")

plt.title("Comparision between Predicted water Prod rate and Actual water Prod rate", size=12, color="blue")
plt.show()










# Random forest for regression
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
plt.xlabel("DATEPRD", color="blue")
plt.ylabel("Oil Production Rate", color="blue")

plt.title("Comparision between Predicted oil Prod rate and Actual Oil Prod rate", size=12, color="blue")
plt.show()

# water

x2_train = df[["AVG_DOWNHOLE_PRESSURE", "BORE_WAT_VOL"]][:split_at]
y2_train = df[["BORE_WAT_VOL"]][:split_at]
x2_test = df[["AVG_DOWNHOLE_PRESSURE", "BORE_WAT_VOL"]][split_at:]
y2_test = df[["BORE_WAT_VOL"]][split_at:]


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
