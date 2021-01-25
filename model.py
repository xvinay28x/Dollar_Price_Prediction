import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model
# load data
df = pd.read_excel("Excelrates.xlsx")
# define x and y
x = df[["Date","Month","Year"]]
y = df["INR"]
# split the data into 2 parts traning and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 0)
# train the model using traning data
reg = linear_model.LinearRegression()
reg.fit(x_train,y_train)
# test the model using testing data
reg.predict(x_test)
# predict 
x=reg.predict([[1,1,21]])
print(x)
pickle.dump(reg,open("model.pkl","wb"))
