
#Insurance cost fot the given features------------


import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import shuffle
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier  
from sklearn import linear_model,preprocessing
import matplotlib.pyplot as pyplot
from matplotlib import style
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r"C:\Users\tahas\OneDrive\Desktop\Pro\insurance.csv",sep=",") 
#print(data.head())

data = data[["age","sex","bmi","children","smoker","charges","region"]]
print(data.head())


# by this we are scaling and replace the colums after sccaling them to dataframe
le=preprocessing.LabelEncoder()
data.age = le.fit_transform(data["age"])
data.region = le.fit_transform(data["region"])
data.bmi = le.fit_transform(data["bmi"])
data.children = le.fit_transform(data["children"])
data.charges = le.fit_transform(data["charges"])
data.sex = le.fit_transform(data["sex"])
data.smoker = le.fit_transform(data["smoker"])

# Here we are dividing our data in features and Labels

X = data.drop(["charges"], axis =1)
y= data["charges"]
print(X.head())


x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.1)


model =GradientBoostingRegressor()

model.fit(x_train,y_train)

accuracy = model.score(x_test,y_test)

print(accuracy)

predictt = model.predict(x_test)
print(predictt.shape)
targ = y
style.use("ggplot")
pyplot.scatter(x_test,y_test)
pyplot.scatter(x_test,predictt)
pyplot.xlabel("Predict")
pyplot.ylabel("target(real)")
pyplot.show()

