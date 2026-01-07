

import pandas as pd
import numpy as np
df=pd.read_csv(r"/content/breast-cancer.csv")
df.info()

df['diagnosis'].unique()

df.describe()

df.shape

df["diagnosis"].shape

df.isnull().sum()



n=df.groupby(["diagnosis"])["area_mean"].sum()
n

import matplotlib.pyplot as plt
n.plot(kind="bar")
plt.show()

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
x=df.drop("diagnosis", axis=1)
y=df["diagnosis"]
model=SVC()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.18,random_state=36)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

