# diabetslogreg
Commenting diabetes data with logistic regression
import numpy as no  
import pandas as pd  
data=pd.read_csv("diabetes.csv")  
data.head()  
data.decribe()  
import matplotlib.pyplot as plt  
data["AgeRange"] = pd.cut(data.Age, bins=[0,41,82], labels=["0-41", "42-82"], include_lowest = True)  
data.head()  
healthy_ = data[data["Outcome"]== 1]["AgeRange"].value_counts()  
sick_ = data[data["Outcome"]== 0]["AgeRange"].value_counts()  
data_ = pd.DataFrame([healthy_, sick_])  
data_.index = ["healthy","sick"]  
data_.plot.bar(stacked=True,color=colors)  
fig = plt.figure(figsize=(10,5))  
plt.hist([data[data["Outcome"]==1]["Age"], data[data["Outcome"]==0]["Age"]], histtype='bar', stacked=True, bins=41, color=["r","b"], width=1, label=["Healthy","Sick"])  
plt.xlabel("Yaş")  
plt.ylabel("N")  
plt.legend()  
data.drop(["AgeRange"], axis=1, inplace=True)  
x = data.drop("Outcome", axis=1)  
y = data["Outcome"]  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)  
from sklearn.preprocessing import StandardScaler  
sc = StandardScaler()  
sc.fit(x_train)  
x_train_std = sc.transform(x_train)  
x_test_std = sc.transform(x_test)  
from sklearn.linear_model import LogisticRegression  
lr = LogisticRegression()  
lr.fit(x_train_std, y_train)  
prediction = lr.predict(x_test_std)  
from sklearn.metrics import classification_report  
print(classification_report(y_test, prediction))  
