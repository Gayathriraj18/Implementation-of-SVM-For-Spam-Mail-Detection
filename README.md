# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages.
2. Read the given csv file and display the few contents of the data.
3. Assign the features for x and y respectively.
4. Split the x and y sets into train and test sets.
5. Convert the Alphabetical data to numeric using CountVectorizer.
6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
7. Find the accuracy of the model.
## Program:
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: 
RegisterNumber:  
*/
```
import pandas as pd
df=pd.read_csv("/content/spam.csv",encoding='Windows-1252')
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
df.head()
df.isnull().sum()
x=df["v1"].values
y=df["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

![91](https://user-images.githubusercontent.com/94154854/204571463-1e453570-6872-4e28-ba6c-3d2ab8144ed1.png)

![92](https://user-images.githubusercontent.com/94154854/204571519-93e60bd7-95b5-4e2e-b79a-860482856d71.png)

![93](https://user-images.githubusercontent.com/94154854/204571551-07db2999-0654-49d5-b036-ad154cba81d1.png)

![94](https://user-images.githubusercontent.com/94154854/204571574-1ff37ed5-0216-46d5-957b-52f23dcbebc0.png)

![95](https://user-images.githubusercontent.com/94154854/204571597-1c636be8-c6d3-4957-b290-91f1104f8b47.png)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
