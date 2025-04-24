# Implementation of Logistic Regression Model to Predict the Placement Status of Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load dataset from CSV.Drop irrelevant columns (e.g., sl_no, salary).Convert categorical columns to numerical using label encoding via .astype('category').cat.codes.
2. Set independent variables X (all columns except status).Set target variable Y as status (placed or not).Split data into training and testing sets using train_test_split.
3. Initialize Logistic Regression model with high max_iter.Fit the model on training data using fit().
4. Predict on test data using predict().Evaluate performance using:accuracy_score,classification_report andconfusion_matrixMake predictions on new/unseen samples.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Prajin S
RegisterNumber:  212223230151
*/
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df=pd.read_csv('Placement_Data.csv')
df
df.info()
df=df.drop('sl_no',axis=1)
df['gender']=df['gender'].astype('category')
df['ssc_b']=df['ssc_b'].astype('category')
df['hsc_b']=df['hsc_b'].astype('category')
df['hsc_s']=df['hsc_s'].astype('category')
df['degree_t']=df['degree_t'].astype('category')
df['workex']=df['workex'].astype('category')
df['specialisation']=df['specialisation'].astype('category')
df['status']=df['status'].astype('category')
df.dtypes
df['gender']=df['gender'].cat.codes
df['ssc_b']=df['ssc_b'].cat.codes
df['hsc_b']=df['hsc_b'].cat.codes
df['hsc_s']=df['hsc_s'].cat.codes
df['degree_t']=df['degree_t'].cat.codes
df['workex']=df['workex'].cat.codes
df['specialisation']=df['specialisation'].cat.codes
df['status']=df['status'].cat.codes
df
df=df.drop('salary',axis=1)
df
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
Y
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=3)
clf=LogisticRegression(max_iter=10000)
clf.fit(xtrain,ytrain)
clf.score(xtest,ytest)
ypred=clf.predict(xtest)
ypred
acc=accuracy_score(ypred,ytest)
acc
clf.predict([[0,87,0,95,0,2,78,2,0,0,1,0]])
clf.predict([[0,8,0,95,0,2,8,2,0,0,1,0]])
cr=classification_report(ytest,ypred)
print("Classification Report:\n",cr)
confusion=confusion_matrix(ytest,ypred)
print("Confusion matrix:\n",confusion)
```

## Output:

![image](https://github.com/user-attachments/assets/a0295a23-8e53-4e93-9f1c-d5bdd961a035)

![image](https://github.com/user-attachments/assets/bf09c95d-d0d3-41c3-86c9-6c9b8ddc0158)

![image](https://github.com/user-attachments/assets/779091b8-c223-41b2-a05a-64e429f37157)

![image](https://github.com/user-attachments/assets/33fbfd70-14a5-410f-96c2-ddaa313bc73f)

![image](https://github.com/user-attachments/assets/05657eec-9d4f-4364-aa22-01eaa58afd48)

![image](https://github.com/user-attachments/assets/986a69c8-1c9a-4bea-bfb4-935c9f295bb4)

![image](https://github.com/user-attachments/assets/af7f2ba1-126d-4d85-a00d-dcdf0f95fe7c)

![image](https://github.com/user-attachments/assets/6caa91c6-109b-47cb-962d-d7ddd01015ef)

![image](https://github.com/user-attachments/assets/909113d8-499c-4bb2-b9bd-ea66a9f90eb4)

![image](https://github.com/user-attachments/assets/66069686-83e2-4214-8c05-b5bb56772a2a)

![image](https://github.com/user-attachments/assets/d454b9fa-35c5-4d3f-946e-7a3db4c1b259)

![image](https://github.com/user-attachments/assets/8fe36056-e48c-4f6d-b16e-1d81263225be)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
