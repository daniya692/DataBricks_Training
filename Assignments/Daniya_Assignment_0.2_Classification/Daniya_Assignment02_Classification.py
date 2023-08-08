#Importing required Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

# 1: Load and explore the dataset
df = pd.read_csv("Iris.csv")
df.head()
df.shape

# Basic exploration of the dataset
df.describe()
df.info()
df.drop('Id', axis=1, inplace=True)
df.shape

#2:  preprocessing is needed for the Iris dataset
df.isnull().sum()
df[df.duplicated()]
df.drop_duplicates(inplace=True)
df.duplicated().sum()
df['Species'].unique()

sns.pairplot(df, hue='Species')
plt.show()

#split the dataset into Independent & Dependent Features
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#Checking the dataset
X
y

#3: Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)


#Label the features 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y)
df['Species'] = le.transform(y)
df


from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#4: Choosing a classification algorithm
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

#5: Train the chosen model on the training data
#Logistic regression
model = LogisticRegression()
model.fit(X_train_scaled,y_train)


#6: Evaluate the model's performance
model_pred = model.predict(X_test_scaled)

accuracy_lr = accuracy_score(y_test, model_pred)
print("Accuracy_lr:", accuracy_lr)


accuracy_lr = accuracy_score(y_test, model_pred)
precision_lr = precision_score(y_test, model_pred, average='weighted')
recall_lr = recall_score(y_test, model_pred, average='weighted')
f1_lr = f1_score(y_test, model_pred, average='weighted')

print("accuracy_lr:", accuracy_lr)
print("precision_lr:", precision_lr)
print("recall_lr:", recall_lr)
print("f1_lr:", f1_lr)

#KNN
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train_scaled,y_train)

clf_pred = clf.predict(X_test_scaled)
accuracy_knn=accuracy_score(y_test,clf_pred)
accuracy_knn

#SVM
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train_scaled,y_train)

svc_pred = svc.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test,svc_pred)
accuracy_svm

accuracy_svm = accuracy_score(y_test, svc_pred)
print("Accuracy_svm:", accuracy_svm)

conf_matrix_svm = confusion_matrix(y_test, svc_pred)
print("Confusion Matrix_svm:")
print(conf_matrix_svm)

report_svm = classification_report(y_test, svc_pred)
print("Classification Report:")
print(report_svm)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
model= DecisionTreeClassifier()
model.fit(X_train_scaled,y_train)

Dec_pred = model.predict(X_test_scaled)
accuracy_dt = accuracy_score(y_test,Dec_pred)
accuracy_dt


#7: Fine-tune the model is not necessary for the Iris dataset 
# because it is already trained with the correct accuracy score and the correct accuracy score  for the dataset itself.

#Model Performance Analysis/comparision
models = ['KNN', 'Logistic Regression', 'Decision Trees', 'SVM']
accuracy_scores = [accuracy_knn, accuracy_lr, accuracy_dt, accuracy_svm]

plt.figure(figsize=(8, 6))
sns.barplot(x=models, y=accuracy_scores)
plt.ylim(0.9, 1.0)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')

# Add labels displaying the accuracy values on top of the bars
for i, accuracy in enumerate(accuracy_scores):
    plt.text(i, accuracy, f'{accuracy:.3f}', ha='center', va='bottom')

plt.show()