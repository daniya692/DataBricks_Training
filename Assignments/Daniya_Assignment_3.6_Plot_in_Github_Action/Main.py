import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
df = pd.read_csv(r"c:\Users\Hp\Downloads\50_Startups.csv")
df.head()
df.shape

#EDA
df.isnull().sum()
df.duplicated().sum()
df.info()

#Visualization
sns.pairplot(data=df)
plt.show()


ax =df.groupby(['State'])['Profit'].mean().plot.bar(figsize = (8,5),fontsize = 8, color = ['lightgreen', 'lightblue', 'Orange'])

ax.set_title("Average Porfit for different States where the startups operate", fontsize = 8)
plt.show()

df.State.value_counts()

#Data Preprocessing
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['State']= encoder.fit_transform(df['State'])

x= df.iloc[:, :-1].values  
y= df.iloc[:, 4].values

#Splitting the dataset
from sklearn.model_selection import train_test_split
# Test size of 0.2
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.2, random_state=0)

# Test size of 0.3 
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.3, random_state=0)  

# Test size of 0.4
x_train3, x_test3, y_train3, y_test3 = train_test_split(x, y, test_size=0.4, random_state=0)

#Model fitting
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression() 

# Train and score for test size 0.2
regressor.fit(x_train1, y_train1)  
print("Train and score for test size 0.2 ---->",regressor.score(x_test1, y_test1))

# Train and score for test size 0.3
regressor.fit(x_train2, y_train2)
print("Train and score for test size 0.3 ---->",regressor.score(x_test2, y_test2))

# Train and score for test size 0.4
regressor.fit(x_train3, y_train3)
print("Train and score for test size 0.4 ---->",regressor.score(x_test3, y_test3))

#Predictions
y_pred1 = regressor.predict(x_test1)
y_pred2 = regressor.predict(x_test2)
y_pred3 = regressor.predict(x_test3)

#Visualization
# Scatter plots for test size 0.2
plt.scatter(y_test1, y_pred1)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Test Size 0.2')


# Scatter plots for test size 0.3
plt.scatter(y_test2, y_pred2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Test Size 0.3')


# Scatter plots for test size 0.4
plt.scatter(y_test3, y_pred3)
plt.xlabel('Actual Values')
plt.ylabel ('Predicted Values')
plt.title('Test Size 0.4')


# Scatter plots
plt.scatter(y_test1, y_pred1, label='0.2 split')
plt.scatter(y_test2, y_pred2, label='0.3 split')
plt.scatter(y_test3, y_pred3, label='0.4 split')

plt.legend()
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True vs Predicted Values")
plt.show()


# Make predictions on some values from the test set
sample_predictions = regressor.predict(x_test3[:5])  
print("Sample Predictions:", sample_predictions)


input_data = [[165349.20, 136897.80, 471784.10,	2]]  	
prediction = regressor.predict(input_data)

print("Prediction:", prediction)


