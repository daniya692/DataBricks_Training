#STEP1: Importing datasets from the Seaborn
import pandas as pd 
import seaborn as sns

df1 = sns.load_dataset("iris")
df1

#STEP2: Handle missing data
df1.isnull().sum()

df2 = sns.load_dataset("tips")
df2

#STEP2: Handle missing data
df2.isnull().sum()

df1.dropna(inplace= True)
df2.dropna(inplace= True)

df1.drop_duplicates()

df2.drop_duplicates(inplace= True)
df2

#STEP3: Create new Column, Calculate the tip percentage 
df2["tip_percentage"] = (df2["tip"] / df2["total_bill"]) * 100


#STEP4: Create new Columns, Calculate the Average Values
#Group the data by the "species" column
grouped_data = df1.groupby("species")

#Calculate the average for each numeric feature within each group
averages = grouped_data.mean()

#Add the new columns containing the averages to the original dataset
df1 = df1.merge(averages, on="species", suffixes=("", "_average"))

# Display the updated dataset with the average columns
print(df1)

#STEP5: Load the dataset into its new SQLite database
import sqlite3

#db_iris = "df1.db"  
connection = sqlite3.connect("df1.db")

df1.to_sql("df1", connection, if_exists="replace", index=False)
df2.to_sql("tips", connection, if_exists="replace", index=False)

connection.close()
