import pandas as pd
import matplotlib.pyplot as plt

#Create a Python class that enables data loading from CSV/XLSX files and provides functionalities
class DataAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        
#a. Load the CSV/XLSX file.
    def load_data(self):
        if self.file_path.endswith('.csv'):
            self.data = pd.read_csv(self.file_path)
        elif self.file_path.endswith('.xlsx'):
            self.data = pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format")
            
#b. Print summaries of all numeric variables in the dataset.
    def summarize_numeric_variables(self):
        if self.data is None:
            raise ValueError("Data not loaded yet")
        
        numeric_columns = self.data.select_dtypes(include=['number']).columns
        numeric_summary = self.data[numeric_columns].describe()
        print(numeric_summary)
    
#c. Generate a bar graph for a specified categorical variable (variable name provided as input).
    def generate_bar_graph(self, categorical_variable):
        if self.data is None:
            raise ValueError("Data not loaded yet")
        
        if categorical_variable not in self.data.columns:
            raise ValueError("Categorical variable not found in dataset")
        
        category_counts = self.data[categorical_variable].value_counts()
        category_counts.plot(kind='bar')
        plt.title(f'Bar Graph for {categorical_variable}')
        plt.xlabel(categorical_variable)
        plt.ylabel('Count')
        plt.show()
        
#d. Plot scatter plots for two specified numeric variables (input provided)
    def plot_scatter(self, x_variable, y_variable):
        if self.data is None:
            raise ValueError("Data not loaded yet")
        
        if x_variable not in self.data.columns or y_variable not in self.data.columns:
            raise ValueError("One or both of the specified variables not found in dataset")
        
        plt.scatter(self.data[x_variable], self.data[y_variable])
        plt.title(f'Scatter Plot: {x_variable} vs {y_variable}')
        plt.xlabel(x_variable)
        plt.ylabel(y_variable)
        plt.show()


# Example
data_analyzer = DataAnalyzer('BostonHousing.csv')
data_analyzer.load_data()

data_analyzer.summarize_numeric_variables()

data_analyzer.generate_bar_graph('chas')

data_analyzer.plot_scatter('rm', 'age')