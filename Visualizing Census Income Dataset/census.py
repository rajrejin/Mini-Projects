import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('D:/FAU/4. WS 23/DSS/Exercises/My-Projects/Visualizing Census Income Dataset/census_income_dataset.csv')

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Drop rows with NaN
df.dropna(inplace=True)

# Define the bins
min_age = df['AGE'].min()  # Get the minimum age from the dataset
max_age = df['AGE'].max()
bins = [min_age, 20] + list(range(30, max_age+10, 10))

# Use pd.cut function can attribute the values into its specific bins
df['AGE_BIN'] = pd.cut(df['AGE'], bins=bins)

# Calculate the frequency of each age bin
age_counts = df['AGE_BIN'].value_counts()

# Create a DataFrame from the age counts
df_age_counts = pd.DataFrame({'AGE_BIN': age_counts.index, 'COUNT': age_counts.values})

# Sort the DataFrame by age bin
df_age_counts.sort_values('AGE_BIN', inplace=True)

# Plot 1: Bubble chart for Age distribution of respondents
plt.figure(figsize=(10,6))
scatter = plt.scatter(df_age_counts['AGE_BIN'].astype(str), df_age_counts['COUNT'], s=df_age_counts['COUNT'], alpha=0.5)
plt.title('Age Distribution of Respondents')
plt.xlabel('Age Group')
plt.ylabel('Number of Respondents')
for i in range(df_age_counts.shape[0]):
    plt.text(df_age_counts['AGE_BIN'].astype(str)[i], df_age_counts['COUNT'][i], df_age_counts['COUNT'][i], ha='center', va='center')
plt.ylim(0, df_age_counts['COUNT'].max() + 500)

# Plot 2: Pie chart for relationship status
plt.figure(figsize=(10,6))
df['RELATIONSHIP'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Relationship Status Distribution')
plt.ylabel('')  # Remove the y-axis label

# Plot 3: Histogram for salary distribution within each educational level
plt.figure(figsize=(10,6))
sns.countplot(x='EDUCATION', hue='SALARY', data=df)
plt.title('Salary Distribution Within Each Educational Level')
plt.xlabel('Education Level')
plt.ylabel('Number of Respondents')
plt.xticks(rotation=90)
plt.show()