# Python-Week-7

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
try:
    iris = pd.read_csv(url)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("File not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Display first few rows
print("First 5 Rows of the Dataset:")
print(iris.head())

# Data types
print("\nData Types:")
print(iris.dtypes)

# Missing values
print("\nMissing Values:")
print(iris.isnull().sum())

# Basic statistics
print("\nBasic Statistics:")
print(iris.describe())

# Group by species and compute mean
print("\nAverage Measurements per Species:")
print(iris.groupby("species").mean())

# Visualizations
sns.set(style="whitegrid")

# Line chart (simulated trend using index)
plt.figure(figsize=(10, 6))
plt.plot(iris.index, iris["sepal_length"], label="Sepal Length")
plt.plot(iris.index, iris["petal_length"], label="Petal Length")
plt.title("Line Chart: Sepal and Petal Lengths Over Index")
plt.xlabel("Index")
plt.ylabel("Length (cm)")
plt.legend()
plt.show()

# Bar chart (average petal length per species)
plt.figure(figsize=(8, 5))
sns.barplot(data=iris, x='species', y='petal_length', estimator=np.mean, ci=None)
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length")
plt.show()

# Histogram of sepal length
plt.figure(figsize=(8, 5))
plt.hist(iris["sepal_length"], bins=20, color="skyblue", edgecolor="black")
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()

# Scatter plot (sepal length vs petal length)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=iris, x="sepal_length", y="petal_length", hue="species")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()
plt.show()
