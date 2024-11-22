# Exploratory Data Analysis (EDA) on Iris Dataset

```python
# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
df = sns.load_dataset("iris")

# 1. Classifying Dependent and Independent Variables
dependent_var = "species"  # Target column
independent_vars = df.columns[:-1].tolist()  # All columns except the target
print(f"Dependent Variable: {dependent_var}")
print(f"Independent Variables: {independent_vars}")

# 2. Reporting Dataset Information
print(f"Shape of Dataset: {df.shape}")
print("\nData Types:\n", df.dtypes)
print("\nUnique Value Counts per Feature:\n", df.nunique())

# 3. Check for Missing Values
missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values)

# 4. Summary Statistics
print("\nSummary Statistics:")
print("Mean:\n", df.mean(numeric_only=True))
print("Median:\n", df.median(numeric_only=True))
print("Mode:\n", df.mode().iloc[0])
print("Range:\n", df.max(numeric_only=True) - df.min(numeric_only=True))
print("Standard Deviation:\n", df.std(numeric_only=True))

# 5. Visualizations
# a. Pair Plot
sns.pairplot(df, hue="species")
plt.show()

# b. Scatter Plot (Example: Sepal Length vs Sepal Width)
plt.scatter(df["sepal_length"], df["sepal_width"], c="blue", alpha=0.6)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Scatter Plot: Sepal Length vs Sepal Width")
plt.show()

# c. Histogram (Example: Sepal Length)
df["sepal_length"].hist(bins=20, color="green", alpha=0.7)
plt.title("Histogram: Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()

# d. Heatmap Correlation
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
