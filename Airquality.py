
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"/Users/anmolchadgal/Downloads/Airquality.csv")

print(df.head())
print(df.info())
print(df.shape)

#missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

#Summary statistics
print("\nSummary Statistics:")
print(df.describe())

#missing numeric values
df["pollutant_min"] = df["pollutant_min"].fillna(df["pollutant_min"].mean())
df["pollutant_max"] = df["pollutant_max"].fillna(df["pollutant_max"].mean())
df["pollutant_avg"] = df["pollutant_avg"].fillna(df["pollutant_avg"].mean())
df["pollutant_range"] = df["pollutant_max"] - df["pollutant_min"]


#Bar plot Average pollution per city
city_avg = df.groupby("city")["pollutant_avg"].mean().reset_index()
#Top 30 cities with highest average pollution
top_cities = city_avg.sort_values(by="pollutant_avg", ascending=False).head(30)
plt.figure(figsize=(14, 6))
sns.barplot(data=top_cities, x="city", y="pollutant_avg", hue="city", palette="coolwarm", legend=False)
plt.title("Top 30 Cities by Average Pollution Level")
plt.xlabel("City")
plt.ylabel("Average Pollutant Level")
plt.xticks(rotation=75)
plt.tight_layout()
plt.grid()
plt.show()


#Histogram Distribution of pollutant_avg
plt.figure(figsize=(8, 5))
plt.hist(df["pollutant_avg"], bins=15, color="skyblue", edgecolor="black")
plt.title("Distribution of Average Pollutants")
plt.xlabel("Pollutant Average")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# average pollutant by state
state_pollution = df.groupby("state")["pollutant_avg"].mean().sort_values(ascending=False)
top_states = state_pollution.head(10)
#pie chart
plt.figure(figsize=(8, 8))
plt.pie(top_states, labels=top_states.index, autopct="%1.1f%%", startangle=140)
plt.title("Top 10 States by Average Pollutant Level")
plt.axis("equal")  # Equal aspect ratio makes the pie chart circular
plt.show()

#Boxplot spread of pollutant values
sns.boxplot(data=df[["pollutant_min", "pollutant_max", "pollutant_avg"]])
plt.title("Boxplot of Pollutant Levels")
plt.grid()
plt.show()

#Scatter plot Latitude vs pollutant_avg
sns.scatterplot(data=df, x="latitude", y="pollutant_avg", hue="state", palette="coolwarm")
plt.title("Pollutant Average by Latitude")
plt.xlabel("Latitude")
plt.ylabel("Pollutant Avg")
plt.legend(loc='center left', bbox_to_anchor=(0.97, 0.5))
plt.grid()
plt.show()

# Correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df[["pollutant_min", "pollutant_max", "pollutant_avg", "pollutant_range"]].corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Detecting outliers (visualize using boxplot)
sns.boxplot(x=df["pollutant_avg"])
plt.title("Outlier Detection in Pollutant Avg")
plt.grid()
plt.show()
