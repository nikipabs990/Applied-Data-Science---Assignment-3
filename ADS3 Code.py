# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:32:00 2023

@author: pabas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.preprocessing import StandardScaler

def plot_scatter_by_gender(df, year, ax=None):
    """
    This function takes in a pandas DataFrame and a year. It filters the DataFrame for the given year and
    plots a scatter plot of male and female life expectancy for each location in the dataset.
    """
    if ax is None:
        ax = plt.gca()

    # Filter data for the given year
    df_year = df[df['Year'] == year]

    # Plot a scatter plot for male and female life expectancy
    ax.scatter(df_year['Male'], df_year['Female'])
   
def plot_life_expectancy_by_gender(df, location, year_col, male_col, female_col):
    """
    This function takes in a pandas DataFrame, a location, and column names for year, male life expectancy,
    and female life expectancy. The function plots a bar chart of male and female life expectancy over time
    for the specified location.
    """

    # Filter data for the given location
    df_location = df[df['Location'] == location]

    # Plot a bar chart for male and female life expectancy
    fig, ax = plt.subplots()
    ax.bar(df_location[year_col] - 0.2, df_location[male_col],
           width=0.4, align='center', label='Male')
    ax.bar(df_location[year_col] + 0.2, df_location[female_col],
           width=0.4, align='center', label='Female')
    ax.set_xticks(df_location[year_col])
    ax.set_xlabel('Year')
    ax.set_ylabel('Life expectancy')
    ax.set_title(f'Life expectancy in {location} by gender')
    ax.legend()
    plt.show()


def plot_pie_chart(df_life_ex, year):
    """
    This function takes in a pandas DataFrame and a year. It filters the DataFrame for the given year and
    plots a pie chart of the percentage of life expectancy for both sexes by location.
    """

    # filter for the given year
    df = df_life_ex[df_life_ex['Year'] == year]

    # plot pie chart for Both Sexes column
    labels = df['Location']
    sizes = df['Both Sexes']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

    # add legend to top right corner
    plt.legend(title='Location', loc='upper right', bbox_to_anchor=(1.3, 1))

    # add title
    plt.title('Both Sexes Percentage in {}'.format(year))

    # show plot
    plt.show()
   
df_life_ex = pd.read_excel(
    'C:/My assignments/ADS 1/Healthy Life Expectancy(HALE) at age 60 (years).xlsx' )
print(df_life_ex)


plot_scatter_by_gender(df_life_ex, 2015)

plot_pie_chart(df_life_ex, 2015)

plot_life_expectancy_by_gender(df_life_ex, 'Africa', 'Year', 'Male', 'Female')

# Generate sample data with 3 clusters
X, y = make_blobs(n_samples=500, centers=3, random_state=42)

# Apply clustering algorithm (K-means)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get cluster labels and cluster centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Create a scatter plot for cluster membership
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.colorbar(label='Cluster')

# Plot the cluster centers
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='Cluster Centers')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Membership and Cluster Centers')
plt.legend()
plt.show()

def linear_model(x, a, b):
    return a * x + b


x_data = df_life_ex['Year'].values
y_data = df_life_ex['Male'].values


plt.scatter(x_data, y_data, label='Data')
plt.xlabel('Year')
plt.ylabel('Male Life Expectancy')
plt.title('Linear Model Fit')
plt.legend()
plt.show()

# Fit the linear model to the data
popt, _ = curve_fit(linear_model, x_data, y_data, p0=[1, 1])

# Make predictions for future values
x_future = np.array([2030])
y_pred = linear_model(x_future, *popt)

# Compute the confidence interval using err_ranges
def err_ranges(x, y, y_fit, popt, pcov, alpha=0.05):
    n = len(y)
    dof = n - len(popt)
    t_val = abs(stats.t.ppf(alpha / 2, dof))
    s_err = np.sqrt(np.diag(pcov))
    confs = t_val * s_err * np.sqrt(1 + 1 / n + (x - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    upper = y_fit + confs

#Feature Selection
selected_features = ['Male', 'Female', 'Both Sexes']

#Data Normalization
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df_life_ex[selected_features])

#Apply Clustering Algorithm
num_clusters = 3  # Choose the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(normalized_data)

#Cluster Interpretation
df_life_ex['Cluster'] = kmeans.labels_
representative_countries = []
for cluster_id in range(num_clusters):
    cluster_countries = df_life_ex[df_life_ex['Cluster'] == cluster_id]
    representative_country = cluster_countries.sample(1)  # Select one representative country per cluster
    representative_countries.append(representative_country)
    
#Comparison within Clusters
for country in representative_countries:
    cluster_id = country['Cluster'].iloc[0]
    cluster_countries = df_life_ex[df_life_ex['Cluster'] == cluster_id]
    # Compare countries within the same cluster and perform analysis
    print(f"Countries in Cluster {cluster_id}:")
    print(cluster_countries)

#Comparison between Clusters
for i, country1 in enumerate(representative_countries):
    for j, country2 in enumerate(representative_countries):
        if i != j:
            cluster_id1 = country1['Cluster'].iloc[0]
            cluster_id2 = country2['Cluster'].iloc[0]
            cluster_countries1 = df_life_ex[df_life_ex['Cluster'] == cluster_id1]
            cluster_countries2 = df_life_ex[df_life_ex['Cluster'] == cluster_id2]
            # Compare countries between different clusters and perform analysis
            print(f"Countries in Cluster {cluster_id1} compared with Cluster {cluster_id2}:")
            print("Cluster 1:")
            print(cluster_countries1)
            print("Cluster 2:")
            print(cluster_countries2)
