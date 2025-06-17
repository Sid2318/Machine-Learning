import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('Mall_Customers.csv')

X = df.iloc[:,[3,4]]
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

st.title("Customer Segmentation using K-Means Clustering")

#elbow diagram

fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss, marker='o')
ax.set_title('The Elbow Diagram')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('WCSS')
st.pyplot(fig)

n = st.number_input("Enter the number of clusters (n_clusters):", min_value=1, max_value=10, value=3, step=1)
kmeans = KMeans(n_clusters=n, random_state=0)
Y = kmeans.fit_predict(X)


fig, ax = plt.subplots()
colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 'pink']

for i in range(n):
    ax.scatter(
        X.loc[Y == i, X.columns[0]],
        X.loc[Y == i, X.columns[1]],
        s=50,
        c=colors[i % len(colors)],
        label=f'Cluster {i+1}'
    )

# Plot centroids
ax.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=100,
        c='black',
        label='Centroids'
)

ax.set_title('Customer Groups')
ax.set_xlabel('Annual Income')
ax.set_ylabel('Spending Score')
ax.legend()
    
    # Show plot in Streamlit
st.pyplot(fig)