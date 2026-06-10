import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

print("Dataset Preview:")
print(df.head())

# -------------------------
# Scaling
# -------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# -------------------------
# Elbow Method
# -------------------------
wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.show()

# -------------------------
# Apply KMeans (choose K=3 based on elbow)
# -------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

df["Cluster"] = clusters

print("\nCluster Distribution:")
print(df["Cluster"].value_counts())

# -------------------------
# Visualization using PCA (2D)
# -------------------------
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

df["PCA1"] = reduced_data[:, 0]
df["PCA2"] = reduced_data[:, 1]

plt.figure(figsize=(7,5))
sns.scatterplot(
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    data=df,
    palette="viridis"
)

plt.title("K-Means Clusters Visualization (PCA Reduced)")
plt.show()

# -------------------------
# Cluster Centers
# -------------------------
print("\nCluster Centers (Scaled):")
print(kmeans.cluster_centers_)