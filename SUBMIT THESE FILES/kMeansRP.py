import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load sentences from JSON file
with open("CleanedDataRP_Modified.json", "r") as file:
    data = json.load(file)

# Extract morphological text
morphology_texts = [entry["morphology"] for entry in data]

# Vectorize morphological text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(morphology_texts)

# Create two figures, each with 3x1 subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 3x3 grid of subplots for clusters 3 to 8

# Apply KMeans clustering for cluster numbers from 3 to 8
for num_clusters in range(3, 9):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(X)

    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Find sentences closest to cluster centers
    closest_sentences, _ = pairwise_distances_argmin_min(cluster_centers, X)

    # Fit PCA to reduce dimensionality to 2 components
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X.toarray())  # Convert sparse matrix to array if using CountVectorizer

    # Plotting clustered sentences for each cluster number
    if num_clusters < 6:  # Plot clusters 3 to 5 in the first figure
        row = 0  # Row index for the first figure
        col = num_clusters - 3  # Column index for the first figure
    else:  # Plot clusters 6 to 8 in the second figure
        row = 1  # Row index for the second figure
        col = num_clusters - 6  # Column index for the second figure

    axs[row, col].scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans.labels_, cmap='viridis')
    axs[row, col].set_title(f'Clusters of Sentences (Number of Clusters: {num_clusters})')
    axs[row, col].set_xlabel('Principal Component 1')
    axs[row, col].set_ylabel('Principal Component 2')

plt.tight_layout(pad=3.0)
plt.show()