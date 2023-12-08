import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Load the JSON file
with open('CleanedDataRP_Modified.json', 'r') as file:
    data = json.load(file)

feature_vectors = []

# Function to extract features around a given index
def extract_features(tags, index, window_size=3):
    start = max(0, index - window_size)
    end = min(len(tags), index + window_size + 1)
    return tags[start:end]

# Iterate over each sentence in the data
for item in data:
    tags = item["morphology"].split()

    # Find instances of the predicate (RP) and create feature vectors
    for i, tag in enumerate(tags):
        if tag == 'RP':
            feature_vector = extract_features(tags, i)
            feature_vectors.append(feature_vector)

# Define window size and padding tag
window_size = 7  # This is the total size of the feature vector
padding_tag = 'UH'

# Function to pad and extract features
def pad_and_extract_features(vector, window_size, padding_tag):
    # Pad the vector if it's shorter than the window size
    padded_vector = vector + [padding_tag] * (window_size - len(vector))
    return padded_vector[:window_size]

for i in range(len(feature_vectors)):
    feature_vectors[i] = pad_and_extract_features(feature_vectors[i],window_size,padding_tag)


# Convert the array of POS tag arrays into text strings
pos_tags_corpus_text = [' '.join(tags) for tags in feature_vectors]

# Create a CountVectorizer instance
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')

# Fit and transform the POS tag text into a Bag of Words representation
bow_matrix = vectorizer.fit_transform(pos_tags_corpus_text)

# Get the vocabulary (unique POS tags) and their corresponding indices
vocabulary = vectorizer.get_feature_names_out()

# Convert the BoW matrix to a dense array for easier manipulation (optional)
bow_matrix_dense = bow_matrix.toarray()

# Print the BoW representation and vocabulary
print("Bag of Words (BoW) Representation:")
print(bow_matrix_dense)
print(len(bow_matrix_dense))
print("\nVocabulary (Unique POS Tags):")
print(vocabulary)

# Calculate Euclidean Distance between all pairs of sentences
euclideanDistances = euclidean_distances(bow_matrix_dense)


# Example: Number of clusters (you can adjust this)
num_clusters = 7

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(euclideanDistances)

# Assign cluster labels to each sentence in your corpus
for i, label in enumerate(cluster_labels):
    print(f"Sentence {i+1}: Cluster {label}")

# Reduce the dimensionality using t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=0)  # You can adjust perplexity and other parameters
bow_matrix_tsne = tsne.fit_transform(euclideanDistances)

# Visualize the clusters using scatter plot
plt.scatter(bow_matrix_tsne[:, 0], bow_matrix_tsne[:, 1], c=cluster_labels, cmap='viridis')
plt.title('t-SNE Visualization of Clusters')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Cluster')
plt.show()    
# # Reduce the dimensionality using PCA
# pca = PCA(n_components=2)  # You can choose 2 or 3 components for 2D or 3D visualization
# bow_matrix_pca = pca.fit_transform(bow_matrix_dense)

# # Visualize the clusters using scatter plot
# plt.scatter(bow_matrix_pca[:, 0], bow_matrix_pca[:, 1], c=cluster_labels, cmap='viridis')
# plt.title('PCA Visualization of Clusters')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.colorbar(label='Cluster')
# plt.show()
