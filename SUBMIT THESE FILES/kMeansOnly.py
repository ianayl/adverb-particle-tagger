import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from collections import Counter

# Load sentences from JSON file
with open("CleanedDataRP_Modified.json", "r") as file:
    data = json.load(file)

# Extract morphological text
morphology_texts = [entry["morphology"].split() for entry in data]  # Split by word to consider each as a tag

# Vectorize morphological text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([" ".join(entry) for entry in morphology_texts])

# Store statistics
cluster_stats = {}
overall_word_freq = Counter()

for num_clusters in range(3, 8):
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(X)

    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Find sentences closest to cluster centers
    closest_sentences, _ = pairwise_distances_argmin_min(cluster_centers, X)

    print(f"Closest sentences for {num_clusters} clusters:")
    for i, sentence_idx in enumerate(closest_sentences):
        print(f"Cluster {i + 1}: {' '.join(morphology_texts[sentence_idx])}")

    # Compute statistics per cluster
    for cluster_label in set(kmeans.labels_):
        indices = [i for i, label in enumerate(kmeans.labels_) if label == cluster_label]
        words_in_cluster = [morphology_texts[i] for i in indices]
        flattened_words = [word for sublist in words_in_cluster for word in sublist]
        word_freq = Counter(flattened_words)
        total_words = sum(word_freq.values())
        word_percentages = {word: (count / total_words) * 100 for word, count in word_freq.items()}
        cluster_stats.setdefault(cluster_label, []).append(word_percentages)

    # Update overall word frequencies
    all_words = [word for sublist in morphology_texts for word in sublist]
    overall_word_freq.update(Counter(all_words))

# Compute overall statistics
total_words = sum(overall_word_freq.values())
overall_word_percentages = {word: (count / total_words) * 100 for word, count in overall_word_freq.items()}

# Display cluster stats
print("\nStats per cluster:")
for cluster, stats in cluster_stats.items():
    print(f"Cluster {cluster}:")
    for idx, word_stats in enumerate(stats):
        print(f"\tSentence {idx + 1}: {word_stats}")

# Display overall stats
print("\nOverall stats for the entire dataset:")
print(overall_word_percentages)