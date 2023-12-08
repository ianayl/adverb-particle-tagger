import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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
window_size = 5  # This is the total size of the feature vector
padding_tag = 'UH'

# Function to pad and extract features
def pad_and_extract_features(vector, window_size, padding_tag):
    # Pad the vector if it's shorter than the window size
    padded_vector = vector + [padding_tag] * (window_size - len(vector))
    return padded_vector[:window_size]

# Pad and extract features for each vector
padded_feature_vectors = [pad_and_extract_features(vector, window_size, padding_tag) for vector in feature_vectors]

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# AT THIS POINT I SHOULD HAVE REALIZED IT'S A BAD IDEA TO DO PCA BUT WE TRIED ANYWAYS.
# One Hot Encoder encodes the values as either 1 or 0, which is not good for tags since 
# There are too many contexts,tags,relations to work with.
# PCA in general is also not recommended.


# Encode each padded feature vector
encoded_vectors = [encoder.fit_transform(np.array(vector).reshape(-1, 1)) for vector in padded_feature_vectors]
print(encoded_vectors)

# Concatenate the encoded feature vectors
concatenated_features = np.concatenate(encoded_vectors, axis=1)

# Standardize the features
scaler = StandardScaler()
standardized_features = scaler.fit_transform(concatenated_features)

# Apply PCA
pca = PCA(n_components=100)  # Adjust n_components as needed
principal_components = pca.fit_transform(standardized_features)

print(principal_components)


# Assuming pca and encoder are already defined and used
loadings = pca.components_
feature_names = encoder.get_feature_names_out()

print(feature_names)