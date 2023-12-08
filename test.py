from input_lib import *
from pca_sentences import * 
import numpy as np
import math

def euclidian_distance(x, y):
    return np.linalg.norm(x-y)

def custom_PCA(vectors):
    data_mat = np.array(vectors)
    covar = data_mat.T @ data_mat
    # covar = np.cov(data_mat.T)
    e_vals, e_vecs = np.linalg.eigh(covar)
    e_vecs = e_vecs.T
    return sorted(zip(e_vals, e_vecs), key=lambda x: x[0], reverse=True)

centered_vecs = []
for i in range(1,15):
    centered_vecs.append(get_centered_vectors_from_file("../CleanedDataRP.json", i))

for i in range(5,14):
    dataset = centered_vecs[i]
    # splitting the data into training and testing
    np.random.shuffle(dataset)
    split = math.floor(len(dataset) * 0.8)
    training, test = dataset[:split], dataset[split+1:]

    # Generating result of applying eigenvectors onto vectors in the testing split
    sigma_k = np.array([ ev[1] for ev in custom_PCA(training)])
    model_res = (sigma_k.T @ sigma_k @ test[3])
    print(f"Loop increment {i}:")
    print(f"original = {test[3]}")
    print(f"result   = {model_res}")
    print(f"Euclidian distance: {euclidian_distance(model_res, test[3])}")
