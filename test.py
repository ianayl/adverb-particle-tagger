from input_lib import *
from pca_sentences import * 

eigenvecs = []
for i in range(0,15):
    eigenvecs.append(get_centered_vectors_from_file("../CleanedDataRP.json", i))


