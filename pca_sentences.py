#!/usr/bin/env python3

#
# pca_sentences.py:
# Code for actually performing PCA on sentences (extracting most characteristic
# eigenvectors on sentences)
#

from input_lib import *
from typing import List
from sklearn.decomposition import PCA

def get_centered_vectors(s: Sentence, size: int) -> List[List]:
    """
    From a sentence s, get vectors centered around adverbial particles (RP's).

    @param size The number of words to capture surrounding the RP -- if there
                are not enough words around the RP to capture, do not create 
                such vector.
    """
    res = []
    # Only pad the sentence with one UH -- if you pad with too many UH's then
    # you have unreasonably long sentences
    padded_sentence = [POS.UH] + s.morph + [POS.UH]
    for i, tag in enumerate(s.morph): 
        # TODO: we cannot actually pad anything! padding makes it wrong!
        if tag == POS.RP:
            bounds = [i-size+1, i+size+2]
            if bounds[0] < 0 or bounds[1] >= len(padded_sentence):
                continue
            res.append(padded_sentence[bounds[0]:bounds[1]])
    return res

def get_centered_vectors_from_file(filename: str, size: int) -> List[List]:
    tmp = get_inputs_from_json(filename)
    return [ v for s in tmp for v in get_centered_vectors(s, size) ]

def perform_PCA(vectors: List[List]):
    pca_res = PCA(n_components=7, whiten=False).fit(vectors)
    return pca_res.components_
