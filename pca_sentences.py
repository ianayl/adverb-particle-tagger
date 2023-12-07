#!/usr/bin/env python3

#
# pca_sentences.py:
# Code for actually performing PCA on sentences (extracting most characteristic
# eigenvectors on sentences)
#

from input_lib import *
from typing import List

def get_centered_RP_vectors(s: Sentence, pad: int) -> List[List]:
    """
    From a sentence s, get vectors centered around adverbial particles (RP's).

    @param pad The number of words to capture surrounding the RP -- if there
               are not enough words around the RP to capture, do not create such
               vector.
    """
    res = []
    for i, tag in enumerate(s.morph): 
        # TODO: we cannot actually pad anything! padding makes it wrong!
        if tag == POS.RP:
            centered = [POS.UH]
