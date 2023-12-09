#!/usr/bin/env python3

#
# input_lib.py:
# Functions and datatypes related to reading input sentences PCA
#

import json
from enum import IntEnum
from typing import List

# Don't need to identify RP or not right now, just eigenclasses that resemble PR's'

# TODO have Robyn try to put these next to eachother in a way that makes sense
class POS(IntEnum):
    """
    Enum representing part of speech tags. Types are defined on this webpage:
    https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    Note that:
    - SYMB is any symbol
    - WPS is WP$ (possessive wh-pronoun)
    - PRPS is PRP$ (possessive pronoun)
    - HYPH is a literal hyphen
    - QUOT is a quote (`` or '')
    - I don't know what XX is supposed to be
    - I don't knwo what NFP is supposed to be
    - I don't knwo what ADD is supposed to be
    - I don't knwo what AFX is supposed to be
    """
    SYMB   = 0
    CC	   = 1
    CD	   = 2
    DT	   = 3
    EX	   = 4
    FW	   = 5
    IN	   = 6
    JJ	   = 7
    JJR	   = 8
    JJS	   = 9
    LS	   = 10
    MD	   = 11
    NN	   = 12
    NNS	   = 13
    NNP	   = 14
    NNPS   = 15
    PDT	   = 16
    POS	   = 17
    PRP	   = 18
    PRPS   = 19
    RB	   = 20
    RBR	   = 21
    RBS	   = 22
    RP	   = 23
    SYM	   = 24
    TO	   = 25
    UH	   = 26
    VB	   = 27
    VBD	   = 28
    VBG	   = 29
    VBN	   = 30
    VBP	   = 31
    VBZ	   = 32
    WDT	   = 33
    WP	   = 34
    WPS	   = 35
    WRB	   = 36
    HYPH   = 37
    QUOT   = 38
    XX     = 39
    PERIOD = 40
    COMMA  = 41
    COLON  = 42
    NFP    = 43
    ADD    = 44
    AFX    = 45

    @staticmethod
    def from_str(s):
        """
        Obtain a POS tag from a string s, if it represents a valid POS type.
        """
        if s == "CC":         return POS.CC
        if s == "CD":         return POS.CD
        if s == "DT":         return POS.DT
        if s == "EX":         return POS.EX 
        if s == "FW":         return POS.FW
        if s == "IN":         return POS.IN
        if s == "JJ":         return POS.JJ
        if s == "JJR":        return POS.JJR
        if s == "JJS":        return POS.JJS
        if s == "LS":         return POS.LS
        if s == "MD":         return POS.MD
        if s == "NN":         return POS.NN
        if s == "NNS":        return POS.NNS
        if s == "NNP":        return POS.NNP
        if s == "NNPS":       return POS.NNPS
        if s == "PDT":        return POS.PDT
        if s == "POS":        return POS.POS
        if s == "PRP":        return POS.PRP
        if s == "PRP$":       return POS.PRPS
        if s == "RB":         return POS.RB
        if s == "RBR":        return POS.RBR
        if s == "RBS":        return POS.RBS
        if s == "RP":         return POS.RP
        if s == "SYM":        return POS.SYM
        if s == "TO":         return POS.TO
        if s == "UH":         return POS.UH
        if s == "VB":         return POS.VB
        if s == "VBD":        return POS.VBD
        if s == "VBG":        return POS.VBG
        if s == "VBN":        return POS.VBN
        if s == "VBP":        return POS.VBP
        if s == "VBZ":        return POS.VBZ
        if s == "WDT":        return POS.WDT
        if s == "WP":         return POS.WP
        if s == "WP$":        return POS.WPS
        if s == "WRB":        return POS.WRB
        if s == "HYPH":       return POS.HYPH
        if s == "XX":         return POS.XX
        if s == "NFP":        return POS.NFP
        if s == "ADD":        return POS.ADD
        if s == "AFX":        return POS.AFX
        if s == ".":          return POS.PERIOD
        if s == ",":          return POS.COMMA
        if s == ":":          return POS.COLON
        if s in ["$"]:        return POS.SYMB 
        if s in ["``", "''"]: return POS.QUOT 
        raise ValueError("No POS tag with value \"" + s + "\".")


class Sentence:
    """
    Class for representing a sentence as a list of POS tags and its original
    text
    """
    def __init__(self, morph: List[POS], text: str):
        self.morph = morph
        self.text = text

    def __str__(self):
        return f"Sentence({self.morph}, \"{self.text}\")"

    def __repr__(self):
        return f"Sentence({self.morph}, \"{self.text}\")"


def get_inputs_from_json(filename: str) -> List[Sentence]:
    with open(filename, 'r') as file:
        return [ Sentence(
                     text=s["text"],
                     morph=list(map(lambda x: POS.from_str(x),
                                s["morphology"].split(" ")))
                 ) for s in json.load(file) ]
