import os, pickle, re
import nltk
import numpy as np
import pandas as pd 
from rouge import Rouge 


class Evaluate():
    def __init__(self):
        self.original = original
        self.summary = summary


    def Rouge(self):
        rouge = Rouge()
        scores = rouge.get_scores(self.original,self.summary)
        return scores
        


