import os, pickle, re
import nltk
import numpy as np
import pandas as pd 
from rouge import Rouge 


class Evaluate(self,original,summary):
    def __init__(self):
        self.original = original
        self.summary = summary


    def Rouge():
        rouge = Rouge()
        scores = rouge.get_scores(self.original,self.summary)
        return scores
        


