import os, pickle, re
import nltk
import numpy as np
import pandas as pd 
import subprocess
from scipy.spatial.distance import cosine
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer


class Preprocess():
    def __init__(self):
        pass

    def preprocess(org_df):
        df = org_df.copy()
        
        for col in df.columns :
            if df[col].dtype == np.object:
                df[col] = df[col].apply(lambda x : get_clean_article(x))
                df[col] = df[col].apply(lambda x : get_article_sentences(x))
                        
        return df

  


    ##~~Pickle helpers~~#
    def getPickleContent(self, pklFile):
        with open (pklFile, 'rb') as fp:
            itemlist = pickle.load(fp)
        return itemlist

    def setPickleContent(self, fileName, itemList):
        with open(fileName+'.pkl', 'wb') as fp:
            pickle.dump(itemList, fp)
    #~~~~~~~~~~~~~~~~~~#

    def get_article_content(self, article_path):
        if os.path.exists(article_path):
            return open(article_pathj , 'r' ,encoding = 'utf-8').read()

    def get_article_paragraphes(self,text) :
        return [paragraph for paragraph in text.split('\n') if len(paragraph) > 1]


    def get_article_sentences(self, text ,token = "[.]+"): # tokenize based on token , default dot 
        org_regex = "[.]+"
        return [sentence for sentence in  re.split(org_regex,text)]


    def stop_word_remove(self ,text):
        ar_stop_list = open(list_dir, "r")
        stop_words = ar_stop_list.read().split('\n')
    
        words = word_tokenize(text)
        needed_words = []
        for w in words:
            if w not in (stop_words) and w != '\ufeff':
                needed_words.append(w)
                
        filtered_sentence = " ".join(needed_words)
      
    return filtered_sentence

    def normalize(self,text):
        text = re.sub(r"[إأٱآا]", "ا", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ؤ", "ء", text)
        text = re.sub(r"ئ", "ء", text)
        #text = re.sub(r"ه", "ة", text) 
        #text = re.sub(r'[^ا-ي ]', "", text)
        
        

        noise = re.compile(""" ّ      | # Tashdid
                                َ    | # Fatha
                                ً    | # Tanwin Fath
                                ُ    | # Damma
                                ٌ    | # Tanwin Damm
                                ِ    | # Kasra
                                ٍ    | # Tanwin Kasr
                                ْ    | # Sukun
                                ـ     # Tatwil/Kashida
                            """, re.VERBOSE)
        text = re.sub(noise, '', text)
        
        lst = []
        for c in text :
        if c == '.' : lst.append(c)
        if not ud.category(c).startswith('P') : lst.append(c)
        text = ''.join(lst)
        
        return text


    def stemming_khoja(self,text):
        myCmd = 'cd tools\stemmer'
        out = os.system(myCmd)    

        myCmd = '!echo "$text" > infile.txt' 
        out = os.system(myCmd) 

        subprocess.call(['java', '-jar', '.khoja-stemmer-command-line.jar'])
        myCmd = '!java -jar khoja-stemmer-command-line.jar infile.txt outfile.txt'
        out = os.system(myCmd)

        path = 'outfile.txt'
        text = get_article_content(path)
        return text



    def stemming_ISR(self,text):
        st = ISRIStemmer()
        stemmed_words = []
        words = word_tokenize(text)
        for w in words:
            stemmed_words.append(st.stem(w))
        stemmed_text = " ".join(stemmed_words)
        return stemmed_text




    def get_clean_article(self, text):
        text = normalize(text)
        text = stop_word_remove(text)
        text = stemming(text)
        return  text


    def getLimit(self, limit, num_sentences):
        return ( limit * num_sentences ) / 100