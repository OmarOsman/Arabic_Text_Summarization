import os, pickle, re
import nltk
import numpy as np
import pandas as pd 
import subprocess
import unicodedata as ud
import pdb
from scipy.spatial.distance import cosine
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
from collections import defaultdict


class Preprocess():
    def __init__(self ,stop_list_dir = r"tools\stop_words_list\stopwords.txt"):
        self.stop_list_dir = stop_list_dir

    def preprocess_df(self,org_df):
        df = org_df.copy() 
        for col in df.columns :
            if df[col].dtype == np.object:
                df[col] = df[col].apply(lambda x : self.get_clean_article(x))
                df[col] = df[col].apply(lambda x : self.get_article_sentences(x))
                        
        return df

    

  

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
            return open(article_path , 'r' ,encoding = 'utf-8').read()

    def get_article_paragraphes(self,text) :
        return [paragraph for paragraph in text.split('\n') if len(paragraph) > 1]
    
    def get_cleaned_article_paragraphes(self,text) :
        return [paragraph.replace('ppp',"") for paragraph in text.split('ppp') if len(paragraph) > 1]


    def get_article_sentences(self, text ,delim = "[.!?]+"): # tokenize based on token , default dot 
        return [sentence.replace('ppp',"").strip() for sentence in  re.split(delim,text) if len(sentence) > 1]
    
    def get_para_sentences(self,paragraphs):
        para_sent_list = [p.split('.') for p in paragraphs]
        para_sent_list_new = []
        for p in para_sent_list :
            l = []
            for s in p :
                if len(s) > 1 : 
                    l.append(s)
            para_sent_list_new.append(l)
            
        return para_sent_list_new

    def get_tokenized_word_sentences(self,sentences):
        """
        input : list of sentences 
        """
        sentence_list = []
        for sentence in sentences :
            words = word_tokenize(sentence.replace('ppp',""))
            needed_words = []
            for w in words:
                if w != '\ufeff':
                    needed_words.append(w)
            filtered_sentence = ""
            if len(needed_words) > 1 : 
                filtered_sentence = " ".join(needed_words)
                sentence_list.append(needed_words)
        return sentence_list


    def stop_word_remove(self ,text):
        ar_stop_list = open(self.stop_list_dir, "r")
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
        text = text.replace("\"" ," ").strip("''")
        text = text.replace('\ufeff' ," ")
        text = text.replace("``" ," ").strip()
        text = text.replace('\n' ,"ppp")
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
            if c == '.': lst.append(c)
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
        text = self.get_article_content(path)
        return text



    def stemming_ISR(self,text):
        st = ISRIStemmer()
        stemmed_words = []
        words = word_tokenize(text)
        for w in words:
            stemmed_words.append(st.stem(w))
        stemmed_text = " ".join(stemmed_words)
        return stemmed_text

    
    def get_golden_summary(self,df):
        


        for index, row in flights.head().iterrows():
            org_summary = df['Orignal'][index]
            sumar_1= df['Summary1'][index]
            sumar_2= df['Summary2'][index]
            sumar_3= df['Summary3'][index]
            sumar_4= df['Summary4'][index]
            sumar_5= df['Summary5'][index]
            all_summaries = [sumar_1,sumar_2,sumar_3,sumar_4,sumar_5]
            sent = self.get_article_sentences(self.get_clean_no_stemming(org_summary))
            sent_idx = { s:i for i,s in enumerate(sent) }


            res = []
            for sam in range(len(all_summaries)):
                s = self.get_article_sentences(self.get_clean_no_stemming(sam))
                res.append((sent_idx[s],s))

            pdb.set_trace()
            dic = defaultdict(int)
            golden_list = []
            for i in range (len (res)):
                for l in range(len(res[i])):
                    dic[res[1][i][l]]+=1

            pdb.set_trace()
            return dic

            








    def get_clean_article(self, text):
        text = self.normalize(text)
        text = self.stop_word_remove(text)
        #text =  self.stemming(text)
        text = self.stemming_ISR(text)
        return text

    def get_clean_no_stemming(self,text):
        text = self.normalize(text)
        text = self.stop_word_remove(text)
        return text

