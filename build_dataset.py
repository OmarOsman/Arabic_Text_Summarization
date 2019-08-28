import os, pickle, re
import nltk
import numpy as np
import pandas as pd 
from scipy.spatial.distance import cosine
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer

class Dataset():

    def __init__(self , data_dir):
        self.data_dir = data_dir
        self.train_dir_name = os.path.join(data_dir ,'EASC-UTF-8/Articles/')
        self.test_dir_name =  os.path.join(data_dir ,'EASC-UTF-8/Articles/')
        self.sources_dir_name = os.path.join(data_dir ,'SOURCES.csv')
        self.stop_word_file= os.path.join(data_dir ,'arabic_stop.txt)
        self.data = pd.DataFrame()
        nltk.download('punkt')

    def get_files_list(dir_name) :
        listOfFile = os.listdir(dir_name)
        allFiles = list()
        for entry in listOfFile:
            fullPath = os.path.join(dir_name, entry) 
            if os.path.isdir(fullPath):
                allFiles = allFiles + getListOfFiles(fullPath)
            else:
                allFiles.append(fullPath)
                    
        return allFiles

    def get_wiki_titles(self):
        src = pd.read_csv(self.sources_dir_name)
        titles = src['URL'].str.split('/')
        titles = titles.str[-1].values
        return titles




    def read_dataset() :
        train_files_path = get_files_list(self.train_dir_name)
        test_files_path = get_files_list(self.test_dir_name)

        j = 0
        for i in range(len(train_files_path)) :
            train_path = train_files_path[i]
            test_path = test_files_path[j : j + 5]
            j+=5
            
            train_file = get_article_content(train_path)
            sum1 =  get_article_content(test_path[0])
            sum2 =  get_article_content(test_path[1])
            sum3 =  get_article_content(test_path[2])
            sum4 =  get_article_content(test_path[3)
            sum5 =  get_article_content(test_path[4])
            
            row_list = []
            row_list.extend((train_file, sum1, sum2,sum3,sum4,sum5))
            data.loc[len(data)] = row_list

        data['title'] = get_wiki_titles()
        self.data  = data
        return 

