import os, pickle, re
import nltk
import numpy as np
import pandas as pd 
import preprocess
import pdb
from scipy.spatial.distance import cosine
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer

class Dataset():

    def __init__(self , data_dir = 'data'):
        self.data_dir = data_dir
        #self.train_dir_name =  os.path.join(self.data_dir ,'/EASC-UTF-8/Articles/')
        #self.test_dir_name  =  os.path.join(self.data_dir ,'/EASC-UTF-8/Articles/')
        #self.train_dir_name =  'data/EASC-UTF-8/Articles/'
        #self.test_dir_name  =  'data/EASC-UTF-8/Articles/'

        self.train_dir_name =  'data\EASC-UTF-8\Articles'
        self.test_dir_name =   'data\EASC-UTF-8\MTurk'
        self.data =pd.DataFrame(columns = ['Orignal' ,'Summary1' ,'Summary2' ,'Summary3' ,'Summary4' ,'Summary5'])
        self.pr = preprocess.Preprocess()
        

    def get_files_list(self,dir_name) :
        listOfFile = os.listdir(dir_name)
        allFiles = list()
        for entry in listOfFile:
            fullPath = os.path.join(dir_name, entry) 
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.get_files_list(fullPath)
            else:
                allFiles.append(fullPath)
                    
        return allFiles

    def read_dataset(self) :
        train_files_path = self.get_files_list(self.train_dir_name)
        test_files_path = self.get_files_list(self.test_dir_name)

        j = 0
        for i in range(len(train_files_path)) :
            train_path = train_files_path[i]
            test_path = test_files_path[j : j + 5]
            j+=5
            
            train_file = self.pr.get_article_content(train_path)
            sum1 =  self.pr.get_article_content(test_path[0])
            sum2 =  self.pr.get_article_content(test_path[1])
            sum3 =  self.pr.get_article_content(test_path[2])
            sum4 =  self.pr.get_article_content(test_path[3])
            sum5 =  self.pr.get_article_content(test_path[4])
            
            row_list = []
            row_list.extend((train_file, sum1, sum2,sum3,sum4,sum5))
            self.data .loc[len(self.data )] = row_list

        return  self.data 

