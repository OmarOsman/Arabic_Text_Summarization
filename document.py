import preprocess as pp
import numpy as np
import math
import pdb
import yake

from numpy import dot
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict
from  sklearn.metrics.pairwise import cosine_similarity


class Doc():

    def __init__(self,original_text , original_sentences , preprocessed_text,
    sentences,paragraphs,para_sent_list ,tokenized_word_sentences,
    title = "" ,golden_summary  = "" ,five_summaries = "") :
        self.original_text = original_text
        self.original_sentences = original_sentences
        self.preprocessed_text = preprocessed_text
        self.golden_summary = golden_summary
        self.five_summaries = five_summaries
        self.title = title 
        self.paragraphs = paragraphs # list of paragrpahs 
        self.para_sent_list = para_sent_list
        self.sentences = sentences
        self.tokenized_word_sentences = tokenized_word_sentences
        self.sentences_length = self.get_sentences_length()
        self.sent2idx = self.sentence2index()
        
        self.number_unique_vocab = 0
        self.key_phrases = self.get_doc_key_phrase(self.preprocessed_text) #list of tuples
        self.key_phrase_frequency = { }# dict
        self.key_phrase_length = self.get_key_phrase_length()
        self.key_phrase_proper_name = self.get_key_phrase_proper_name()
        
        self.topic_threshold = 0.3
        self.tf_idf ,self.tfidf_array ,self.centroid_vector = self.get_tfidf_centroid_vector()
        

        self.tf_idf_matrix = self.get_tfidf_matrix()
        self.cosine_similarity_matrix = self.get_pairwise_cosine_similarity_matrix()
        self.similarity_threshold = 0.1
        self.similartiy_degree_dic = self.get_similarity_degree_dic()
        self.max_similarity_degree = self.get_max_similarity_degree()
        
        
        
        
        


    def sentence2index(self):
        return { s:i for i,s in enumerate(self.sentences)}

    def orgsentence2index(self):
        return { s:i for i,s in enumerate(self.original_sentences)}


    
    
    def get_sentences_length(self):
        return [len(sentence) for sentence in self.sentences]

    
#### Key Phrase Feature    
    def get_doc_key_phrase(self,text):
        max_ngram_size = 1 # specifying parameters
        custom_kwextractor = yake.KeywordExtractor(lan="ar", n = max_ngram_size, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=15, features=None)
        keywords = custom_kwextractor.extract_keywords(text)
        return keywords

    def get_key_phrase_frequency(self,sentence) :
        """it indicates how many times the keyphrase appeared in the sentence

            Parameters:
            sentence (string): the sentence string 

            Returns:
            dict:dict where each element is {key_phrase : key_phrase_frequency}

        """
        key_phrase_frequency = defaultdict(int)
        total_number_kp_doc = 0

        for kp in self.key_phrases :
            number_sentences_contain_KP = 0
            total_number_kp_doc += self.preprocessed_text.count(kp[0])
            for s in self.sentences :
                if kp[0] in s :
                    number_sentences_contain_KP +=1
            if number_sentences_contain_KP > 0 : key_phrase_frequency[kp[0]] = number_sentences_contain_KP

        #self.normalized_key_phrase_frequency = key_phrase_frequency.copy()
        #for i in range(len(normalized_key_phrase_frequency)) = normalized_key_phrase_frequency[i] = list(normalized_key_phrase_frequency[i])
        for key in key_phrase_frequency.keys() : key_phrase_frequency[key] /= total_number_kp_doc
        return key_phrase_frequency


    def get_key_phrase_length(self) :
        """it indicates how many times the keyphrase appeared in the sentence

            Parameters:
             

            Returns:
            list:lists where each element is the square root of keyphrase 

        """
        key_phrase_length = { kp : math.sqrt(len(kp)) for kp in self.key_phrase_frequency.keys()}
        return key_phrase_length


    def get_key_phrase_proper_name(self):
        key_phrase_proper_name = {kp : 1 for kp in self.key_phrase_frequency.keys()}
        return key_phrase_proper_name


    def get_key_phrase_score(self,sentence):

        """compute the keyphrase score for input sentence 

            Parameters:
            sentence (string): the sentence string 

            Returns:
            float : total_key_phrase_score

        """     
        total_key_phrase_score = 0
        for kp in self.key_phrase_frequency:
            if kp in sentence :
                total_key_phrase_score += self.key_phrase_frequency[kp] *  self.key_phrase_length[kp] *                     self.key_phrase_proper_name[kp]
        return total_key_phrase_score
    
    
    
#### Sentence length Score Feature
    def outliers_iqr(self,sentence_length):
        
        """ compute interquartile range for current sentence
        """
        quartile_1, quartile_3 = np.percentile(self.sentences_length, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        return  1 if (sentence_length > upper_bound) | (sentence_length < lower_bound) else 0
    
    def sentence_length_score(self,sentence):
        document_max_length = max(self.sentences_length)
        is_outlier = self.outliers_iqr(len(sentence))
        
        if is_outlier :  return 0
        else : return len(sentence) / document_max_length

            
          
 
  

#### Sentence Location Score 
    def sentence_location_score(self,sentence):
        """compute the sentence_location_scorerase score for input sentence 

        Parameters:
        sentence (string): the sentence string 

        Returns:
        float : the sentence_location_score score 

        """
        
        for paragrpah_index,list_para in enumerate(self.para_sent_list) :
            for sent_index , sent in enumerate(list_para):            
                if sentence.strip() == sent.strip() : 
                    if sent_index  == 0 :
                        if paragrpah_index == 0 : return 3  
                        elif paragrpah_index == len(self.paragraphs) - 1 : return 2
                        else : return 1
                    
                    elif paragrpah_index == 0  or  paragrpah_index == len(self.paragraphs) - 1 : return 1 / math.sqrt(sent_index)
        return  1 / (math.sqrt(sent_index + (paragrpah_index * paragrpah_index)))

    
#### Similarity 
    def cosine_similarity_V1(self,vec_sentence_1 ,vec_sentence_2):
        """compute the cosine similarity  between two sentence vectors 

        Parameters:
         1D numpy_array (vec_sentence_1): sentence_1 vector 
         1D numpy_array (vec_sentence_2): sentence_2 vector 
        

        Returns:
        float : the cosine similarity between two vectors

        """
        cos_sim = dot(vec_sentence_1, vec_sentence_2) / (norm(vec_sentence_1) * norm(vec_sentence_2))
        return cos_sim




    def cosine_similarity_V2(self,vec_sentence_1 ,vec_sentence_2):
        """compute the cosine similarity  between two sentence vectors 

        Parameters:
         1D numpy_array (vec_sentence_1): sentence_1 vector 
         1D numpy_array (vec_sentence_2): sentence_2 vector 
        

        Returns:
        float : the cosine similarity between two vectors

        """
        vec_1 = np.array(vec_sentence_1).reshape(-1,1)
        vec_2 = np.array(vec_sentence_2).reshape(-1,1)

        score = 0.0
        if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
            score = ((1 - cosine(v1, v2)) + 1) / 2
        return score
        
        

    

    def similarity_title_score(self,sentence):
        """compute the sentence_location_scorerase score for input sentence 

        Parameters:
        sentence (string): the sentence string 

        Returns:
        float : the sentence_location_score score 

        """
        title_KP = get_key_phrase_frequency(title)
        title_KP = get_key_phrase_frequency(sentence)

       # score = cosine_similarity (self.title ,sentence) * 
        pass



#### Centroid  vector Feature
    def get_tfidf_centroid_vector(self):
        """compute the the bag of words model 
        Parameters:
        sentences (list): list of string sentences

        Returns:
        2D numpy_array : 2D numpy array (each row coreesponds to the sentence vecotr , each column correspond to a word)
        1D numpy_array : 1D Centroid Vector

        """

        tfidf = TfidfVectorizer(norm=None, use_idf = True ,sublinear_tf=False, smooth_idf=False).fit(self.sentences)
        tfidf_array = tfidf.transform(self.sentences).toarray()
        

        centroid_vector = tfidf_array.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())
        for i in range(centroid_vector.shape[0]):
            if centroid_vector[i] <= self.topic_threshold:
                centroid_vector[i] = 0
        return tfidf, tfidf_array ,centroid_vector
    
    def get_centroid_score(self,sentence):
        vec_sentence = self.tf_idf.transform([sentence.strip()]).toarray()[0]
        vec_sentence = np.squeeze(vec_sentence)
        return self.cosine_similarity_V1(vec_sentence ,self.centroid_vector)
    
#### Centrailty Feature 
    def get_tfidf_matrix(self):
        return self.tf_idf.transform(self.sentences)
    
    def get_pairwise_cosine_similarity_matrix(self):
        return cosine_similarity(self.tf_idf_matrix,self.tf_idf_matrix)
    
    def get_similarity_degree_dic(self):
        sim_dict = { sent_index : len(np.where(self.cosine_similarity_matrix[sent_index] > self.similarity_threshold)[0] - 1)                                    for sent_index in range(len(self.sentences))}
        return sim_dict
            
 
    def get_max_similarity_degree(self):
        return max(self.similartiy_degree_dic.values())
        
    
    def get_centrality_score(self,sentence):
        sent_idx = self.sent2idx[sentence]
        return self.similartiy_degree_dic[sent_idx] / self.max_similarity_degree

        
#### Cue Phrases Feature 
    def cue_phrases_score(self,sentence):
        cue_phrases = ["الاهم" , "بالتحديد" , "كنتيجة" , "الافضل" ,"الأهم","الأفضل"]
        number_of_sentence_cue_phrases = 0
        for cue in cue_phrases : 
            for word in sentence :
                if cue.strip() == word.strip() : 
                    number_of_sentence_cue_phrases+=1
                

        number_of_document_list_cue_phrases = 0
        for s in self.original_sentences :
            for cue in cue_phrases :
                for word in s :
                    if cue == word : 
                        number_of_document_list_cue_phrases+=1
                        
        return number_of_sentence_cue_phrases / number_of_document_list_cue_phrases if number_of_document_list_cue_phrases else 0
    
    
#### Strong phrases Feature 
    def strong_words_score(self,sentence):
        strong_words = ["وثّق","أكّد" , "أكد" , "أسهم"]
        number_of_strong_words = 0
        for strong in strong_words : 
            for word in sentence :
                if strong.strip() == word.strip() : 
                    number_of_strong_words+=1
                

        number_of_document_list_strong_words = 0
        for s in self.original_sentences :
            for strong in strong_words :
                for word in s :
                    if strong.strip() == word.strip() : 
                        number_of_document_list_strong_words+=1
                        
        return number_of_strong_words / number_of_document_list_strong_words if number_of_document_list_strong_words else 0
    
    
    
    def getLimit(self, limit, total_num_sentences):
        return ( limit * total_num_sentences ) / 100



    def summarize(self,max_legnth) :
        features = [self.get_key_phrase_score ,self.sentence_location_score,self.get_centroid_score,
                    self.get_centrality_score ,self.sentence_length_score ,self.cue_phrases_score ,
                    self.strong_words_score]
        lst = []
        sentence_scores = []
        max_legnth_summary = len(self.golden_summary) if len(self.golden_summary) else max_legnth
        for index,sentence in enumerate(self.sentences) :
            total_score = 0
            for feature in features :
                score = feature(sentence)
                total_score += score
            sentence_scores.append((index,total_score))


        ordered_list = sorted(sentence_scores,key =  lambda x : x[1] ,reverse = True)
        summary = ordered_list[:max_legnth_summary]

        last_summary = sorted(summary,key =  lambda x : x[0])
        sum_list = [self.original_sentences[x] for (x,y) in last_summary]
        text_list = ".".join(sum_list)
        return text_list

        
    
        


