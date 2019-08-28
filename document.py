from helper import Helper
import numpy as np
import math
import yake
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from collections import defaultdict


class Doc():

    def __init__(self,original_text ,preprocessed_text,golden_summary ,five_summaries ,title = "",sentences,preprocessed_sentences,tokenized_word_sentences) :
        self.original_text = original_text
        self.preprocessed_text = preprocessed_text
        self.golden_summary = golden_summary
        self.five_summaries = five_summaries
        self.title = title 
        self.paragraphs = []
        self.sentences = sentences
        self.preprocessed_sentences = preprocessed_sentences
        self.tokenized_word_sentences = tokenized_word_sentences
        self.number_vocab = 0
        self.bow = []
        self.key_phrases = [] #list of tuples
        self.key_phrase_frequency = { }# dict
        self.key_phrase_length = {}
        self.key_phrase_proper_name = {}




        
    def get_doc_key_phrase(self,text):
        max_ngram_size = 1 # specifying parameters
        custom_kwextractor = yake.KeywordExtractor(lan="ar", n = max_ngram_size, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=15, features=None)
        keywords = custom_kwextractor.extract_keywords(text)
        self.key_phrases = keywords
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
        self.key_phrase_frequency = key_phrase_frequency
        return key_phrase_frequency


    def get_key_phrase_length(self) :
        """it indicates how many times the keyphrase appeared in the sentence

            Parameters:
             

            Returns:
            list:lists where each element is the square root of keyphrase 

        """
        key_phrase_length = { kp[0] : math.sqrt(len(kp[0])) for kp in self.key_phrases}
        self.key_phrase_length =  key_phrase_length
        return key_phrase_length


    def get_key_phrase_proper_name(self):
        key_phrase_proper_name = {kp[0] : 1 for kp in self.key_phrases}
        self.key_phrase_proper_name =  key_phrase_proper_name
        return key_phrase_length


    def get_key_phrase_score(self,sentence):
         """compute the keyphrase score for input sentence 

            Parameters:
            sentence (string): the sentence string 

            Returns:
            dict:dict where each element is {key_phrase : key_phrase_frequency}

        """

        total_key_phrase_score = 0
        for kp[0] in self.key_phrases :
            if kp[0] in sentence :
                total_key_phrase_score += self.key_phrase_frequency[kp[0]] *  dictself.key_phrase_length[kp[0]] * key_phrase_proper_name[kp[0]]
        return total_key_phrase_score


    def sentence_location_score(self,sentence):
         """compute the sentence_location_scorerase score for input sentence 

            Parameters:
            sentence (string): the sentence string 

            Returns:
            float : the sentence_location_score score 

        """
        para_list = [p.split[('.') for p in (self.paragraphs]
        score = 0
        for paragrpah_index,list_para in enumerate(para_list) :
            for sent_index , sent in enumerate(list_para):
                if sentence == sent : 
                    if sent_index  == 0 :
                        if paragrpah_index == 0 : score = 3 
                        elif paragrpah_index == len(self.paragraphs) - 1 : score = 2
                        else : score = 1
                    
                    elif paragrpah_index == 0  or  paragrpah_index == len(self.paragraphs) - 1 : score = 1 / math.sqrt(sent_index)
                    else : score = score = 1 / (math.sqrt(sent_index + (paragrpah_index * paragrpah_index))
        return score 

    def cosine_similarity(self,sentence):
         """compute the cosine similarity  between sentence and to

            Parameters:
            sentence (string): the sentence string 

            Returns:
            float : the sentence_location_score score 

        """
    

    def similarity_title_score(self,sentence):
         """compute the sentence_location_scorerase score for input sentence 

            Parameters:
            sentence (string): the sentence string 

            Returns:
            float : the sentence_location_score score 

        """
        title_KP = get_key_phrase_frequency(title)
        title_KP = get_key_phrase_frequency(sentence)

        score = cosine_similarity (self.title ,sentence) * 






    










    

    





  



    def get_bow(self, sentences):
        vectorizer = CountVectorizer()
        sent_word_matrix = vectorizer.fit_transform(sentences)

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = transformer.fit_transform(sent_word_matrix)
        tfidf = tfidf.toarray()

        centroid_vector = tfidf.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())
        for i in range(centroid_vector.shape[0]):
            if centroid_vector[i] <= self.topic_threshold:
                centroid_vector[i] = 0
        return tfidf, centroid_vector


    def similarity(self, v1, v2):
    score = 0.0
    if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
        score = ((1 - cosine(v1, v2)) + 1) / 2
    return score


    def get_topic_idf(self, sentences):
        vectorizer = CountVectorizer()
        sent_word_matrix = vectorizer.fit_transform(sentences)

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = transformer.fit_transform(sent_word_matrix)
        tfidf = tfidf.toarray()

        centroid_vector = tfidf.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())

        feature_names = vectorizer.get_feature_names()
        word_list = []
        for i in range(centroid_vector.shape[0]):
            if centroid_vector[i] > 0.3:
                word_list.append(feature_names[i])

        return word_list




    def summarize(self, text, limit):
        raw_sentences = self.help.getArticleSentences(text)
        clean_sentences = self.help.getCleanSentences(raw_sentences)
        centroid_words = self.get_topic_idf(clean_sentences)
        self.word_vectors_cache(clean_sentences)
        centroid_vector = self.compose_vectors(centroid_words)

        sentences_scores = []
        for i in range(len(clean_sentences)):
            words = clean_sentences[i].split()
            sentence_vector = self.compose_vectors(words)
            score = self.help.similarity(sentence_vector, centroid_vector)
            sentences_scores.append((i, raw_sentences[i], score, sentence_vector))

        sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)
        count = 0
        sentences_summary = []
        for s in sentence_scores_sort:
            if s[0] == 0:
                sentences_summary.append(s)
                count += 1
                sentence_scores_sort.remove(s)
                break

        for s in sentence_scores_sort:
            if count > limit:
                break
            include = True
            for ps in sentences_summary:
                sim = self.help.similarity(s[3], ps[3])
                if sim > self.sim_threshold:
                    include = False
            if include:
                sentences_summary.append(s)
                count += 1

        sentences_summary = sorted(sentences_summary, key=lambda el: el[0], reverse=False)
        summary = " ".join([s[1] for s in sentences_summary])

        return summary
        


