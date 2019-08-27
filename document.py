from helper import Helper
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class Doc():

    def __init__(self,original_text ,preprocessed_text,golden_summary ,five_summaries ,title = "",sentences,tokenized_word_sentences) :
        self.original_text = original_text
        self.preprocessed_text = preprocessed_text
        self.golden_summary = golden_summary
        self.five_summaries = five_summaries
        self.title = title 
        self.sentences = sentences
        self.tokenized_word_sentences = tokenized_word_sentences
        self.number_vocab = 
        self.bow = 



        


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
        


