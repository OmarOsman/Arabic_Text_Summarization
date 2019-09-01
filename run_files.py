import os, pickle, re
import document
import preprocess
import argparse


def get_summary(input_text):
    pr = preprocess.Preprocess()
    original_text = input_text
    preprocessed_text = pr.get_clean_article(original_text)
    sentences = pr.get_article_sentences(preprocessed_text)
    paragraphs = pr.get_cleaned_article_paragraphes(preprocessed_text)
    para_sent_list = pr.get_para_sentences(paragraphs)
    preprocessed_sentences = pr.get_article_sentences(preprocessed_text)
    tokenized_word_sentences = pr.get_tokenized_word_sentences(preprocessed_sentences)
    
    doc = document.Doc(
    original_text = original_text , preprocessed_text = preprocessed_text.replace('ppp',""),
    sentences = sentences,preprocessed_sentences = preprocessed_sentences ,
    paragraphs = paragraphs ,para_sent_list = para_sent_list ,tokenized_word_sentences = tokenized_word_sentences)
    
    summary = doc.summarize()
    return summary

def run():
    input_dir = "input"
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,help="path to input text document")
    ap.add_argument("-o", "--output", required=True,help="path to output Summarized Document")
    args = vars(ap.parse_args())
    
    input_path = os.path.join(test_dir,args.input)
    output_path = os.path.join(test_dir,args.output)
    
    input_text = pr.get_article_content(input_path)
    summary = get_summary(input_text)
    
    
    
    
    
    

if __name__=="main":
    run()
    