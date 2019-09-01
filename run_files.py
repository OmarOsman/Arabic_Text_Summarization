import os, pickle, re
import document
import preprocess
import argparse
import pdb


def get_summary(input_text):
    pr = preprocess.Preprocess()
    original_text = input_text
    preprocessed_text = pr.get_clean_article(input_text)
    sentences = pr.get_article_sentences(preprocessed_text)
    original_sentences = pr.get_article_sentences(input_text)
    paragraphs = pr.get_cleaned_article_paragraphes(preprocessed_text)
    para_sent_list = pr.get_para_sentences(paragraphs)
    tokenized_word_sentences = pr.get_tokenized_word_sentences(sentences)
    
    doc = document.Doc(
    original_text = original_text ,  original_sentences = original_sentences ,
    preprocessed_text = preprocessed_text.replace('ppp',""),
    sentences = sentences,
    paragraphs = paragraphs ,para_sent_list = para_sent_list ,tokenized_word_sentences = tokenized_word_sentences)
    
    summary = doc.summarize()
    return summary

def run():
    input_dir = "input"
    output_dir = "output"
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,help="path to input text document")
    #ap.add_argument("-o", "--output", required=True,help="path to output Summarized Document")
    args = vars(ap.parse_args())
    
    
    input_path = os.path.join(input_dir,args['input'])
    output_path = os.path.join(output_dir,args['input'])
    
    pr = preprocess.Preprocess()    
    input_text = pr.get_article_content(input_path)
    summary = get_summary(input_text)
    
    #pdb.set_trace()
    with open(output_path,'w' ,encoding = "utf-8") as f: f.write(summary)


if __name__ == "__main__":
    run()
    