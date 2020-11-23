import os.path
import nltk
import csv
#nltk.download('punkt')
import gensim
import numpy
import sys
Project_path = ""
sys.path.append(Project_path)
from HeadnoteGeneration import configs
import  glob
import re
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from HeadnoteGeneration.utilities import NLPutils as nlp

def clean_text(text):
    return text.replace('[','').replace(']','').replace('\n','').replace('\t','').replace("Para's",'').replace("Parra's",'').replace("parra's",'').replace("para's",'').replace('paras','').replace('parras','').replace('Parra','').replace('Para','').replace('para','').replace('(','').replace(')','')

def split_sentences(text):
    '''
    punkt_param = PunktParameters()
    punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'no','g.a','c.s','ors','anr'])
    sentence_splitter = PunktSentenceTokenizer(punkt_param)
    sentences = sentence_splitter.tokenize(text.lower())
    list =[]
    for sentence in sentences:
        list = list + sentence.split(';')
    return list
    '''
    text = text.lower().replace('dr.','dr ').replace('vs.','vs ').replace('mr.','mr ').replace('mrs.','mrs ').replace('prof.','prof ').replace('no.','number ').replace('ors.','ors ').replace('anr.','anr ').replace('i.e.','ie ').replace('e.s.i.','esi ').replace('inc.','inc ').replace('fir.','fir ').replace('etc.','etc ')
    list =[]
    sentences = re.split(r'[;?!]',text)
    for sentence in sentences:
        list = list + re.split(r'(?<!\.[a-zA-Z])\.(?![a-zA-Z]\.)', sentence)
    return list


def populate_list(path,list):
    for filename in glob.glob(os.path.join(path, '*.txt')):
        print(filename)
        with open(filename,'r') as f :
            text = f.read()
            #sentences = split_sentences(clean_text(text))
            sentences = nlp.cleared_sentences(nlp.split_into_sentences(text))
            #print(sentences)
            for sentence in sentences:
                list_for_words = nltk.word_tokenize(sentence.lower())
                if len(list_for_words) > 1:
                    list.append(list_for_words)
        f.close()
    return list

def populate_two_set_law_list(list):
    sentences = []
    with open('/home/ubuntu/baseline_AC/test_judgement_AC_combined_models_duplicate_included.csv') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        count = 0
        for line in csvreader:
            text = line[0]
            component_label = line[1]
            sentences.append(text)

        for sentence in sentences:
            list_for_words = nltk.word_tokenize(sentence.lower())
            if len(list_for_words) > 1:
                list.append(list_for_words)
    return list

def build_word_embedding(final_list,embedding_filename, hyperparameters):
    size = hyperparameters[0]
    window = hyperparameters[1]
    min_count = hyperparameters[2]
    epochs = hyperparameters[3]

    # train model
    model = gensim.models.Word2Vec(final_list,size=size, window=window, min_count=min_count, workers=10)
    model.train(final_list, total_examples=len(final_list), epochs=epochs)
    # summarize the loaded model
    print(model)
    # summarize vocabulary
    words = list(model.wv.vocab)
    print(words)
    # access vector for one word
    print(model['constitution'])
    # save model
    model.save(embedding_filename)

    return True


def print_word_embedding(embedding_filename):
    # load model
    new_model = gensim.models.Word2Vec.load(embedding_filename)
    print(new_model)


def main():
    parameter_config = configs.word2vecembeddig_config()
    hyperparameters = [parameter_config.size, parameter_config.window, parameter_config.mincount,
                       parameter_config.epochs]
    final_list = []
    #final_list = populate_list(configs.path_To_headnote, final_list)
    #final_list = populate_list(configs.path_To_text, final_list)
    #final_list = populate_list(configs.path_To_Summary, final_list)
    final_list = populate_list(configs.path_To_argument_corpus, final_list)
    final_list = populate_two_set_law_list(final_list)
    print(final_list)

    exit = build_word_embedding(final_list,embedding_filename='argumentcorpus_law_two_set_wordembeddingv1.bin',hyperparameters=hyperparameters)
    if exit:
        print_word_embedding(embedding_filename = 'argumentcorpus_law_two_set_wordembeddingv1.bin')


if __name__ == '__main__':
    main()
