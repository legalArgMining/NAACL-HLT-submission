import os.path
import string
import glob
import pandas as pd
import textblob
import nltk
import gensim
import numpy
import sys
Project_path = ""
sys.path.append(Project_path)
from HeadnoteGeneration import configs
from HeadnoteGeneration.Embedding import word_embedding_model_word2vec,sentence_embedding_models
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from bllipparser import RerankingParser
from nltk.data import find
from nltk.tree import Tree
from transformers import BertTokenizer, BertModel
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import  numpy
from simpletransformers.classification import ClassificationModel


'''
def read_data(readFrom):
    text =[]
    label = []
    for filename in glob.glob(os.path.join(readFrom, '*.txt')):
        with open(filename,'r') as f:
            lines = f.readlines()
            for line in lines:
                array_line = line.split('XXXXX')
                #print(array_line)
                if len(array_line) > 1 :
                    text.append(" ".join(array_line[0].split()))
                    label.append(array_line[1].strip('\n'))

        f.close()

    # create a dataframe using texts and lables

    trainDF = pd.DataFrame()
    trainDF['text'] = text
    trainDF['label'] = label
    return trainDF
'''
def read_data(readFromFile):
    import csv
    dataset = pd.DataFrame(columns = ['text', 'label'])
    with open(readFromFile) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        count = 0
        for line in csvreader:
            text = line[1]
            print(text)
            label = line[2]
            print(label)
            if count !=0 :
                dataset = dataset.append({'text': text, 'label': label}, ignore_index=True)
            count += 1
    return dataset


def check_verb_tense(x, flag, tense_verb_family):
    count = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in tense_verb_family[flag]:
                count += 1
    except:
        pass

    return count

def check_adjective_type(x,flag, adjective_family):
    count = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in adjective_family[flag]:
                count += 1
    except:
        pass

    return count


def is_true_val(val):
    if val > 0:
        return 1
    else:
        return 0

def check_pos_tag(x, flag, pos_family):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

def check_first_person_indicators(text):
    indicator_words = ["I", "me", "myself", "my", "mine"]
    flag = False
    for indicator in indicator_words:
        if indicator.lower() in text.lower():
            flag = True

    if flag == True:
        return 1
    else:
        return 0

def if_modal_verb_present(text):
    modals = ["can", "could", "might", "must", "will", "would", "should"]
    flag = False
    for modal in modals:
        if (modal.lower() in text.lower()) or ("may" in text):
            flag = True

    if flag == True:
        return 1
    else:
        return 0

def check_forward_indicators(text):
    indicators = ["As a result", "As the consequence", "Because", "Clearly", "Consequently", "Considering this", "Furthermore", "Hence", "leading to the consequence", "So","taking account", "That is why", "The reason is that", "Therefore", "This means that", "This shows that", "This will result", "thus","Thus,"]
    flag = False
    for indicator in indicators:
        if indicator.lower() in text.lower():
            flag = True

    if flag == True:
        return 1
    else:
        return 0


def check_backward_indicators(text):
    indicators = ["Additionally", "As a matter of fact", "because", "Besides", "due to", "Finally", "First of all", "Firstly", "for example","For instance", "Furthermore", "In addition", "In addition to this", "In the first place", "due to the fact that", "It should also be noted", "Moreover", "On one hand", "On the other hand", "One of the main reasons", "Secondly", "Similarly", "since", "So", "The reason", "To begin with", "To offer an instance", "What is more"]
    flag = False
    for indicator in indicators:
        if indicator.lower() in text.lower():
            flag = True

    if flag == True:
        return 1
    else:
        return 0


def check_thesis_indicators(text):
    indicators = ["All in all", "All things considered", "As far as I am concerned", "Based on somereasons","considering both the previous fact", "Finally","For the reasons mentioned above", "From explanation above", "From this point ofview", "I agree that", "I agree with", "I agree with the statement that", "I believe that", "I do not agree with this statement","I firmly believe that", "I highly advocate that", "I highly recommend", "I strongly believe that", "I think that", "I totally agree", "I totally agree to this opinion", "I would have to argue that", "I would reaffirm my position that", "In conclusion", "in my opinion", "In my personal point of view", "in my point of view","In summary", "it can be said that", "it is clear that", "it seems to me that", "my deep conviction", "My sentiments", "Overall", "Personally", "the above explanations", "This, however", "To conclude", "To my way of thinking", "To sum up", "Ultimately"]
    flag = False
    for indicator in indicators:
        if indicator.lower() in text.lower():
            flag = True

    if flag == True:
        return 1
    else:
        return 0

def get_depth_parse_tree(text, parser):
    parser_list = parser.parse(text)
    depth = 0
    depth = parser_list.get_reranker_best().ptb_parse.as_nltk_tree().height()
    return depth


def check_rebuttal_indicators(text):
    indicators = ["Admittedly", "although","besides", "but", "Even though", "However", "Otherwise"]
    flag = False
    for indicator in indicators:
        if indicator.lower() in text.lower():
            flag = True

    if flag == True:
        return 1
    else:
        return 0

def check_explicit_connectives(text):
    explicit_connectives = ["because","when","since","although", "and","or","nor","however","otherwise","then","as a result","for example"]
    flag = False
    for connective in explicit_connectives:
        if connective.lower() in text.lower():
            flag = True

    if flag == True:
        return 1
    else:
        return 0

def check_conjoined_connectives(text):
    conjoined_connectives = ["when and if", "if and when"]
    flag = False
    for connective in conjoined_connectives:
        if connective.lower() in text.lower():
            flag = True

    if flag == True:
        return 1
    else:
        return 0

def check_parallel_connectives(text):
    parallel_connectives = ["on the one hand","on the other hand"]
    flag = False
    for connective in parallel_connectives:
        if connective.lower() in text.lower():
            flag = True

    if "either" in text.lower() and "or" in text.lower():
        flag = True

    if flag == True:
        return 1
    else:
        return 0

def bert_average_embedding(text,tokenizer,model):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    np_embedding = last_hidden_states.detach().numpy()
    np_sentence_embed = np_embedding[0]
    #np_rows , np_columns = np_sentence_embed.shape
    np_word_embedding_average = np_sentence_embed.mean(axis=0)
    word_embed_list = np_word_embedding_average.tolist()
    return word_embed_list

def bert_finetuned_sentence_embedding(text, model):
    predictions, model_outputs, all_embedding_outputs, all_layer_hidden_states  = model.predict([text])
    embedding = all_embedding_outputs[0][0]
    embedding_list = embedding.tolist()
    return embedding_list


def extract_features(trainDF,parser,model):
    #extract_features(trainDF,parser,tokenizer,model)
    #extract_features(trainDF,parser)
    '''
    1.Word Count of the documents – total number of words in the documents
    2.Character Count of the documents – total number of characters in the documents
    3.Average Word Density of the documents – average length of the words used in the documents
    4.Puncutation Count in the Complete Essay – total number of punctuation marks in the documents
    5.Upper Case Count in the Complete Essay – total number of upper count words in the documents
    6.Title Word Count in the Complete Essay – total number of proper case (title) words in the documents
    7.Frequency distribution of Part of Speech Tags:
        a.Noun Count
        b.Verb Count
        c.Adjective Count
        d.Adverb Count
        e.Pronoun Count
    '''
    #trainDF = pd.DataFrame()
    #trainDF['text'] = text

    #structural features
    trainDF['char_count'] = trainDF['text'].apply(len)
    trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
    trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count'] + 1)
    trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
    trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
    trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

    #syntactic features

    #model_dir = find('models/bllip_wsj_no_aux').path
    #parser = RerankingParser.from_unified_model_dir(model_dir)

    pos_family = configs.pos_family
    tense_verb_family = configs.tense_verb_family
    adjective_family = configs.adjective_family

    trainDF['noun_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'noun', pos_family=pos_family))
    trainDF['verb_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'verb',pos_family=pos_family))
    trainDF['adj_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adj',pos_family=pos_family))
    trainDF['adv_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adv',pos_family=pos_family))
    trainDF['pron_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'pron',pos_family=pos_family))
    trainDF['noun_density'] = trainDF['noun_count'] / (trainDF['word_count'] + 1)
    trainDF['verb_density'] = trainDF['verb_count'] / (trainDF['word_count'] + 1)
    trainDF['adj_density'] = trainDF['adj_count'] / (trainDF['word_count'] + 1)
    trainDF['adv_density'] = trainDF['adv_count'] / (trainDF['word_count'] + 1)
    trainDF['pron_density'] = trainDF['pron_count'] / (trainDF['word_count'] + 1)

    trainDF['is_verb_past_tense'] = trainDF['text'].apply(lambda x: is_true_val(check_verb_tense(x, 'verb_past', tense_verb_family=tense_verb_family)))
    trainDF['is_verb_present_tense'] = trainDF['text'].apply(lambda x: is_true_val(check_verb_tense(x, 'verb_present', tense_verb_family=tense_verb_family)))
    trainDF['is_modal_verb_present'] = trainDF['text'].apply(lambda x: if_modal_verb_present(x))
    trainDF['is_adjective_superlative'] = trainDF['text'].apply(lambda x: is_true_val(check_adjective_type(x, 'adjective_superlative', adjective_family=adjective_family)))
    trainDF['is_adjective_comparative'] = trainDF['text'].apply(lambda x: is_true_val(check_adjective_type(x, 'adjective_comparative', adjective_family=adjective_family)))
    trainDF['depth_parse_tree'] = trainDF['text'].apply(lambda x: get_depth_parse_tree(x, parser))


    #indicator features
    trainDF['first_person_indicators'] = trainDF['text'].apply(lambda x: check_first_person_indicators(x))
    trainDF['forward_indicators'] = trainDF['text'].apply(lambda x: check_forward_indicators(x))
    trainDF['backward_indicators'] = trainDF['text'].apply(lambda x: check_backward_indicators(x))
    trainDF['thesis_indicators'] = trainDF['text'].apply(lambda x: check_thesis_indicators(x))
    trainDF['rebuttal_indicators'] = trainDF['text'].apply(lambda x: check_rebuttal_indicators(x))

    #discourse features
    trainDF['explicit_connectives'] = trainDF['text'].apply(lambda x: check_explicit_connectives(x))
    trainDF['conjoined_connectives'] = trainDF['text'].apply(lambda x: check_conjoined_connectives(x))
    trainDF['parallel_connectives'] = trainDF['text'].apply(lambda x: check_parallel_connectives(x))


    '''
    Sentence-embedding for each sentence
    '''
    trainDF['sentence_embedding'] = trainDF['text'].apply(lambda x: sentence_embedding_models.sent2vec(x,'word2vec',embedding_filename='/home/baseline_AC/lth-legaltextminingsystem/project/HeadnoteGeneration/argumentcorpus_law_two_set_wordembeddingv1.bin').tolist())
    
    #trainDF['sentence_embedding'] = trainDF['text'].apply(lambda x: bert_finetuned_sentence_embedding(x,model) )
    #trainDF['sentence_embedding'] = trainDF['text'].apply(lambda x: bert_average_embedding(x,tokenizer,model) )
    #print(trainDF['sentence_embedding'])
    #trainDF['sentence_embedding'] = trainDF['word_density'].apply(lambda x: [x,x*x,x*x*x])
    return trainDF

def make_feature_vector(trainDF):
    fin_list=[]
    for index,row in trainDF.iterrows():
        fin_list.append([row['char_count'], row['word_count'],row['word_density'],row['punctuation_count'],row['title_word_count'],row['upper_case_word_count'],row['noun_count'], row['verb_count'], row['adj_count'], row['adv_count'], row['pron_count'], row['noun_density'], row['verb_density'], row['adj_density'], row['adv_density'], row['pron_density'], row['is_verb_past_tense'], row['is_verb_present_tense'], row['is_modal_verb_present'], row['is_adjective_superlative'], row['is_adjective_comparative'], row['depth_parse_tree'], row['first_person_indicators'], row['forward_indicators'], row['backward_indicators'], row['thesis_indicators'], row['rebuttal_indicators'], row['explicit_connectives'], row['conjoined_connectives'], row['parallel_connectives']]+row['sentence_embedding'])
        '''
        list.append(row['noun_count'])
        list.append(row['verb_count'])
        list.append(row['adj_count'])
        list.append(row['adv_count'])
        list.append(row['pron_count'])
        '''

    print(fin_list)
    return fin_list

def main():
    dataset = read_data(configs.path_To_trainning_files)

    # split the dataset into training and validation datasets
    #train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size = 0.20)

    #trainDF_x = extract_features(train_x)
    #validDF_x = extract_features(valid_x)

    # label encode the target variable
    #encoder = preprocessing.LabelEncoder()
    #train_y = encoder.fit_transform(train_y)
    #valid_y = encoder.fit_transform(valid_y)




if __name__ == '__main__':
    main()
