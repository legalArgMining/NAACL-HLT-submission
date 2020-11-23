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
from sklearn.externals import joblib

#import tensorflow as tf
from HeadnoteGeneration.Embedding import word_embedding_model_word2vec,sentence_embedding_models
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
import HeadnoteGeneration.Classifiers.feature_extraction as fex
from sklearn.metrics import classification_report, confusion_matrix
import ssl
from bllipparser import RerankingParser
from nltk.data import find
from nltk.tree import Tree
#import mxnet as mx
#from bert_embedding import BertEmbedding
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizer, XLNetTokenizer, XLNetModel
#import torch
#import torch.optim as optim
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.autograd import Variable
#from torch.utils.data import Dataset, DataLoader
from simpletransformers.classification import ClassificationModel


'''
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('averaged_perceptron_tagger')
'''

def train_svm(kernel_type):
    trainDF = fex.read_data('/home/baseline_AC/train_AC_combined_models.csv')
    testDF = fex.read_data('/home/baseline_AC/test_AC_combined_models_duplicate_included.csv')
    model_dir = find('models/bllip_wsj_no_aux').path
    parser = RerankingParser.from_unified_model_dir(model_dir)
    
    #ctx = mx.gpu(0)
    #bert = BertEmbedding(ctx=ctx)
    
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #model = BertModel.from_pretrained('bert-base-uncased')
   
    #tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    #model = RobertaModel.from_pretrained('roberta-base')

    #tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    #model = XLNetModel.from_pretrained('xlnet-base-cased')
    
    model_path = '/home/AC_models_Argument_corpus/roberta/'
    model = ClassificationModel('roberta',model_path, num_labels=4,args ={"config":{"output_hidden_states": True}})

    #trainDF_x = fex.extract_features(trainDF,parser)
    #trainDF_x = fex.extract_features(trainDF, parser, tokenizer, model)
    trainDF_x = fex.extract_features(trainDF,parser,model)
    feature_train_x = fex.make_feature_vector(trainDF_x)

    #testDF_x = fex.extract_features(testDF,parser)
    #testDF_x = fex.extract_features(testDF, parser, tokenizer, model)
    testDF_x = fex.extract_features(testDF,parser,model)
    feature_test_x = fex.make_feature_vector(testDF_x)

    # label encode the target variable
    train_y = []
    test_y = []

    for index,row in trainDF.iterrows():
        if row['label'] == 'Claim':
            train_y.append(1)
        elif row['label'] == 'Premise':
            train_y.append(0)
        elif row['label'] == 'MajorClaim':
            train_y.append(3)
        else:
            train_y.append(2)

    for index,row in testDF.iterrows():
        if row['label'] == 'Claim':
            test_y.append(1)
        elif row['label'] == 'Premise':
            test_y.append(0)
        elif row['label'] == 'MajorClaim':
            test_y.append(3)
        else:
            test_y.append(2)

    #train_y = encoder.fit_transform(train_y)

    svmclassifier = svm.SVC(kernel=kernel_type)
    svmclassifier.fit(feature_train_x, train_y)
    filename = 'finalized_model_linear.sav'
    #joblib.dump(svmclassifier, filename)

    y_pred = svmclassifier.predict(feature_test_x)
    print("argument corpus results for test:")
    print(confusion_matrix(test_y, y_pred))
    print(classification_report(test_y, y_pred))

    print("two law set results for test:")
    testlawDF = fex.read_data('/home/baseline_AC/test_judgement_AC_combined_models_duplicate_included.csv')
    #testlawDF_x = fex.extract_features(testlawDF,parser)
    #testlawDF_x = fex.extract_features(testlawDF, parser, tokenizer, model)
    testlawDF_x = fex.extract_features(testlawDF, parser, model)
    feature_test_law = fex.make_feature_vector(testlawDF_x)

    test_y_law = []
    for index,row in testlawDF.iterrows():
        if row['label'] == 'Claim':
            test_y_law.append(1)
        elif row['label'] == 'Premise':
            test_y_law.append(0)
        elif row['label'] == 'MajorClaim':
            test_y_law.append(3)
        else:
            test_y_law.append(2)

    y_pred_2 = svmclassifier.predict(feature_test_law)
    print(confusion_matrix(test_y_law, y_pred_2))
    print(classification_report(test_y_law, y_pred_2))
    filename = 'finalized_model_svm_roberta_finetuned_embedding.sav'
    joblib.dump(svmclassifier, filename)



    #sess = tf.Session()

'''
def write_the_test_result(readFrom,writeTo,model_name):
    k=1
    for filename in glob.glob(os.path.join(readFrom, '*.txt')):
        f1 = open(filename,'r')
        f2 = open(writeTo+str(k)+'.txt','w')
        text = f1.read()
        sentences_l = text.split('XXXX')
        sentences = sentences_l[0:len(sentences_l) - 1]
        validDF_x = fex.extract_features(sentences)
        feature_valid_x = fex.make_feature_vector(validDF_x)

        svmclassifier = joblib.load(model_name)
        y_pred = svmclassifier.predict(feature_valid_x)
        #print(svmclassifier.classes_)


        for i in range(0,len(feature_valid_x)):
            if y_pred[i] == 0:
                label_tag = 'Claim'
            elif y_pred[i] == 1:
                label_tag = 'Majorclaim'
            elif y_pred[i] == 2:
                label_tag = 'None'
            else:
                label_tag = 'Premise'
            print(sentences[i]+'XXXXXX'+label_tag+'\n')
            f2.write(sentences[i]+'XXXXXX'+label_tag+'\n')
        f1.close()
        f2.close()
        k += 1
'''

def main():
    train_svm(kernel_type='linear')
    #model_name = 'finalized_model.sav'
    #readFromFile1 = configs.path_To_headnote_extractive
    #wrtieTo = configs.path_To_headnote_extractive_lebelled
    #write_the_test_result(readFrom=readFrom,writeTo=wrtieTo,model_name=model_name)


if __name__ == '__main__':
    main()
