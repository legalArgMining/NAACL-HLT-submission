import os.path
import nltk
import gensim
import numpy as np
from HeadnoteGeneration import configs
#import mxnet as mx
#from bert_embedding import BertEmbedding


#sentence embedding using word2vec word embeddings
#1. average of the sum of the word embeddings
def sent2vec(sentence,word_embedding_type,embedding_filename):
    parameter_config = configs.word2vecembeddig_config()
    #parameter_config.size=200
    embedding = np.zeros(parameter_config.size)
    if word_embedding_type == 'fasttext':
        model = gensim.models.FastText.load(embedding_filename)
    else:
        model = gensim.models.Word2Vec.load(embedding_filename)
    words= nltk.word_tokenize(sentence.lower())
    leng = len(words)
    if len(words) > 1:
        for word in words:
            if word not in model:
                print(word)
                leng -= 1
            else:
                embedding = np.add(model[word], embedding)
        if leng > 0:
            embedding = np.divide(embedding, leng)
    return embedding


def main():
    sentence="If the decree abrakadabra is merely declaratory and no remedy is provided in case of non-compliance, the proper remedy is to file a suit"
    embedding=sent2vec(sentence,word_embedding_type='word2vec',embedding_filename='/Users/ubuntu/pointsoflawextractor/completecorpuswordembeddingv1.bin')
    print(embedding)
    #embedding=sent2vec(sentence,word_embedding_type='fasttext', embedding_filename='legalembedding_fasttextv1.bin')
    #print(embedding)

if __name__ == '__main__':
    main()


