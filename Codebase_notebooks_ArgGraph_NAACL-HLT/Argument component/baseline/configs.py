#Defined paths to be used in the modules
path_To_allrows_dataset = '/home/ayan/legal_data/allrows_dataset/'
path_To_count_csv_dataset = '/home/ayan/legal_data/'
path_To_headnote = '/home/ayan/legal_data/judgement_headnoteProcessed_final/'
path_To_text = '/home/ayan/legal_data/judgement_textProcessed_final/'
path_To_headnote_vector ='/home/ayan/legal_data/judgement_headnote_sentencevectors_final/'
path_To_text_vector = '/home/ayan/legal_data/judgement_text_sentencevectors_final/'
path_To_headnote_extractive='/home/ayan/legal_data/judgement_headnote_extractive/'
path_To_arguments_annotations = '/home/ayan/legal_data/brat-project-final/'
path_To_arguments_annfiles ='/home/ayan/legal_data/brat-project-ann/'
path_To_trainning_files ='/home/ayan/legal_data/training_set_argumentcorpus/'
path_To_Summary ='/home/ayan/legal_data/judgement_summaryPart_final/'
path_To_summary_extractive='/home/ayan/legal_data/judgement_summary_extractive/'
path_To_headnote_extractive_lebelled = '/home/ayan/legal_data/judgement_headnote_labelled_extractive/'
path_To_header_text= '/home/ayan/legal_data/judgement_header_text/'
path_To_sts_benchmark = '/home/ayan/legal_data/stsbenchmark/'
path_To_argument_csv = '/home/ayan_chandra/vaartani_ayan/baseline_AC/'
path_To_argument_corpus = '/home/ayan_chandra/vaartani_ayan/baseline_AC/ArgumentAnnotatedEssays-2.0//brat-project-final/'

# word-embedding hyper-parameters for word2vec
class word2vecembeddig_config(object):

    def __init__(self):
        self.size= 300
        self.mincount=1
        self.epochs =125
        self.window =12


pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

tense_verb_family = {
    'verb_present': ['VB','VBG','VBP','VBZ'],
    'verb_past' : ['VBD','VBN']
}

adjective_family = {
    'adjective_superlative' : ['JJS'],
    'adjective_comparative' : ['JJR']
}

class siamese_config(object):
    def __init__(self):
        pass

