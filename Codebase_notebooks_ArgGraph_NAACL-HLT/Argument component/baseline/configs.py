#Defined paths to be used in the modules
path_To_allrows_dataset = ''
path_To_count_csv_dataset = ''
path_To_headnote = './judgement_headnoteProcessed_final/'
path_To_text = './judgement_textProcessed_final/'
path_To_headnote_vector ='./judgement_headnote_sentencevectors_final/'
path_To_text_vector = './judgement_text_sentencevectors_final/'
path_To_headnote_extractive='./judgement_headnote_extractive/'
path_To_arguments_annotations = './brat-project-final/'
path_To_arguments_annfiles ='./brat-project-ann/'
path_To_trainning_files ='./training_set_argumentcorpus/'
path_To_Summary ='./judgement_summaryPart_final/'
path_To_summary_extractive='./judgement_summary_extractive/'
path_To_headnote_extractive_lebelled = './judgement_headnote_labelled_extractive/'
path_To_header_text= './judgement_header_text/'
path_To_sts_benchmark = './stsbenchmark/'
path_To_argument_csv = './baseline_AC/'
path_To_argument_corpus = './baseline_AC/ArgumentAnnotatedEssays-2.0//brat-project-final/'

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

