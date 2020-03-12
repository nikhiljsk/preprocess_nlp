import gensim
import numpy as np
from itertools import chain
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



# ------------------------ Count Vectorizer ------------------------
def custom_tokenizer(para):
    words = list()
    for sent in para.split(' . '):
        words.append(sent.split())
    return list(chain(*words))


def count_vectorizer(sentences, params={}):
    default_params = {'strip_accents': None, 
                    'lowercase': True,
                    'preprocessor': None, 
                    'tokenizer': None, 
                    'stop_words': None, 
                    'ngram_range': (1, 1), 
                    'analyzer': 'word', 
                    'max_df': 1.0, 
                    'min_df': 1, 
                    'max_features': None, 
                    'vocabulary': None}
    default_params.update(params)
    
    cv = CountVectorizer(sentences, **default_params)
    cv_trans_sent = cv.fit_transform(sentences)
    
    return cv, cv_trans_sent


# ------------------------ TF-IDF ------------------------
def tfidf_vectorizer(sentences, params={}):
    default_params = {'smooth_idf': True,
                    'use_idf': True,
                    'strip_accents': None, 
                    'lowercase': True,
                    'preprocessor': None, 
                    'tokenizer': None, 
                    'stop_words': None, 
                    'ngram_range': (1, 1), 
                    'analyzer': 'word', 
                    'max_df': 1.0, 
                    'min_df': 1, 
                    'max_features': None, 
                    'vocabulary': None}
    default_params.update(params)
    
    tf = TfidfVectorizer(**default_params)
    tf_trans_sent = tf.fit_transform(sentences)
    
    return tf, tf_trans_sent


def top_words_tfidf(tf_obj, doc, topn=20):  
    # Function code credits: https://kavita-ganesan.com/extracting-keywords-from-text-tfidf/
    tf_idf_vector = tf_obj.transform(doc)
    tuples = zip(tf_idf_vector.tocoo().col, tf_idf_vector.tocoo().data)
    sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
    
    feature_names = tf_obj.get_feature_names()
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]

    return results


# ------------------------ Word2Vec ------------------------
def load_word2vec(path=None):
    try:
        model = KeyedVectors.load_word2vec_format(path, binary=True)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print("Please download the dataset from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit")
        print("-----!! MODEL NOT LOADED !!-----")
        print('\n\n\nError:\t', e)

        
def train_word2vec(documents, params={}):
    default_params = {'size': 100,
                     'window': 10,
                     'min_count': 1, 
                     'workers': 8}
    default_params.update(params)
    model = gensim.models.Word2Vec (documents, **default_params)
    model.train(documents,total_examples=len(documents),epochs=50)
    
    return model


# ------------------------ GloVe ------------------------
def load_glove(path=None):
    try:
        temp = glove2word2vec(path, path+'.word2vec')
    except Exception as e:
        print("Please download the glove.6B.zip dataset from: https://nlp.stanford.edu/projects/glove/")
        print("-----!! MODEL NOT LOADED !!-----")
        print('\n\n\nError:\t', e)
        return None
    
    model = KeyedVectors.load_word2vec_format(path+'.word2vec', binary=False)
    print("Model loaded successfully!")
    return model


# ------------------------ Word2Vec & GloVe ------------------------
def get_most_similar(model, pos_word, neg_word=None, topn=1):
    return model.wv.most_similar(positive=pos_word, negative=neg_word, topn=topn)


def get_similarity_score(model, w1, w2):
    return model.wv.similarity(w1, w2)


def get_sentence_wise_vector(model, docs):
    # Initialize dictionary with existing vocab
    w2v_words = {}
    for ele in list(model.wv.vocab):
        w2v_words[ele] = 0
    
    sent_vectors = []; # the avg-w2v for each sentence/review is stored in this list
    for sent in docs: # for each review/sentence
        sent_vec = np.zeros(model.wv[list(model.wv.vocab.keys())[0]].shape) # as word vectors are of zero length
        cnt_words =0; # num of words with a valid vector in the sentence/review
        for word in sent: # for each word in a review/sentence
            if word in w2v_words:
                vec = model.wv[word]
                sent_vec += vec
                cnt_words += 1
        if cnt_words != 0:
            sent_vec /= cnt_words
        sent_vectors.append(sent_vec)
    
    return sent_vectors