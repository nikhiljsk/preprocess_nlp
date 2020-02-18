# Preprocess NLP Text

## Framework Description

A simple and fast framework for 
* Preprocessing or Cleaning of text
* Extracting top words or reduction of vocabulary 
* Feature Extraction

Uses parallel execution by leveraging the [multiprocessing library](https://docs.python.org/3.6/library/multiprocessing.html) in Python. This repo holds an easy way to clean text with various defined stages implemented using standardized techniques in Natural Language Processing (NLP). Contains both sequential and parallel ways (For less CPU intensive processes) for preprocessing text with an option of user-defined number of processes. Contains code which can be used to find the top words in the corpus, lets you choose a threshold to consider the words that can stay in the corpus and replaces the others, A simple way to reduce vocabulary size decreasing input vector size for deep learning models and various feature extraction helpers. Refer below for details


### Preprocess/Cleaning Module
Uses [nltk](https://www.nltk.org/) for few of the stages defined below.
Various stages of cleaning include:

| Stage                     | Description                                                                           |
| ------------------------- |:-------------------------------------------------------------------------------------:|
| remove_tags_nonascii      | Remove HTML tags, emails, URLs, non-ascii characters and converts accented characters |
| lower_case                | Converts the text to lower_case                                                       |
| expand_contractions       | Expands the word contractions                                                         |
| remove_punctuation        | Remove punctuation from text, but sentences are seperated by ' . '                    |
| remove_esacape_chars      | Remove escapse characters like \n, \t etc                                             |
| remove_stopwords          | Remove stopwords using nltk python                                                    |
| remove_numbers            | Remove all digits in the text                                                         |
| lemmatize                 | Uses WordNetLemmatizer to lemmatize text                                              |
| stemming                  | Uses SnowballStemmer for stemming of text                                             |
| min_word_len              | Minimum word length to keep in text                                                   |

---
### Reduction of Vocabulary 
Shortlists top words based on the percentage as input. Replaces the words not shortlisted and replaces them efficienctly. Also, supports parallel and sequential processing. 

---

### Feature Extraction Module
Uses [Spacy Pipe](https://spacy.io/usage/processing-pipelines) module to avoid unnecessary parsing to increase speed.
Various stages of feature extraction include:

| Stage                     | Description                                                                           |
| ------------------------- |:-------------------------------------------------------------------------------------:|
| nouns                     | Extract the list of Nouns from the given string                                       |
| verbs                     | Extract the list of Verbs from the given string                                       |
| adjs                      | Extract the list of Adjectives from the given string                                  |
| noun_phrases              | Extract the list of Noun Phrases (Noun chunks) from the given string                  |
| keywords                  | Uses [YAKE](https://github.com/LIAAD/yake) for extracting keywords from text          |
| ner                       | Extracts Person, Location and Organization as named entities                          |
| numbers                   | Extracts all digits in the text                                                       |


## Code - Components

Various Python files and there purpose mentioned here:

* [`preprocess_nlp.py`](https://github.com/nikhiljsk/preprocess_nlp/blob/master/preprocess/preprocess_nlp.py) - Contains functions which are built around existing techniques for preprocessing or cleaning text. Defines both sequential and parallel ways of code execution for preprocessing.
* [`Preprocessing_Example_Notebook.ipynb`](https://github.com/nikhiljsk/preprocess_nlp/blob/master/preprocess/Preprocessing_Example_Notebook.ipynb)    - How-to-use example notebook for preprocessing or cleaning stages
* [`requirements.txt`](https://github.com/nikhiljsk/preprocess_nlp/blob/master/requirements.txt)                        - Required libraries to run the project
* [`vocab_elimination_nlp.py`](https://github.com/nikhiljsk/preprocess_nlp/blob/master/vocab_elimination/vocab_elimination_nlp.py) - Contains functions which are built around existing techniques for shortlisting top words and reducing vocab size
* [`Vocab_Elimination_Example_Notebook.ipynb`](https://github.com/nikhiljsk/preprocess_nlp/blob/master/vocab_elimination/Vocab_Elimination_Example_Notebook.ipynb) - How-to-use example notebook for vocabulary reduction/elimination or replacement.

## How to run

1. pip install -r requirements.txt
2. Import [`preprocess_nlp.py`](https://github.com/nikhiljsk/preprocess_nlp/blob/master/preprocess/preprocess_nlp.py) and use the functions [`preprocess_nlp`](https://github.com/nikhiljsk/preprocess_nlp/blob/master/preprocess/preprocess_nlp.py#L34)(for sequential) and [`asyn_call_preprocess`](https://github.com/nikhiljsk/preprocess_nlp/blob/master/preprocess/preprocess_nlp.py#L149)(for parallel) as defined in notebook
3. Import [`vocab_elimination_nlp.py`](https://github.com/nikhiljsk/preprocess_nlp/blob/master/vocab_elimination/vocab_elimination_nlp.py) and use functions as defined in the notebook [`Vocab_Elimination_Example_Notebook.ipynb`](https://github.com/nikhiljsk/preprocess_nlp/blob/master/vocab_elimination/Vocab_Elimination_Example_Notebook.ipynb)
4. Import [`feature_extraction.py`](https://github.com/nikhiljsk/preprocess_nlp/blob/master/feature_extraction/feature_extraction.py) and use functions as defined in notebook [`Feature_Extraction_Example_Notebook.ipynb`](https://github.com/nikhiljsk/preprocess_nlp/blob/master/feature_extraction/Feature_Extraction_Example_Notebook.ipynb)

---

## Sequential & Parallel Processing

1. Sequential   - Processes records in a sequential order, does not consume a lot of CPU Memory but is slower compared to Parallel processing
2. Parallel     - Can create multiple processes (customizable/user-defined) to preprocess text parallelly, Memory intensive and faster.
<br>

---
### Future Updates
* Vectorization tools like TF-IDF, GloVe, Word2Vec, Bag of Words

*Refer the code for Docstrings and other function related documentation.* 
<br>
Cheers :)
