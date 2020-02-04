# Preprocess NLP Text

## Framework Description

A simple framework for preprocessing of text using parallel execution by leveraging the multiprocessing library in Python. Completely written is Python code, this repo holds an easy way to preprocess text with various defined stages implemented using standardized techniques in Natural Language Processing (NLP). Contains both sequential and parallel ways for preprocessing text with an option of user-defined number of processes.

Various stages of preprocessing include:

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

## Code - Components

There is a single python code file which defines both sequential and parallel ways of code execution for preprocessing.

* [`preprocess_nlp.py`](https://github.com/nikhiljsk/preprocess_nlp/blob/master/preprocess_nlp.py)                       - Contains functions which are built around existing techniques for preprocessing text. Defines both sequential and parallel ways of code execution for preprocessing.
* [`Preprocessing_Example_Notebook.ipynb`](https://github.com/nikhiljsk/preprocess_nlp/blob/master/Preprocessing_Example_Notebook.ipynb)    - How-to-use example notebook
* [`requirements.txt`](https://github.com/nikhiljsk/preprocess_nlp/blob/master/requirements.txt)                        - Required libraries to run the project

## How to run

1. pip install -r requirements.txt
2. Import [`preprocess_nlp.py`](https://github.com/nikhiljsk/preprocess_nlp/blob/master/preprocess_nlp.py) and use the functions [`preprocess_nlp`](https://github.com/nikhiljsk/preprocess_nlp/blob/master/preprocess_nlp.py#L34)(for sequential) and [`asyn_call_preprocess`](https://github.com/nikhiljsk/preprocess_nlp/blob/master/preprocess_nlp.py#L149)(for parallel) as defined in notebook

---

## Sequential & Parallel Processing

1. Sequential   - Processes records in a sequential order, does not consume a lot of CPU Memory but is slower compared to Parallel processing
2. Parallel     - Can create multiple processes (customizable/user-defined) to preprocess text parallelly, Memory intensive and faster.
<br>

---

*Refer the code for Docstrings and other function related documentation.* 
<br>
Cheers :)
