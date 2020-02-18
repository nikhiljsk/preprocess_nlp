import re
import nltk
import string
import unicodedata
import contractions
import multiprocessing
from itertools import chain
from bs4 import BeautifulSoup
from IPython.display import clear_output


def remove_tags(text):
    """
    Helper function for preprocess_nlp which is used to remove HTML Tags, Accented chars, Non-Ascii Characters, Emails and URLs
    
    :param text: A string to be cleaned
    
    <Returns the cleaned text>
    """
    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    
    
    text = unicodedata.normalize('NFKD', stripped_text).encode('ascii', 'ignore').decode('utf-8', 'ignore') # Remove Accented characters
    text = re.sub(r'[^\x00-\x7F]+','', text) # Remove Non-Ascii characters
    text = re.sub("[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", '', text) # Remove Emails
    text = re.sub(r"http\S+", "", text) # Remove URLs
    return text


def preprocess_nlp(strings, stages, ind=None, return_dict=None):
    """
    Function to preprocess a list of strings using standard nlp preprocessing techniques. 
    Note: NaN values are not supported and would raise a runtime Exception (Only <str> type is supported)
    
    :param strings: A list of strings to be preprocessed
    :param stages: A dictionary with keys as stages and values as Boolean/Integer. Can be used to customize the stages in preprocessing
    :param ind: Automatically called while using 'async_call_preprocess', indicates Index of process call
    :param return_dict: Automatically called while using 'async_call_preprocess', stores the preprocessed content for each process call
    
    (Default parameters for stages):
    {'remove_tags_nonascii': True, 
    'lower_case': True,
    'expand_contractions': False, 
    'remove_punctuation': True, 
    'remove_escape_chars': True, 
    'remove_stopwords': False, 
    'remove_numbers': True, 
    'lemmatize': False, 
    'stemming': False, 
    'min_word_len': 2}
    
    <Returns preprocessed strings>
    """
    default_stages = {'remove_tags_nonascii': True, 
                      'lower_case': True,
                      'expand_contractions': False,
                      'remove_escape_chars': True,
                      'remove_punctuation': True,
                      'remove_stopwords': False,
                      'remove_numbers': True,
                      'lemmatize': False,
                      'stemming': False,
                      'min_word_len': 2}
    
    # Update the key-values based on dictionary passed
    default_stages.update(stages)
    
    # Initializations
    processed_para = list()
    cached_stopwords = nltk.corpus.stopwords.words('english')
    stemmer = nltk.stem.snowball.SnowballStemmer("english")
    lemmatizer = nltk.stem.WordNetLemmatizer() 

    # Iterate over sentences
    for loc, paragraph in enumerate(strings):
        processed_sent = list()
        for sentence in nltk.sent_tokenize(paragraph):
            if default_stages['remove_tags_nonascii']: # Remove HTML Tags, Accented Chars, Emails, URLs, Non-Ascii
                sentence = remove_tags(sentence)
                
            if default_stages['lower_case']: # Lower-case the sentence
                sentence = sentence.lower().strip()
                
            if default_stages['expand_contractions']: # Expand contractions
                sentence = contractions.fix(sentence)
                
            if default_stages['remove_escape_chars']: # Remove multiple spaces & \n etc.
                sentence = re.sub('\s+', ' ', sentence)
                
            if default_stages['remove_punctuation']: # Remove all punctuations
                sentence = sentence.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))  # If replace with punct with space
                # sentence = sentence.translate(str.maketrans('', '', string.punctuation)) # If replace without space
                
            if default_stages['remove_stopwords']: # Remove Stopwords
                sentence = ' '.join([word for word in sentence.split() if word not in cached_stopwords])
                
            if default_stages['remove_numbers']: # Remove digits
                sentence = sentence.translate(str.maketrans('', '', string.digits))
            
            if default_stages['lemmatize']: # Lemmatize words
                sentence = ' '.join([lemmatizer.lemmatize(word) for word in sentence.split()])
                
            if default_stages['stemming']: # Stemming words
                sentence = ' '.join([stemmer.stem(word) for word in sentence.split()])
                
            if default_stages['min_word_len']: # Remove words less than minimum length
                sentence = ' '.join([word for word in sentence.split() if len(word) >= default_stages['min_word_len']])
            
            if len(sentence) < default_stages['min_word_len']:
                continue

            processed_sent.append(sentence)
        processed_para.append(' . '.join(processed_sent)) # Note: " . " is used as seperators between sentences, remove the dot and join by space if you don't want to differentiate sentences
        
        if (loc+1)%1000==0: # Print the number of records processed (Note: Does not work well if called asynchronously)
            clear_output(wait=True)
            print('Preprocessing done till: ', loc+1, '/', len(strings), sep='', flush=True)
    
    if ind == None:
        return processed_para
    
    return_dict[ind] = processed_para

    
def calculate_ranges(a, b):
    """
    Helper function for async_call_preprocess to equally divide the number of strings between multiple threads/processes.
    
    :param a: type(int)
    :param b: type(int)
    
    <Returns a list of ranges>
    
    Ex: (1200, 3) - To divide 1200 records into 3 threads we get [0, 400, 800, 1200]
    """
    try:
        ranges = list(range(0, a, a//b))
        if ranges[-1] != a:
            ranges.append(a)
        return ranges
    except ValueError:
        return [0, a]
    

def async_call_preprocess(strings, stages, n_processes=3):
    """
    Function to create async processes for faster processing. Automatically creates processe and assigns data to each process call
    
    :param strings: A list of strings to be preprocessed
    :param stages: A dictionary with keys as stages and values as Boolean/Integer. Can be used to customize the stages in preprocessing
    :param n_processes: Integer value of number of processess to be created
    (Default parameters for stages)
    {'remove_tags_nonascii': True, 
     'lower_case': True,
     'expand_contractions': False, 
     'remove_punctuation': True, 
     'remove_escape_chars': True, 
     'remove_stopwords': False, 
     'remove_numbers': True, 
     'lemmatize': False, 
     'stemming': False, 
     'min_word_len': 2}
    
    <Returns a list of preprocessed strings, aggregated from processess>
    """
    # Calculate the indices of strings to be passed to multiple processes
    ranges = calculate_ranges(len(strings), n_processes)

    # Create a Job Manager to share a dictionary that could store results of multiple processes 
    jobs = []    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    # Start creating processes and pass the records/strings according to the indices generated
    for i in range(len(ranges)-1):
        string_set = strings[ranges[i] : ranges[i+1]]
        p = multiprocessing.Process(target=preprocess_nlp, args=(string_set, stages, i, return_dict))
        jobs.append(p)
        p.start()
    
    # Wait for the result of each process
    for proc in jobs:
        proc.join()
    
    return list(chain(*list(return_dict.values())))