import spacy
import nltk
import yake
import multiprocessing
from collections import defaultdict
from IPython.display import clear_output


def calculate_ranges(a, b):
    """
    Helper function for async_call_get_features to equally divide the number of strings between multiple threads/processes.
    
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
    

def remove_duplicates(old_list):
    """
    Function to remove duplicate values in a list without changing the order
    
    :param old_list: List with duplicate values
    
    <Returns a list without duplicates values>
    """
    new_list = []
    for item in old_list:
        if item not in new_list:
            new_list.append(item)
    return new_list


def get_noun(doc):
    """
    Function to extract Nouns from the given spacy document.
    
    :param doc: Document parsed by Spacy
    
    <Returns a string of nouns seperated by ','>
    """
    noun_list = []
    for word in doc:
        if word.pos_ in ['PROPN', 'NOUN']:
            noun_list.append(word.text)
    noun_list = remove_duplicates(noun_list)
    return ",".join(noun_list)


def get_adj(doc):
    """
    Function to extract Adjectives from the given spacy document.
    
    :param doc: Document parsed by Spacy
    
    <Returns a string of adjectives seperated by ','>
    """
    adj_list = []
    for word in doc:
        if word.pos_ in ['ADJ']:
            adj_list.append(word.text)
    adj_list = remove_duplicates(adj_list)
    return ",".join(adj_list)


def get_verb(doc):
    """
    Function to extract Verbs from the given spacy document.
    
    :param doc: Document parsed by Spacy
    
    <Returns a string of verbs seperated by ','>
    """
    verb_list = []
    for word in doc:
        if word.pos_ in ['VERB']:
            verb_list.append(word.text)
    verb_list = remove_duplicates(verb_list)
    return ",".join(verb_list)


def get_ner(doc):
    """
    Function to extract NERS (Person, Location, Organization) from the given spacy document.
    
    :param doc: Document parsed by Spacy
    
    <Returns a dictionary of ners with types as keys and entities as keys>
    """
    ner_dict = defaultdict(list)
    for ent in doc.ents:
        if ent.label_ in ['PERSON']:
            ner_dict['PER'].append(ent.text)
        elif ent.label_ in ['NORP', 'ORG']:
            ner_dict['ORG'].append(ent.text)
        elif ent.label_ in ['LOC', 'GPE']:
            ner_dict['LOC'].append(ent.text)
    
    for k, _ in ner_dict.items():    
        ner_dict[k] = ','.join(remove_duplicates(ner_dict[k]))

    return dict(ner_dict)


def get_keyword(docs):
    """
    Function to extract keywords using YAKE from the given list of strings.
    
    :param docs: Strings to extract keywords from
    
    <Returns a list of string where each string contains keywords seperated by ','>
    """
    # Params to be passed for YAKE keyword Extractor
    language = "en"
    max_ngram_size = 3
    deduplication_thresold = 0.9
    deduplication_algo = 'seqm'
    numOfKeywords = 1000
    
    # Initialization
    list_of_keys = list()
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, top=numOfKeywords, features=None)
    
    # Iterate over each document and get keywords
    for loc, each_article in enumerate(docs):
        keywords = custom_kw_extractor.extract_keywords(each_article)
        temp1 = list()
        for i, j in keywords:
            temp1.append(j)
        list_of_keys.append(",".join(temp1))
    return list_of_keys


def get_number(docs):
    """
    Function to extract numbers from the given list of document.
    
    :param docs: Strings to extract numbers from
    
    <Returns a list of string where each string contains numbers seperated by ','>
    """
    numbers_list = list()
    for doc in docs:
        numbers_list.append([str(s) for s in doc.split() if s.isdigit()])
    return [','.join(x) for x in numbers_list]


def get_features(docs, stages={}, ind=None, send_end=None):
    """
    Function to extract features from the given list of strings. Uses the Spacy functions, Pipe is used to avoid unnecessary parsing to increase speed.
    
    :param docs: Strings to extract features from
    :param stages: Dictionary that contains stages to be executed
    :param ind: Automatically called while using 'async_call_get_features', indicates Index of process call
    :param send_end: Automatically called while using 'async_call_get_features', returns the preprocessed content for each process call
    
    <Returns a tuple of extracted features, 7 tuple items> \n
    
    (default_stages = {
        'nouns': True,
        'verbs': True,
        'adjs': False,
        'noun_phrases': False,
        'keywords': False,
        'ner': False,
        'numbers': False,
        })
    """
    default_stages = {
        'nouns': True,
        'verbs': True,
        'adjs': False,
        'noun_phrases': False,
        'keywords': False,
        'ner': False,
        'numbers': False,
    }
    default_stages.update(stages)

    # Define what stages to disable in the PIPE function of Spacy
    disable_list = list()
    if default_stages['nouns']==default_stages['verbs']==default_stages['adjs']==False:
        disable_list.append('tagger')
    if default_stages['ner']==False:
        disable_list.append('ner')
    if default_stages['noun_phrases']==False:
        disable_list.append('parser')
    
    # Initialization
    nlp = spacy.load('en_core_web_sm')    
    noun_chunks = list()
    verbs_list = list()
    ners_list = list()
    nouns_list = list()
    adjs_list = list()
    yake_keywords = list()
    numbers_list = list()

    # Iterate over each doc to get POS, Parsing
    for loc, doc in enumerate(nlp.pipe(docs, disable=disable_list)):
        if default_stages['verbs']:
            verbs_list.append(get_verb(doc))
            
        if default_stages['adjs']:
            adjs_list.append(get_adj(doc))
            
        if default_stages['nouns']:
            nouns_list.append(get_noun(doc))    
            
        if default_stages['ner']:
            ners_list.append(get_ner(doc))
            
        if default_stages['noun_phrases']:
            noun_chunks.append(','.join(remove_duplicates([str(x) for x in list(doc.noun_chunks)])))
        
        # Print the progress    
        if (loc+1)%500==0: # Print the number of records processed (Note: Does not work well if called asynchronously)
            clear_output(wait=True)
            print("Spacy POS", flush=True)
            print('Processing done till: ', loc+1, '/', len(docs), sep='', flush=True)

    
    if default_stages['keywords']:
        clear_output(wait=True)
        print("Extracting Keywords...")
        yake_keywords = get_keyword(docs)
        
        
    if default_stages['numbers']:
        clear_output(wait=True)
        print("Extracting Numbers...")
        numbers_list = get_number(docs)
    
    # If called directly/Sequentially
    if ind==None:
        return (nouns_list, verbs_list, adjs_list, ners_list, noun_chunks, yake_keywords, numbers_list)
    
    # If asynchronous call
    if send_end!=None:
        send_end.send((nouns_list, verbs_list, adjs_list, ners_list, noun_chunks, yake_keywords, numbers_list))


def async_call_get_features(strings, stages={}, n_processes=3):
    """
    Function to create async processes for faster processing. Automatically creates processe and assigns data to each process call
    
    :param strings: A list of strings to be processed or extracted features from
    :param stages: Dictionary that contains stages to be executed
    :param n_processes: Integer value of number of processess to be created
    
    <Returns a tuple of extracted features, 7 tuple items> \n
    
    (default_stages = {
        'nouns': True,
        'verbs': True,
        'adjs': False,
        'noun_phrases': False,
        'keywords': False,
        'ner': False,
        'numbers': False,
        })
    """
    # Calculate the indices of strings to be passed to multiple processes
    ranges = calculate_ranges(len(strings), n_processes)

    # Create a Job list
    jobs = []    
    pipe_list = []

    # Start creating processes and pass the records/strings according to the indices generated
    for i in range(len(ranges)-1):
        recv_end, send_end = multiprocessing.Pipe(False)
        string_set = strings[ranges[i] : ranges[i+1]]
        p = multiprocessing.Process(target=get_features, args=(string_set, stages, i, send_end))
        jobs.append(p)
        pipe_list.append(recv_end)
        p.start()

    # Wait for the result of each process
    for proc in jobs:
        proc.join()
      
    result_list = [x.recv() for x in pipe_list]
    
    all_list = [[], [], [], [], [], [], []]
    for k, _ in enumerate(result_list):
        for i, j in enumerate(result_list[k]):
            all_list[i] += j
        
    return all_list