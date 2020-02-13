import re
import numpy as np
import multiprocessing
from itertools import chain
from collections import Counter
import matplotlib.pyplot as plt
from IPython.display import clear_output


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
    
    
def freq_of_words(strings):
    """
    Function to calculate the frequency of words in a list of strings. Calculates the cummulative frequency to get the percentage of total count to extract top words. Also plots a graph where X-axis (Count of words) Y-axis (Percentage till that word)
    
    :param strings: List of strings as input
    
    <Returns a tuple of three objects: all_words, frequency_of_each_word, percentage_of_frequency>
    """
    # Freq of words
    freq = Counter(list(chain(*[y.split() for y in chain(*[x.split(' . ') for x in strings])]))).most_common()
    words, freqs = zip(*freq)

    # Cummulative Freq
    cum_freq = np.cumsum(freqs)

    # Percentage till that word
    y = sum(freqs)
    per_freq = [x/y for x in cum_freq]

    # Plot graph
    plt.plot(per_freq)
    
    return words, freqs, per_freq


def shortlist_words(words, freqs, per_freq, threshold_freq=0.90):
    """
    Function to shortlist top % of words
    
    :param words: All the words
    :param freqs: Corresponding Frequency
    :param per_freq: Cummulative percentage of frequency
    :param threshold_freq: Top % of words to shortlist or select
    
    <Returns a tuple of two objects: shortlisted_words list and corresponding frequencies list>
    """
    # Get words that cover 97% of total frequency
    for x in per_freq:
        if x>=threshold_freq:
            short_words = words[:per_freq.index(x)+1]
            short_freqs = freqs[:per_freq.index(x)+1]
            del_words = words[per_freq.index(x)+1:]
            del_freqs = freqs[per_freq.index(x)+1:]
            break

    print('Vocab Size:', per_freq.index(x)+1)
    print('Left-out words:', len(words)-per_freq.index(x)-1)
    
    return short_words, short_freqs


def vocab_elimination(strings, short_words, replace_with='<unk>', ind=None, return_dict=None):
    """
    Function to replace the words that are not in short_words in the strings
    
    :param strings: A list of strings
    :param short_words: List of words that are allowed to be in the strings
    :param replace_with: String to be replace with the words not in short_words
    :param ind: Automatically called while using 'async_call_vocab_elimination', indicates Index of process call
    :param return_dict: Automatically called while using 'async_call_vocab_elimination', stores the preprocessed content for each process call
    
    <Returns replaced strings>
    """
    short_words = set(short_words)
    final_sent = list()
    
    for i, paragraph in enumerate(strings):
        t = list()
        for sentence in paragraph.split(' . '):
            temp = list()
            for word in sentence.split():
                if word in short_words:
                    temp.append(word)
                else:
                    temp.append(replace_with)
            t.append(' '.join(temp))
        final_sent.append(' . '.join(t))
        
        if (i+1)%1000==0:
            clear_output(wait=True)
            print('Processing done till: ', i+1, '/', len(strings), sep='', flush=True)
    if ind == None:
        return final_sent
    
    return_dict[ind] = final_sent
    

def async_call_vocab_elimination(strings, short_words, replace_with='<unk>', n_processes=5):
    """
    Function to create async processes for faster processing. Automatically creates processes and assigns data to each process call
    
    :param strings: A list of strings
    :param short_words: List of words that are allowed to be in the strings
    :param replace_with: String to be replace with the words not in short_words
    :param n_processes: Integer value of number of processess to be created
    
    <Returns a list of replaced strings, aggregated from processess>
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
        p = multiprocessing.Process(target=vocab_elimination, args=(string_set, short_words, replace_with, i, return_dict))
        jobs.append(p)
        p.start()

    # Wait for the result of each process
    for proc in jobs:
        proc.join()

    return list(chain(*list(return_dict.values())))