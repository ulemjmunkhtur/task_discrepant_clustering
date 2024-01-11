import os
import urllib
import requests
import pandas as pd
from tqdm import tqdm
import numpy as np
# from string import punctuation
import re
import csv

import nltk#
import nltk
from tqdm import tqdm
import urllib
import requests
import numpy as np
import re
import time
import nltk
import utils

#ThreadPoolExecutor is a class that allows for parallelizing tasks
from concurrent.futures import ThreadPoolExecutor


def get_word_frequency(word):
    '''
    taken from the frequency.py file, takes in a word and returns count
    '''
    #tries getting the frequency 3 times, if it fails it returns zero
    retries = 3
    for _ in range(retries):
        try:
            encoded_query = urllib.parse.quote(word)
            params = {'corpus': 'eng-us', 'query': encoded_query, 'topk': 10, 'format': 'tsv'}
            params = '&'.join('{}={}'.format(name, value) for name, value in params.items())
            response = requests.get('https://api.phrasefinder.io/search?' + params)
            response_flat = re.split('\n|\t', response.text)[:-1]
            response_table = pd.DataFrame(np.reshape(response_flat, newshape=(-1,7))).iloc[:,:2]
            response_table.columns = ['word', 'count']
            count = response_table['count'].astype(float).sum()
            return word, count
        except requests.exceptions.ConnectionError:
            time.sleep(5)  #tries again after 5 seconds
    return 0  


def get_words(letter):
    cap_letter = letter.upper()
    words_nltk = nltk.corpus.brown.words()
    words_webtext = nltk.corpus.webtext.words()
    words_reuters = nltk.corpus.reuters.words()
    words_words = nltk.corpus.words.words()

    combined = words_nltk + words_webtext + words_reuters + words_words
    letter_words = []
    for w in combined:
        if w.startswith(letter) or w.startswith(cap_letter):
            if re.match("^[a-z]*$", w):
                if len(w) > 2 and len(w) < 15:
                    letter_words.append(w)

    lower_words = set([w.lower() for w in letter_words])

    #parallelizes get_word_frequency() for each word in lower words
    #tqdm displays the progress bar
    with ThreadPoolExecutor(max_workers=10) as executor:
        word_frequencies = list(tqdm(executor.map(get_word_frequency, lower_words), total=len(lower_words)))
    
    #sorts the list in descending order based on word frequencies 
    word_frequencies.sort(key=lambda x: x[1], reverse=True)
    #this extracts the 2500 words with the highest frequency 
    top_words = [word for word, freq in word_frequencies[:2500]]

    #save to CSV
    file_save = os.path.join(f"output/letter_a", f"{letter}_og.csv")
    with open(file_save, 'w') as f:
        for word in top_words:
            f.write(word + '\n')

    return top_words



def add_missing_words(letter): 
    '''
    compares the list of words generated with get_words() starting with the given letter with the words that the participant responded with
    if the participant has a word that isn't already in the list, add it to the list
    generates: 
    - {letter}_words.csv: contains all the unique words from participants added to {letter}_og.csv
    - {letter}_action.csv: contains all the participant responses and whether that word was found or added to the list
    '''
    words_df = pd.read_csv(f"output/letter_a/{letter}_og.csv", header=None, names=["Word"])
    responses_df = pd.read_csv(f"forager/data/fluency_lists/transformed_data/{letter}_transformed.csv")

    #extract the words
    words_list = words_df["Word"].str.lower().tolist()
    responses_words = responses_df["Response"].str.lower().unique().tolist()

    #identify the from participant data that are not in the other CSV
    missing_words = []
    action = []
    for word in responses_words: 
        if word not in words_list:
            missing_words.append(word)
            action.append("Added")
        else:
            action.append("Found")

    #append the missing words to the words_df
    if missing_words:
        missing_df = pd.DataFrame(missing_words, columns=["Word"])
        words_df = words_df.append(missing_df, ignore_index=True)

        #save the updated DataFrame back to the CSV
        words_df.to_csv(f"output/letter_{letter}/{letter}_words.csv", index=False, header=False)
        print(f"Added {len(missing_words)} missing words to {letter}_og.csv and put them in {letter}_words.csv")
    else:
        print("All words from the participant data are present in the generated words CSV.")
    
    #create df with the action performed
    action_df = pd.DataFrame({"Response": responses_words, "Action Performed": action})
    action_df.to_csv(f"output/letter_{letter}/{letter}_action.csv", index=False)


def transform(participant_responses):
    #flatten the list of tuples into a DataFrame
    flattened_data = [(participant, response) for participant, responses in participant_responses for response in responses]
    df = pd.DataFrame(flattened_data, columns=['Participant', 'Response'])

    df.to_csv("forager/data/fluency_lists/transformed_data/excluded_animals.csv", index=False)

def exlude_animals(): 
    """
    excludes the words participants say that aren't on the fluency list"""
    data_path = "forager/data/fluency_lists/transformed_data"
    lexical_data_path = "forager/data/lexical_data/letter_animals"
    letter_category = "animals"  # ex. words starting with 'a'
    labels = pd.read_csv(f"forager/data/lexical_data/letter_animals/USE_frequencies.csv",header=None)[0].values.tolist()
# labels = pd.read_csv(os.path.join("/Users/ulemjmunkhtur/Desktop/GitHub/task-discrepant-clustering/forager/data/lexical_data/animals/",f"USE_frequencies.csv"), names=['word', 'logct', 'ct']) 
# # Prepare the data
    data, replacement_df, original_df = utils.prepareData(data_path, lexical_data_path, letter_category, labels)
    transform(data)


def csv_to_list(words_file_csv):
    '''convert words in csv file to list
    '''
    items_list = []
    data = pd.read_csv(words_file_csv) 
    for index, row in data.iterrows():
        word = row[0].replace(".", "")
        items_list.append(word)
    return items_list


def dataset_words(path_to_csv):
    '''get list of words in the dataset
    '''
    characters = [" ", "[", "]", "//", ".", '\\', ",", "'", '"', "|", "`", "/", "{", "}", ":", ";", "<", ">", "?", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "+", "=", "~"]
    
    with open(path_to_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
    
        # Skip the header row
        next(reader)
    
        # Create an empty list to store the values in the second column
        words = []
    
        # Loop through each row and append the value in the second column to the list
        for row in reader:
            # has the form '13807\tsand' so word is after '\t'
            word = row[0].split('\t')[1]
            # remove random chars and keep whatever's till that random char
            for char in characters:
                if char in word:
                    word = word.split(char)[0]

            words.append(word.lower())

    return list(set(words))

def dataset_compare(brown_list, dataset_words):
    '''
    path_to_csv: path to csv file with all words in dataset
    return list of words in dataset not present in letter_words
    '''
    words_not_in_brown =  set(dataset_words) - set(brown_list) 
    return list(words_not_in_brown)

def add_words(words_not_in_brown, brown_list):
    '''return list of words not in brown added to words in brown list for letter
    '''
    return words_not_in_brown + brown_list

def check_corpus(word_list):
    for word in word_list:
        if ' ' in word[1:-1] or '-' in word[1:-1]:
            print(word)



### SAMPLE CODE ###

#get_words('f')
# add_missing_words('a')