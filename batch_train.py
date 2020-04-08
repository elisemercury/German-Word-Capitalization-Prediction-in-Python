import os
import random
import re
from datetime import datetime
from gensim.models import Word2Vec
from nltk.tokenize import *
from nltk.corpus import stopwords 
import numpy as np
import pickle

# german stopwords to exclude from lowering
de_stop_words = stopwords.words('german')
de_stop_words = np.array(list(de_stop_words))
exclude = np.array(["allen", "andere", "anderen", "hat", "man", "muss", "waren", "weg", "will"])
exclude = (np.isin(de_stop_words, exclude))
de_stop_words = de_stop_words[exclude==False]
de_stop_words_cap = [i.capitalize() for i in de_stop_words]
de_stop_words_dict = {de_stop_words_cap[i]: de_stop_words[i] for i in range(len(de_stop_words_cap))}

def batch_train(output_file, max_batch, max_train): 
    """output_file....string, empty .bin file for saving the trained model
       max_batch......int, maximum batch of file folders to train at once
       ticker.........int, maximum number of file folder to be used for training, must be a multiple of max_batch
       ---
       The textdata fro training will be extrated from ..\extracted\AA...ZZ\wiki_00...wiki_99 """
    
    train, train_list, test_list = [], [], []
    dir_list = os.listdir(os.getcwd() + "\\extracted\\")
    random.shuffle(dir_list)

    l = 0
    folder = 0
    count_size = 0
    while l < len(dir_list):
        for i in dir_list:
            for filename in os.listdir(os.getcwd() + "\\extracted\\" + dir_list[folder]):
                count_size += os.path.getsize(os.getcwd() + "\\extracted\\" + dir_list[folder] + "\\" + filename)
            folder += 1
            l += 1
    max_size = round(count_size * max_train)

    l=0
    folder=0
    size = 0
    while size < max_size:
        
        while l < max_batch and size < max_size:
            for filename in os.listdir(os.getcwd() + "\\extracted\\" + dir_list[folder]):
                with open(os.path.join(os.getcwd() + "\\extracted\\" + dir_list[folder] + "\\" + filename), 'rt', encoding="utf-8") as f:
                    size += os.path.getsize(os.getcwd() + "\\extracted\\" + dir_list[folder] + "\\" + filename)
                    train.append(f.read())
            folder += 1
            l += 1
        
        clean_train = [x.encode('utf-8').decode('utf-8') for x in train]
        print("1 at", datetime.now())

        # cleaning

        # removing double space and line breaks
        clean_train = [(x.replace("\n", " ")).replace("  ", " ") for x in clean_train] 
        print("2 at", datetime.now())

        # remove <doc> tag, link and article id in beginning
        clean_train = [re.sub(r'<.+?> ', '', x) for x in clean_train]
        print("3 at", datetime.now())

        # remove digits, and the following words f.e. 1. August 2020 - 1. and 2020 are removed
        clean_train = [re.sub("\S*\d\S*", '', x).strip() for x in clean_train]
        print("4 at", datetime.now())

        # tokenize in list of articles with sentences as list
        clean_train = [sent_tokenize(x) for x in clean_train]
        clean_train = [item for sublist in clean_train for item in sublist] # flatten list; one big list with multiple sentences
        print("5 at", datetime.now())

        # remove punctuation
        clean_train = [re.sub(r'[^\w\s]', '', x).strip() for x in clean_train]
        print("6 at", datetime.now())

        # tokenize all words in sentences = on bis list, with list of articles with tokenized words
        clean_train = [word_tokenize(x) for x in clean_train]  
        print("7 at", datetime.now())

        # convert stopwords to lower case
        replace_matched_items(clean_train, de_stop_words_dict)
        print("8 at", datetime.now())

        #training

        if folder <= max_batch:
            model = Word2Vec(clean_train)
            model.save(output_file)
            print("Model trained at", datetime.now())
            del model
            train = []
            l = 0
        else:
            model = Word2Vec.load(output_file)
            model.train(clean_train, total_examples=(len(clean_train)), epochs=model.epochs)
            model.save(output_file)
            del model
            print("Model updated at ", datetime.now()) 
            train = []
            l = 0

    train_list[:] = dir_list[0:folder]
    test_list[:] = dir_list[folder:len(dir_list)]
    with open("train_list_" + str(max_batch) + ".pkl", "wb") as f:
        pickle.dump(train_list, f)
    with open("test_list_" + str(max_batch) + ".pkl", "wb") as f:
        pickle.dump(test_list, f)
    print("Trained on" , str(round((size/count_size)*100, 2)) , "% of" , str(max_size) + "B data" ,
          "\nFolders used for training:" , dir_list[0:folder])

# function for replacing elements in a list matching the given dictionary keys
def replace_matched_items(word_list, dictionary):
    for lst in word_list:
        for ind, item in enumerate(lst):
            lst[ind] = dictionary.get(item, item)