import sys
import os
import re
import gensim
import string
import math
import random
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

class predict():

    str_output, unknown = [], []
    start_time, end_time, time_elapsed, total_words, TP, TN, FP, FN, UP, UN, TPR, TNR, PPV, F1, MCC, ACC = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

    def __init__(self, str_input, model_filename, evaluation = None, progress = "", print_eval = ""):
        """  
        str_input.........string, takes any German text as input
        model_filename....string, takes a pretrained Word2Vec Embedding .bin file, will be loaded with Gensim
        evaluation........optional, string, takes same text as in str_input but with correct capitalization
        progress..........optional, use progress="bar" for displaying prediction progress bar by %,
                                    use progress="words" for displaying prediction progress by words,
        print_eval........optional, if print_eval="no" then will not print evaluation results
                                    
        """
        
        self.str_input = str_input
        self.model_filename = model_filename
        self.evaluation = evaluation
        self.progress = progress
        self.print_eval = print_eval

        #global str_output, start_time, end_time, unknown
        found = []
        predict_output, unknown = np.array([]) , np.array([])
        predict.start_time = datetime.now()
        
        print(" Prediction ".center(40, '#') ,
                "\n""\n" 
                "Input Text:     " , str(str_input[0:90]) + "...")
        print("\n predicting...", end= "   ")
        
        model = gensim.models.Word2Vec.load(model_filename)
        predict.model = model
        words = list(predict.model.wv.vocab)
        
        split_orig = np.array(re.findall(r"\w+|[^\w\s]", str_input, re.UNICODE))
        split_in_words = np.isin(split_orig, words)
        split_in_words = split_orig[split_in_words] 
        
        words = dict(zip(words,words))

        l = 0
        while l < (len(split_orig)):
            for i in split_orig:
                # is target word punctuation?
                if i in string.punctuation:
                    predict_output = np.append(predict_output, i)
                    if progress=="words":
                        print(i, end=" ")
                    elif progress=="bar":
                        predict.progbar(len(predict_output), len(split_orig), 20)
                    l += 1
                # is target word number?
                elif str.isdigit(i) == True:
                    predict_output = np.append(predict_output, i)
                    if progress=="words":
                        print(i, end=" ")
                    elif progress=="bar":
                         predict.progbar(len(predict_output), len(split_orig), 20)
                    l += 1    
                # is target word first word in text?         
                elif l == 0:
                    if words.get(split_orig[l].lower()) or words.get(split_orig[l].capitalize()): 
                        # capitalize
                        predict_output = np.append(predict_output, i.capitalize())
                        if progress=="words":
                            print(i.capitalize(), end= " ")
                        elif progress=="bar":
                            predict.progbar(len(predict_output), len(split_orig), 20)
                        l += 1  
                    else:
                        # not present in vocab, unknown
                        predict_output = np.append(predict_output, i.upper())
                        unknown = np.append(unknown, i)
                        if progress=="words":
                            print(i.upper(), end= " ")
                        elif progress=="bar":
                            predict.progbar(len(predict_output), len(split_orig), 20)
                        l += 1 
                # is target word after a dot?                        
                elif predict_output[-1] == ".":
                    if words.get(split_orig[l].lower()) or words.get(split_orig[l].capitalize()):
                        # capitalize
                        predict_output = np.append(predict_output, i.capitalize())
                        if progress=="words":
                            print(i.capitalize(), end= " ")
                        elif progress=="bar":
                            predict.progbar(len(predict_output), len(split_orig), 20)
                        l += 1 
                    else:
                        # not present in vocab, unknown
                        predict_output = np.append(predict_output, i.upper())
                        unknown = np.append(unknown, i)
                        if progress=="words":
                            print(i.upper(), end= " ")
                        elif progress=="bar":
                            predict.progbar(len(predict_output), len(split_orig), 20)
                        l += 1                         
                else:
                    # is word spelled lower case present in vocab?
                    if words.get(split_orig[l].lower()):

                        # is word spelled upper case present in vocab?
                        if words.get(split_orig[l].capitalize()):

                            if len(found) > 0:
                                # predict with last 5 and next 5 words
                                context_words_list = np.array(found[-6:len(found)])
                                context_words_list = np.append(context_words_list, split_in_words[l+1:l+6])

                                # calculate probability of target word given context words
                                word_vocabs = [predict.model.wv.vocab[w] for w in context_words_list if w in predict.model.wv.vocab]
                                word2_indices = [word.index for word in word_vocabs]

                                l1 = np.sum(predict.model.wv.vectors[word2_indices], axis=0)
                                if word2_indices and predict.model.cbow_mean:
                                    l1 /= len(word2_indices)

                                # propagate hidden -> output and take softmax to get probabilities
                                targetword_indices = [(predict.model.wv.vocab[split_orig[l].lower()]).index, (predict.model.wv.vocab[split_orig[l].capitalize()]).index]
                                targetword_values = (predict.model.trainables.syn1neg[targetword_indices]).T

                                prob_values = np.exp(np.dot(l1, targetword_values))
                                prob_values /= sum(prob_values)
                                result_index = prob_values.argmax() 

                                if result_index == 0:
                                    # lower
                                    predict_output = np.append(predict_output, i.lower())
                                    found.append(i.lower())
                                    if progress=="words":
                                        print(i.lower(), "(predicted)", end= " ")
                                    elif progress=="bar":
                                        predict.progbar(len(predict_output), len(split_orig), 20)
                                    l += 1 
                                elif result_index == 1:
                                    # cap
                                    predict_output = np.append(predict_output, i.capitalize())
                                    found.append(i.capitalize())
                                    if progress=="words":
                                        print(i.capitalize(), "(predicted)", end= " ")
                                    elif progress=="bar":
                                        predict.progbar(len(predict_output), len(split_orig), 20)
                                    l += 1  
                                else:   
                                    # unknown                                                                      
                                    predict_output = np.append(predict_output, i.upper())
                                    unknown = np.append(unknown, i)
                                    if progress=="words":
                                        print(i.upper(), end= " ")
                                    elif progress=="bar":
                                        predict.progbar(len(predict_output), len(split_orig), 20)
                                    l += 1                                    

                            else:
                                # if no predecessing words found yet
                                if len(split_in_words) >= (l+1):
                                    # predict with next 5 words
                                    context_words_list = split_in_words[l+1:l+6] 

                                    # calculate probability of target word given context words
                                    word_vocabs = [predict.model.wv.vocab[w] for w in context_words_list if w in predict.model.wv.vocab]
                                    word2_indices = [word.index for word in word_vocabs]

                                    l1 = np.sum(predict.model.wv.vectors[word2_indices], axis=0)
                                    if word2_indices and predict.model.cbow_mean:
                                        l1 /= len(word2_indices)

                                    # propagate hidden -> output and take softmax to get probabilities
                                    targetword_indices = [(predict.model.wv.vocab[split_orig[l].lower()]).index, (predict.model.wv.vocab[split_orig[l].capitalize()]).index]
                                    targetword_values = (predict.model.trainables.syn1neg[targetword_indices]).T

                                    prob_values = np.exp(np.dot(l1, targetword_values))
                                    prob_values /= sum(prob_values)
                                    result_index = prob_values.argmax() 

                                    if result_index == 0:
                                        # lower
                                        predict_output = np.append(predict_output, i.lower())
                                        found.append(i.lower())
                                        if progress=="words":
                                            print(i.lower(), "(predicted)", end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1 
                                    elif result_index == 1:
                                        # cap
                                        predict_output = np.append(predict_output, i.capitalize())
                                        found.append(i.capitalize())
                                        if progress=="words":
                                            print(i.capitalize(), "(predicted)", end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1  
                                    else:   
                                        # unknown                                                                      
                                        predict_output = np.append(predict_output, i.upper())
                                        unknown = np.append(unknown, i)
                                        if progress=="words":
                                            print(i.upper(), end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1                                    
                                  
                                else:
                                    # no words before and no words after target, unknown                                                                      
                                    predict_output = np.append(predict_output, i.upper())
                                    unknown = np.append(unknown, i)
                                    if progress=="words":
                                        print(i.upper(), end= " ")
                                    elif progress=="bar":
                                        predict.progbar(len(predict_output), len(split_orig), 20)
                                    l += 1

                        else:
                            # word only as lower spelling in vocab
                            predict_output = np.append(predict_output, i.lower())
                            found.append(i.lower())
                            if progress=="words":
                                print(i.lower(), end= " ")
                            elif progress=="bar":
                                predict.progbar(len(predict_output), len(split_orig), 20)
                            l += 1                           

                    elif words.get(split_orig[l].capitalize()):        
                        # word only as cap spelling in vocab
                        predict_output = np.append(predict_output, i.capitalize())
                        found.append(i.capitalize())
                        if progress=="words":
                            print(i.capitalize(), end=" ")
                        elif progress=="bar":
                            predict.progbar(len(predict_output), len(split_orig), 20)
                        l += 1
                    else: 
                        # unknown
                        predict_output = np.append(predict_output, i.upper())
                        unknown = np.append(unknown, i)
                        if progress=="words":
                            print(i.upper(), end= " ")
                        elif progress=="bar":
                            predict.progbar(len(predict_output), len(split_orig), 20)
                        l += 1

        
        
        str_output = " ".join(predict_output)                             
        str_output = re.sub(r'\s([?:.,!](?:\s|$))', r'\1', str_output)
        predict.str_output = str_output
        predict.unknown = unknown
        predict.end_time = datetime.now()

        if evaluation != None:
            return predict.evaluate(str_input, str_output, evaluation, print_eval)

        print("\n""\n"
            "Predicted Text: " + str(predict.str_output),
            "\n""\n")  
        return

    def progbar(curr, total, full_progbar):
        frac = curr/total
        filled_progbar = round(frac*full_progbar)
        print('\r',"predicting...  ", '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='\r')
        return

    def evaluate(str_input, predict_output, actual_output, print_eval = ""):
        # split and remove punctuation
        actual_output_wo_punc = np.array(re.findall(r"\w+", actual_output, re.UNICODE)) # split also by punct
        predict_output_wo_punc = np.array(re.findall(r"\w+", predict_output, re.UNICODE))
        
        # remove digits
        actual_output_clean = np.array([])
        i=0
        while i < len(actual_output_wo_punc):
            for j in actual_output_wo_punc:
                if j.isdigit():
                    i+=1
                else:
                    actual_output_clean = np.append(actual_output_clean, j)
                    i+=1
        
        predict_output_clean = np.array([])
        i=0
        while i < len(predict_output_wo_punc):
            for j in predict_output_wo_punc:
                if j.isdigit():
                    i+=1
                else:
                    predict_output_clean = np.append(predict_output_clean, j)
                    i+=1 

        del actual_output_wo_punc
        del predict_output_wo_punc

        if len(predict.unknown) > 0:
            predicted_wo_unknown = (np.isin(predict_output_clean, np.char.upper(predict.unknown)))
            predicted_wo_unknown = predict_output_clean[predicted_wo_unknown==False]
            actual_wo_unknown = (np.isin(actual_output_clean, predict.unknown))
            actual_wo_unknown = actual_output_clean[actual_wo_unknown==False]
        else:
            predicted_wo_unknown = predict_output_clean
            actual_wo_unknown = actual_output_clean

        # TP
        pred_cap = (np.isin(predicted_wo_unknown, np.char.capitalize(predicted_wo_unknown))) # predicted words that are cap
        act_cap = (np.isin(actual_wo_unknown, np.char.capitalize(actual_wo_unknown))) # words actually being cap

        if pred_cap.shape != act_cap.shape:
            predict.TP = np.NaN
            predict.total_words = np.NaN
            predict.time_elapsed = np.NaN 
            predict.words_sec = np.NaN 
            predict.TP = np.NaN  
            predict.TN = np.NaN  
            predict.FP = np.NaN  
            predict.FN = np.NaN  
            predict.UP = np.NaN  
            predict.UN = np.NaN  
            predict.TPR = np.NaN  
            predict.TNR = np.NaN  
            predict.PPV = np.NaN  
            predict.F1 = np.NaN  
            predict.MCC = np.NaN  
            predict.ACC = np.NaN 
            return

        else:    
            TP = int(sum(np.logical_and(pred_cap==True, act_cap==True)))
            predict.TP = TP
            
            # FP 
            FP_bool = np.logical_and(pred_cap==True, act_cap==False)
            FP_words = predicted_wo_unknown[FP_bool]
            FP = int(sum(FP_bool))
            predict.FP = FP

            # TN
            pred_low = (np.isin(predicted_wo_unknown, np.char.lower(predicted_wo_unknown)))
            act_low = (np.isin(actual_wo_unknown, np.char.lower(actual_wo_unknown)))
            TN = int(sum(np.logical_and(pred_low==True, act_low==True)))
            predict.TN = TN

            # FN
            FN_bool = np.logical_and(pred_low==True, act_low==False)
            FN_words = predicted_wo_unknown[FN_bool]
            FN = int(sum(FN_bool))
            predict.FN = FN

            # Unknown Positive (cap) / Unknown Negative (low)
            if len(predict.unknown) > 0:
                UP_bool = (np.isin(actual_output_clean, np.char.capitalize(predict.unknown)))
                UP = int(sum(UP_bool))
                UN_bool = (np.isin(actual_output_clean, np.char.lower(predict.unknown)))
                UN = int(sum(UN_bool))
            else:
                UP_bool = []
                UP = 0
                UN_bool = []
                UN = 0
            predict.UN = UN
            predict.UP = UP

            # Performance measurement excl. UP/UN

            # sensitivity, recall, true-positive-rate
            if TP != 0 or FN != 0:
                TPR = TP/(TP + FN)
            else:
                TPR = 0
            predict.TPR = TPR

            # specificity, sensitivity, true-negative-rate
            if TN != 0 or FP != 0:
                TNR = TN/(TN + FP)
            else:
                TNR = 0
            predict.TNR = TNR

            # precision, PPV
            if TP != 0 or FP != 0:
                PPV = TP/(TP + FP)
            else:
                PPV = 0
            predict.PPV = PPV

            # F1 score
            if PPV != 0 and TPR != 0:
                F1 = round((2 * ((PPV * TPR)/(PPV + TPR))), 4)
            else:
                F1 = 0
            predict.F1 = F1

            # MMC, Matthews Corr. Coeff.
            MCC_1 = (TP * TN) - (FP * FN)

            MCC_2 = math.sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN))

            if MCC_2 == 0:
                MCC = 0
            else:
                MCC = round((MCC_1/MCC_2), 4)

            predict.MCC = MCC
            
            # ACC, Accuracy
            if FN != 0 or FP != 0:
                ACC = ((TP+TN)/(TP+TN+FN+FP))
            else:
                ACC = 1
            predict.ACC = ACC


            # Performance measurement incl. UP/UN

            # sensitivity, recall, true-positive-rate
            if TP != 0 or FN != 0 or UP != 0:
                TPR2 = TP/(TP + FN + UP)
            else:
                TPR2 = 0
            predict.TPR2 = TPR2

            # specificity, sensitivity, true-negative-rate
            if TN != 0 or FP != 0 or UN != 0:
                TNR2 = TN/(TN + FP + UN)
            else:
                TNR2 = 0
            predict.TNR2 = TNR2

            # precision, PPV
            if TP != 0 or FP != 0 or UN != 0:
                PPV2 = TP/(TP + FP + UN)
            else:
                PPV2 = 0
            predict.PPV2 = PPV2

            # F1 score
            if PPV2 != 0 and TPR2 != 0:
                F12 = round((2 * ((PPV2 * TPR2)/(PPV2 + TPR2))), 4)
            else:
                F12 = 0
            predict.F12 = F12

            # MMC, Matthews Corr. Coeff.
            MCC_12 = (TP * TN) - ((FP + UN) * (FN + UP))

            MCC_22 = math.sqrt((TP+FP+UN)  * (TP+FN+UP) * (TN+FP+UN) * (TN+FN+UP))

            if MCC_22 == 0:
                MCC2 = 0
            else:
                MCC2 = round((MCC_12/MCC_22), 4)

            predict.MCC2 = MCC2
            
            # ACC, Accuracy
            if FN != 0 or FP != 0 or UN != 0 or UP != 0:
                ACC2 = ((TP+TN)/(TP+TN+FN+UP+FP+UN))
            else:
                ACC2 = 1
            predict.ACC2 = ACC2

            ########

            # Total words predicted
            # excl. UP/UN
            total_words = TP+TN+FN+FP
            predict.total_words = total_words

            # incl. UP/UN
            total_words2 = TP+TN+FN+UP+FP+UN
            predict.total_words2 = total_words2

            ## Prediction time
            time_elapsed = predict.end_time - predict.start_time
            # Words / Sec
            words_sec = round(total_words/(time_elapsed.total_seconds()), 2)
            predict.words_sec = words_sec
            time_elapsed = str(time_elapsed)
            time_elapsed = time_elapsed.split(".")[0] 
            predict.time_elapsed = time_elapsed

            print(  "\n""\n"
                    "Predicted Text: " + str(predict.str_output[0:90]) + "..." ,
                    "\n""\n")
            
            if print_eval != "no":
                result = print(" Evaluation ".center(40, '#') ,
                        "\n""\n" 
                        "Total of" , total_words , "Words Predicted in" , time_elapsed , "that is" , words_sec , "words/sec."
                        "\n""\n"    
                        f'{"True Positive":23} ==> {TP:5d}' ,
                        "\n" 
                        f'{"False Positive":23} ==> {FP:5d}' ,
                        "\n"
                        f'{"True Negative":23} ==> {TN:5d}' ,
                        "\n"
                        f'{"False Negative":23} ==> {FN:5d}' ,
                        "\n""\n"
                        f'{"Unknown Positive":23} ==> {UP:5d}' ,
                        "\n"
                        f'{"Unknown Negative":23} ==> {UN:5d}' ,
                        "\n""\n"  
                        "TPR:\t\t", round((TPR*100), 4) , "%" , "\t" , "TNR:\t\t" , round((TNR*100), 4) , "%"
                        "\n"
                        "Accuracy:\t", round((ACC*100), 4) , "%" , "\t" , "Precision:\t" , round((PPV*100), 4) , "%"
                        "\n"
                        "F1:\t\t", F1 , "\t" , "MCC:\t\t" , MCC ,
                        "\n""\n"
                        "False Positive Words:", FP_words[0:10] ,
                        "\n"
                        "False Negative Words:", FN_words[0:10] ,
                        "\n"
                        "Unknown Positive:", actual_output_clean[UP_bool][0:10] ,
                        "\n"
                        "Unknown Negative:", actual_output_clean[UN_bool][0:10] )  
                return result 

    def batch_test(test_list, model_filename, export_df=""):
        print(" Batch Testing... ".center(60, '#'), "\n")
        columns = ["art_nr", "total_words", "total_words2", "test_time", "words/s", "TP", "TN", "FP", "FN", "UP", "UN", "TPR", "TNR", "PPV", "F1", "MCC", "ACC", "TPR2", "TNR2", "PPV2", "F12", "MCC2", "ACC2"]
        df_evaluation = pd.DataFrame([], columns=columns)
        unknown_list = np.array([])
        model = gensim.models.Word2Vec.load(model_filename)
        predict.model = model

        l = 0
        while l < (len(test_list)):
            for i in test_list:
                print("Article", l+1 , "/", len(test_list) , "\n")
                predict(i, model_filename, i, progress="", print_eval="no")
                df_evaluation.loc[l+1] = [(l+1), 
                                        predict.total_words, 
                                        predict.total_words2,
                                        predict.time_elapsed, 
                                        predict.words_sec,
                                        predict.TP, 
                                        predict.TN, 
                                        predict.FP, 
                                        predict.FN, 
                                        predict.UP, 
                                        predict.UN, 
                                        predict.TPR, 
                                        predict.TNR, 
                                        predict.PPV, 
                                        predict.F1, 
                                        predict.MCC, 
                                        predict.ACC,
                                        predict.TPR2, 
                                        predict.TNR2, 
                                        predict.PPV2, 
                                        predict.F12, 
                                        predict.MCC2, 
                                        predict.ACC2]
                unknown_list = np.append(unknown_list, predict.unknown)
                l +=1
        if export_df != "no":
            df_evaluation.to_pickle(str("df_evaluation_" + export_df))
            with open("unknown_list_" + str(export_df) + ".pkl", "wb") as f:
                pickle.dump(unknown_list, f)
        return df_evaluation.head()    