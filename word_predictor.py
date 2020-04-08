import sys
import os
import re
import gensim
import string
import math
import random
import numpy as np
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
        words = list(model.wv.vocab)
        
        split_orig = np.array(re.findall(r"\w+|[^\w\s]", str_input, re.UNICODE))
        split_in_words = np.isin(split_orig, words)
        split_in_words = split_orig[split_in_words] 
        
        words = dict(zip(words,words))

        l = 0
        while l < (len(split_orig)):
            for i in split_orig:
                if i in string.punctuation:
                    predict_output = np.append(predict_output, i)
                    if progress=="words":
                        print(i, end=" ")
                    elif progress=="bar":
                        predict.progbar(len(predict_output), len(split_orig), 20)
                    l += 1
                elif str.isdigit(i) == True:
                    predict_output = np.append(predict_output, i)
                    if progress=="words":
                        print(i, end=" ")
                    elif progress=="bar":
                         predict.progbar(len(predict_output), len(split_orig), 20)
                    l += 1             
                elif l == 0:
                    if words.get(split_orig[l].lower()):
                        if words.get(split_orig[l].capitalize()):
                            predict_output = np.append(predict_output, i.capitalize())
                            if progress=="words":
                                print(i.upper(), end= " ")
                            elif progress=="bar":
                                predict.progbar(len(predict_output), len(split_orig), 20)
                            l += 1 
                        else: 
                            predict_output = np.append(predict_output, i.capitalize())
                            if progress=="words":
                                print(i.upper(), end= " ")
                            elif progress=="bar":
                                predict.progbar(len(predict_output), len(split_orig), 20)
                            l += 1 
                    elif words.get(split_orig[l].capitalize()):
                            predict_output = np.append(predict_output, i.capitalize())
                            if progress=="words":
                                print(i.upper(), end= " ")
                            elif progress=="bar":
                                predict.progbar(len(predict_output), len(split_orig), 20)
                            l += 1
                    else:
                        predict_output = np.append(predict_output, i.upper())
                        unknown = np.append(unknown, i)
                        if progress=="words":
                            print(i.upper(), end= " ")
                        elif progress=="bar":
                            predict.progbar(len(predict_output), len(split_orig), 20)
                        l += 1                         
                elif predict_output[-1] == ".":
                    predict_output = np.append(predict_output, i.capitalize())
                    if progress=="words":
                        print(i.capitalize(), end= " ")
                    elif progress=="bar":
                        predict.progbar(len(predict_output), len(split_orig), 20)
                    l += 1
                else:
                    if words.get(split_orig[l].lower()):
                    #if split_orig[l].lower() in words:
                    
                        if words.get(split_orig[l].capitalize()):
                        #if split_orig[l].capitalize() in words:
                            if len(found) > 0:
                                if len(split_in_words) >= (l+4):
                                    predict_with = [] + found[-11:-1]
                                    predict_with.append(split_in_words[l+1])
                                    predict_with.append(split_in_words[l+2])    
                                    predict_with.append(split_in_words[l+3])
                                    
                                    predict_next = model.predict_output_word(predict_with[0:len(predict_with)], topn=model.wv.vectors.shape[0]) 
                                    predict_next_dict = dict(predict_next)
                                
                                    val_low = predict_next_dict.get(split_orig[l].lower())
                                    val_cap = predict_next_dict.get(split_orig[l].capitalize())

                                    if val_low >= val_cap:
                                        predict_output = np.append(predict_output, i.lower())
                                        found.append(i.lower())
                                        if progress=="words":
                                            print(i.lower(), "(predicted)", end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1 
                                    elif val_low <= val_cap:
                                        predict_output = np.append(predict_output, i.capitalize())
                                        found.append(i.capitalize())
                                        if progress=="words":
                                            print(i.capitalize(), "(predicted)", end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1  
                                    else:                                                                         
                                        predict_output = np.append(predict_output, i.upper())
                                        unknown = np.append(unknown, i)
                                        if progress=="words":
                                            print(i.upper(), end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1                                    

                                elif len(split_in_words) >= (l+3):
                                    predict_with = [] + found[-11:-1]
                                    predict_with.append(split_in_words[l+1])
                                    predict_with.append(split_in_words[l+2])    

                                    predict_next = model.predict_output_word(predict_with[0:len(predict_with)], topn=model.wv.vectors.shape[0]) 
                                    predict_next_dict = dict(predict_next)
                                    
                                    val_low = predict_next_dict.get(split_orig[l].lower())
                                    val_cap = predict_next_dict.get(split_orig[l].capitalize())                                            

                                    if val_low >= val_cap:
                                        predict_output = np.append(predict_output, i.lower())
                                        found.append(i.lower())
                                        if progress=="words":
                                            print(i.lower(), "(predicted)", end= " ")      
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1 
                                    elif val_low <= val_cap:
                                        predict_output = np.append(predict_output, i.capitalize())
                                        found.append(i.capitalize())
                                        if progress=="words":
                                            print(i.capitalize(), "(predicted)", end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1  
                                    else:                                                                         
                                        predict_output = np.append(predict_output, i.upper())
                                        unknown = np.append(unknown, i)
                                        if progress=="words":
                                            print(i.upper(), end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1                                              

                                elif len(split_in_words) > (l+1): # predict with l+1
                                    predict_with = [] + found[-11:-1]
                                    predict_with.append(split_in_words[l+1])
                                    predict_next = model.predict_output_word(predict_with[0:len(predict_with)], topn=model.wv.vectors.shape[0]) 

                                    predict_next_dict = dict(predict_next)
                                    
                                    val_low = predict_next_dict.get(split_orig[l].lower())
                                    val_cap = predict_next_dict.get(split_orig[l].capitalize())                                              

                                    if val_low >= val_cap:
                                        predict_output = np.append(predict_output, i.lower())
                                        found.append(i.lower())
                                        if progress=="words":
                                            print(i.lower(), "(predicted)", end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1 
                                    elif val_low <= val_cap:
                                        predict_output = np.append(predict_output, i.capitalize())
                                        found.append(i.capitalize())
                                        if progress=="words":
                                            print(i.capitalize(), "(predicted)", end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1  
                                    else:                                                                         
                                        predict_output = np.append(predict_output, i.upper())
                                        unknown = np.append(unknown, i)
                                        if progress=="words":
                                            print(i.upper(), end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1       

                                else:
                                    if len(found) >= 10:
                                        predict_next = model.predict_output_word(found[-10:len(found)], topn=model.wv.vectors.shape[0]) 

                                        predict_next_dict = dict(predict_next)

                                        val_low = predict_next_dict.get(split_orig[l].lower())
                                        val_cap = predict_next_dict.get(split_orig[l].capitalize()) 

                                        if val_low >= val_cap:
                                            predict_output = np.append(predict_output, i.lower())
                                            found.append(i.lower())
                                            if progress=="words":
                                                print(i.lower(), "(predicted)", end= " ")
                                            elif progress=="bar":
                                                predict.progbar(len(predict_output), len(split_orig), 20)
                                            l += 1 
                                        elif val_low <= val_cap:
                                            predict_output = np.append(predict_output, i.capitalize())
                                            found.append(i.capitalize())
                                            if progress=="words":
                                                print(i.capitalize(), "(predicted)", end= " ")
                                            elif progress=="bar":
                                                predict.progbar(len(predict_output), len(split_orig), 20)
                                            l += 1  
                                        else:                                                                         
                                            predict_output = np.append(predict_output, i.upper())
                                            unknown = np.append(unknown, i)
                                            if progress=="words":
                                                print(i.upper(), end= " ")
                                            elif progress=="bar":
                                                predict.progbar(len(predict_output), len(split_orig), 20)
                                            l += 1

                                    elif len(found) >= 5:  
                                        # predict only with the last 5
                                        predict_next = model.predict_output_word(found[-5:len(found)], topn=model.wv.vectors.shape[0]) 

                                        predict_next_dict = dict(predict_next)

                                        val_low = predict_next_dict.get(split_orig[l].lower())
                                        val_cap = predict_next_dict.get(split_orig[l].capitalize()) 

                                        if val_low >= val_cap:
                                            predict_output = np.append(predict_output, i.lower())
                                            found.append(i.lower())
                                            if progress=="words":
                                                print(i.lower(), "(predicted)", end= " ")
                                            elif progress=="bar":
                                                predict.progbar(len(predict_output), len(split_orig), 20)
                                            l += 1 
                                        elif val_low <= val_cap:
                                            predict_output = np.append(predict_output, i.capitalize())
                                            found.append(i.capitalize())
                                            if progress=="words":
                                                print(i.capitalize(), "(predicted)", end= " ")
                                            elif progress=="bar":
                                                predict.progbar(len(predict_output), len(split_orig), 20)
                                            l += 1  
                                        else:                                                                         
                                            predict_output = np.append(predict_output, i.upper())
                                            unknown = np.append(unknown, i)
                                            if progress=="words":
                                                print(i.upper(), end= " ")
                                            elif progress=="bar":
                                                predict.progbar(len(predict_output), len(split_orig), 20)
                                            l += 1    

                                    else:
                                        predict_next = model.predict_output_word(found[0:len(found)], topn=model.wv.vectors.shape[0]) 

                                        predict_next_dict = dict(predict_next)

                                        val_low = predict_next_dict.get(split_orig[l].lower())
                                        val_cap = predict_next_dict.get(split_orig[l].capitalize()) 

                                        if val_low >= val_cap:
                                            predict_output = np.append(predict_output, i.lower())
                                            found.append(i.lower())
                                            if progress=="words":
                                                print(i.lower(), "(predicted)", end= " ")
                                            elif progress=="bar":
                                                predict.progbar(len(predict_output), len(split_orig), 20)
                                            l += 1 
                                        elif val_low <= val_cap:
                                            predict_output = np.append(predict_output, i.capitalize())
                                            found.append(i.capitalize())
                                            if progress=="words":
                                                print(i.capitalize(), "(predicted)", end= " ")
                                            elif progress=="bar":
                                                predict.progbar(len(predict_output), len(split_orig), 20)
                                            l += 1  
                                        else:                                                                         
                                            predict_output = np.append(predict_output,i.upper())
                                            unknown = np.append(unknown, i)
                                            if progress=="words":
                                                print(i.upper(), end= " ")
                                            elif progress=="bar":
                                                predict.progbar(len(predict_output), len(split_orig), 20)
                                            l += 1    

                            else:
                                if len(split_orig) >= (l+4):
                                    predict_with = []
                                    predict_with.append(split_in_words[l+1])
                                    predict_with.append(split_in_words[l+2])    
                                    predict_with.append(split_in_words[l+3])

                                    predict_next = model.predict_output_word(predict_with[0:len(predict_with)], topn=model.wv.vectors.shape[0]) 

                                    predict_next_dict = dict(predict_next)
                                    
                                    val_low = predict_next_dict.get(split_orig[l].lower())
                                    val_cap = predict_next_dict.get(split_orig[l].capitalize())                                              

                                    if val_low >= val_cap:
                                        predict_output = np.append(predict_output, i.lower())
                                        found.append(i.lower())
                                        if progress=="words":
                                            print(i.lower(), "(predicted)", end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1 
                                    elif val_low <= val_cap:
                                        predict_output = np.append(predict_output, i.capitalize())
                                        found.append(i.capitalize())
                                        if progress=="words":
                                            print(i.capitalize(), "(predicted)", end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1  
                                    else:                                                                         
                                        predict_output = np.append(predict_output, i.upper())
                                        unknown = np.append(unknown, i)
                                        if progress=="words":
                                            print(i.upper(), end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1          

                                elif len(split_wo_punc) >= (l+3):    
                                    predict_with = []
                                    predict_with.append(split_in_words[l+1])
                                    predict_with.append(split_in_words[l+2])    

                                    predict_next = model.predict_output_word(predict_with[0:len(predict_with)], topn=model.wv.vectors.shape[0]) 

                                    predict_next_dict = dict(predict_next)
                                    
                                    val_low = predict_next_dict.get(split_orig[l].lower())
                                    val_cap = predict_next_dict.get(split_orig[l].capitalize())                                              

                                    if val_low >= val_cap:
                                        predict_output = np.append(predict_output, i.lower())
                                        found.append(i.lower())
                                        if progress=="words":
                                            print(i.lower(), "(predicted)", end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1 
                                    elif val_low <= val_cap:
                                        predict_output = np.append(predict_output, i.capitalize())
                                        found.append(i.capitalize())
                                        if progress=="words":
                                            print(i.capitalize(), "(predicted)", end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1  
                                    else:                                                                         
                                        predict_output = np.append(predict_output, i.upper())
                                        unknown = np.append(unknown, i)
                                        if progress=="words":
                                            print(i.upper(), end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1                                                

                                elif len(split_wo_punc) > (l+1): # predict with l+1                  
                                    predict_with = []
                                    predict_with.append(split_in_words[l+1])  

                                    predict_next = model.predict_output_word(predict_with[0:len(predict_with)], topn=model.wv.vectors.shape[0]) 

                                    predict_next_dict = dict(predict_next)
                                    
                                    val_low = predict_next_dict.get(split_orig[l].lower())
                                    val_cap = predict_next_dict.get(split_orig[l].capitalize())                                                

                                    if val_low >= val_cap:
                                        predict_output = np.append(predict_output, i.lower())
                                        found.append(i.lower())
                                        if progress=="words":
                                            print(i.lower(), "(predicted)", end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1 
                                    elif val_low <= val_cap:
                                        predict_output = np.append(predict_output, i.capitalize())
                                        found.append(i.capitalize())
                                        if progress=="words":
                                            print(i.capitalize(), "(predicted)", end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1  
                                    else:                                                                         
                                        predict_output = np.append(predict_output, i.upper())
                                        unknown = np.append(unknown, i)
                                        if progress=="words":
                                            print(i.upper(), end= " ")
                                        elif progress=="bar":
                                            predict.progbar(len(predict_output), len(split_orig), 20)
                                        l += 1                                               
                                else:                                                                      
                                    predict_output = np.append(predict_output, i.upper())
                                    unknown = np.append(unknown, i)
                                    if progress=="words":
                                        print(i.upper(), end= " ")
                                    elif progress=="bar":
                                        predict.progbar(len(predict_output), len(split_orig), 20)
                                    l += 1

                        else:
                            predict_output = np.append(predict_output, i.lower())
                            found.append(i.lower())
                            if progress=="words":
                                print(i.lower(), end= " ")
                            elif progress=="bar":
                                predict.progbar(len(predict_output), len(split_orig), 20)
                            l += 1                           

                    elif words.get(split_orig[l].capitalize()):        
                    #elif split_orig[l].capitalize() in words:
                        predict_output = np.append(predict_output, i.capitalize())
                        found.append(i.capitalize())
                        if progress=="words":
                            print(i.capitalize(), end=" ")
                        elif progress=="bar":
                            predict.progbar(len(predict_output), len(split_orig), 20)
                        l += 1
                    else: 
                        predict_output = np.append(predict_output, i.upper())
                        unknown = np.append(unknown, i)
                        if progress=="words":
                            print(i.upper(), end= " ")
                        elif progress=="bar":
                            predict.progbar(len(predict_output), len(split_orig), 20)
                        l += 1

        
        predict.end_time = datetime.now()
        str_output = " ".join(predict_output)                             
        str_output = re.sub(r'\s([?:.!"](?:\s|$))', r'\1', str_output)
        predict.str_output = str_output.replace('" ', ' ')
        predict.unknown = unknown

        if evaluation != None:
            return predict.evaluate(str_input, predict_output, evaluation, print_eval)

        print("\n""\n"
            "Predicted Text: " , str(predict.str_output[0:90]) + "..." ,
            "\n""\n")  
        return

    def progbar(curr, total, full_progbar):
        frac = curr/total
        filled_progbar = round(frac*full_progbar)
        print('\r',"predicting...  ", '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='\r')
        return

    def evaluate(str_input, predict_output, actual_output, print_eval = ""):
        #global time_elapsed, total_words, TP, TN, FP, FN, UP, UN, TPR, TNR, PPV, F1, MCC, ACC
        actual_output = np.array(re.findall(r"\w+|[^\w\s]", actual_output, re.UNICODE)) # split also by punct

        time_elapsed = predict.end_time - predict.start_time
        time_elapsed = str(time_elapsed)
        time_elapsed = time_elapsed.split(".")[0] 
        predict.time_elapsed = time_elapsed

        if len(predict.unknown) > 0:
            predicted_wo_unknown = (np.isin(predict_output, np.char.upper(predict.unknown)))
            predicted_wo_unknown = predict_output[predicted_wo_unknown==False]
            
            actual_wo_unknown = (np.isin(actual_output, predict.unknown))
            actual_wo_unknown = actual_output[actual_wo_unknown==False]
        else:
            predicted_wo_unknown = predict_output
            actual_wo_unknown = actual_output
        # TP
        pred_cap = (np.isin(predicted_wo_unknown, np.char.capitalize(predicted_wo_unknown))) # predicted words that are cap
        act_cap = (np.isin(actual_wo_unknown, np.char.capitalize(actual_wo_unknown))) # words actually being cap
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

        #Unknown Positive (cap) / Unknown Negative (low)
        if len(predict.unknown) > 0:
            UP_arr = (np.isin(actual_output, np.char.capitalize(predict.unknown)))
            UP = int(sum(UP_arr))
            UN_arr = (np.isin(actual_output, np.char.lower(predict.unknown)))
            UN = int(sum(UN_arr))
        else:
            UP_arr = []
            UP = 0
            UN_arr = []
            UN = 0
        predict.UN = UN
        predict.UP = UP

        # sensitivity, recall, true-positive-rate
        TPR = TP/int(sum(act_cap))
        predict.TPR = TPR
        # specificity, sensitivity, true-negative-rate
        TNR = TN/int(sum(act_low))
        predict.TNR = TNR
        # precision, PPV
        PPV = TP/(TP+FP+UP)
        predict.PPV = PPV
        # F1 score
        F1 = round((2*((PPV*TPR)/(PPV+TPR))), 4)
        predict.F1 = F1
        # MMC, Matthews Corr. Coeff.
        MCC_1 = (TP*TN)-((FP+UN)*(FN+UP))
        MCC_2 = ((TP+(FP+UN))*(TP+(FN+UP))*(TN+(FP+UN))*(TN+(FN+UP)))
        if MCC_1 == 0:
            MCC_1 = 1
        if MCC_2 == 0:
            MCC_2 == 1
        else:
            MCC_2 = math.sqrt(MCC_2) 
        MCC = round((MCC_1/MCC_2), 4)
        predict.MCC = MCC
        
        ACC = ((TP+TN)/(TP+TN+FP+FN+UN+UP))
        predict.ACC = ACC

        total_words = TP+TN+FN+FP+UN+UP
        predict.total_words = total_words

        print(  "\n""\n"
                "Predicted Text: " , str(predict.str_output[0:90]) + "..." ,
                "\n""\n")
        
        if print_eval != "no":
            result = print(" Evaluation ".center(40, '#') ,
                    "\n""\n" 
                    "Total of " , total_words , " Words Predicted in " , time_elapsed ,
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
                    "Unknown Positive:", actual_output[UP_arr][0:10] ,
                    "\n"
                    "Unknown Negative:", actual_output[UN_arr][0:10] )  
            return result 
        