import pandas as pd
import word_predictor
import pickle

def batch_test(test_list, model_filename, export_df=""):
    print(" Batch Testing... ".center(60, '#'), "\n")
    columns = ["art_nr", "total_words", "test_time", "TP", "TN", "FP", "FN", "UP", "UN", "TPR", "TNR", "PPV", "F1", "MCC", "ACC"]
    df_evaluation = pd.DataFrame([], columns=columns)
    l = 0
    while l < (len(test_list)):
        for i in test_list:
            print("Article", l+1 , "/", len(test_list) , "\n")
            word_predictor.predict(i, model_filename, i, progress="bar", print_eval="no")
            df_evaluation.loc[l+1] = [(l+1), word_predictor.predict.total_words, 
                                      word_predictor.predict.time_elapsed, 
                                      word_predictor.predict.TP, 
                                      word_predictor.predict.TN, 
                                      word_predictor.predict.FP, 
                                      word_predictor.predict.FN, 
                                      word_predictor.predict.UP, 
                                      word_predictor.predict.UN, 
                                      word_predictor.predict.TPR, 
                                      word_predictor.predict.TNR, 
                                      word_predictor.predict.PPV, 
                                      word_predictor.predict.F1, 
                                      word_predictor.predict.MCC, 
                                      word_predictor.predict.ACC]
            l +=1
    if export_df != "no":
        df_evaluation.to_pickle("df_evaluation") 
    return df_evaluation.head()
            