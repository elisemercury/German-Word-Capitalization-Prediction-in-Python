# German Word Capitalization Prediction
Machine Learning model for predicting first letter capitalization in German text.

Takes any German text as input and the file of a pre-trained Word2Vec or FastText CBOW model. The algorithm will iterate over each word in the input text and predict, whether the word should be written lowercase or capitalized.

Implemented using Python 3.7.3, with the help of the Gensim machine learning library v.3.8.1 and the supporting papers by [Tomas Mikolov et al.](https://arxiv.org/abs/1301.3781) and [Piotr Bojanowski et al](https://arxiv.org/abs/1607.04606).

# Usage

For predicting whether the words in a German text should be written with lower of upper case first letter, use the function below.
This takes a single Python string of German text as input. It iterates over each word in the text and predicts its first letter spelling by using pre-trained Word2Vec (or FastText for the second script) word embeddings (pre-trained using Gensim v.3.8.1).

> import Word2Vec_word_predictor
> Word2Vec_word_predictor.predict(self, str_input, model_filename, evaluation = None, progress = "", print_eval = "")

  str_input.........string, takes any German text as input
  model_filename....string, takes a pretrained Gensim Word2Vec Embedding file, will be loaded with Gensim 
  evaluation........optional, string, takes same text as in str_input but with correct capitalization
  progress..........optional, use progress="bar" for displaying prediction progress bar by %,
                              use progress="words" for displaying prediction progress by words,
  print_eval........optional, if print_eval="no" then will not print test evaluation results

For predicting on a list of multiple German texts, use this function below.
This takes a Python list of German texts as input and predicts first letter spelling for all words in the texts. After testing is done, the test results can optionally be outputed as Pandas dataframe as a pickle file by adjusting "export_df".

> Word2Vec_word_predictor.predict.batch_test(test_list, model_filename, export_df="")

  test_list.........list, a list of German texts/sentences
  model_filename....string, takes a pretrained Gensim Word2Vec Embedding file, will be loaded with Gensim
  export_df.........optional, string, takes the string for the name of the test evaluation output dataframe,
                                      use export_df="no" for not exporting a test evaluation dataframe, 
                                      exports as pickle file and includes metrics like Sensitivity, Specificity, 
                                      Accuracy, MCC, etc. Final file name will be "df_evaluation_" + export_df.pkl
