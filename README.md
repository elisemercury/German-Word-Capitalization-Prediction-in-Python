# German-Word-Capitalization-Prediction-in-Python
Machine Learning model for predicting first letter capitalization in German text.

Takes any German text as input and the file of a pre-trained Word2Vec or FastText CBOW model. The algorithm will iterate over each word in the input text and predict, whether the word should be written lowercase or capitalized.

Implemented using Python 3.7.3, with the help of the Gensim machine learning library v.3.8.1 and the supporting papers by [Tomas Mikolov et al.](https://arxiv.org/abs/1301.3781) and [Piotr Bojanowski et al](https://arxiv.org/abs/1607.04606).
