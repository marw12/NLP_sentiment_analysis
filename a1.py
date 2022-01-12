# -*- coding: utf-8 -*-

import nltk
import sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score

def load_data():
    positive_lines = []
    negative_lines = []

    # Load positive and negative sentences (assuming the dataset files are in the same directory as the python file)
    with open('rt-polarity.pos', 'r', encoding= "ISO-8859-1") as file_reader:
        positive_lines = list(map(lambda x: x.replace("\n", "").strip(), file_reader.readlines()))

    with open('rt-polarity.neg', 'r', encoding = "ISO-8859-1") as file_reader:
        negative_lines = list(map(lambda x: x.replace("\n", "").strip(), file_reader.readlines()))

    # label the datasetets and combine them into one variable called sentences
    sentences = [[line,1] for line in positive_lines] + [[line,0] for line in negative_lines]

    N = len(sentences)

    # we get random indexes to shuffle the dataset
    indx = np.random.permutation(N)

    # Split Training set to be 80% and test set to be 20%
    N_train = int(0.8 * N)

    x_train, y_train = [sentences[i][0] for i in indx[:N_train]], [sentences[i][1] for i in indx[:N_train]]
    x_test, y_test = [sentences[i][0] for i in indx[N_train:]], [sentences[i][1] for i in indx[N_train:]]

    return ([x_train, y_train], [x_test, y_test])

def standard_preprocessing(text):

    # change text to lower_case
    text = text.lower()
    # remove all special charctacters and symbols
    text = re.sub("\\W", " ", text)

    # tokenize text
    words = word_tokenize(text)

    # Initialize stopwords set
    stopwords_set = set(stopwords.words('english'))

    # remove stopwords and punctuation
    filtered_words = [w for w in words if w not in stopwords_set]

    return filtered_words

# function used to preprocess with stemming
def stemming_preprocessor(text):
    # Initialize stemmer
    stemmer = PorterStemmer()

    filtered_words = standard_preprocessing(text)

    # stem words
    stemmed_output = ' '.join([stemmer.stem(word = w) for w in filtered_words])
    return stemmed_output

# function used to preprocess with lemmatizing
def lemmatize_preprocessor(text):
  # Initialize the Wordnet Lemmatizer
  lemmatizer = WordNetLemmatizer()

  filtered_words = standard_preprocessing(text)

  # Lemmatize list of words and join
  lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in filtered_words])
  return lemmatized_output

# Function processes training data
def preprocess_training_data(x_train, y_train, preprocess):

    # we use CountVectorizer to ignore words that appear in less than 2 samples
    # cv will extract the frequency of each word

    if preprocess == "unigram_lemmatize":
      cv = CountVectorizer(min_df = 2, preprocessor=lemmatize_preprocessor, ngram_range=(1,1))
    elif preprocess == "bigram_lemmatize":
      cv = CountVectorizer(min_df = 2, preprocessor=lemmatize_preprocessor, ngram_range=(2,2))
    elif preprocess == "unigram_stem":
      cv = CountVectorizer(min_df = 2, preprocessor=stemming_preprocessor, ngram_range=(1,1))
    elif preprocess == "bigram_stem":
      cv = CountVectorizer(min_df = 2, preprocessor=stemming_preprocessor, ngram_range=(2,2))

    x_train_cv = cv.fit_transform(x_train)
    
    # converts text to a matrix of token counts
    feature_vector = x_train_cv.toarray()

    return feature_vector, cv

# Function processes test data
def preprocess_test_data(x_test, cv):
    x_test_cv = cv.transform(x_test)
    feature_vector = x_test_cv.toarray()
    
    return feature_vector

def calculate_accuracy(predictions, y_test):
    num_correct = 0
    num_sampled = len(predictions)

    for (test_no, prediction) in enumerate(predictions):
        if (prediction == y_test[test_no]):
            num_correct += 1

    accuracy = num_correct/num_sampled
    return accuracy

# Function runs logistic regression
def run_logistic_regression(feature_vector_train, y_train, feature_vector_test, y_test):

  # Initialize Logistic regression model
  model = LogisticRegression()

  # fit the model with the data
  model.fit(feature_vector_train, y_train)

  # use model to predict
  predictions = model.predict(feature_vector_test)

  return f1_score(y_test, predictions)

# Read training and test data
training_data, test_data = load_data()

x_train, y_train = training_data
x_test, y_test = test_data

# list of all available preprocesses on the data
preprocess_list = ["unigram_lemmatize", "unigram_stem", "bigram_lemmatize", "bigram_stem"]

# loop throught all the preprocesses and print the results
for preprocess in preprocess_list:

  # preprocess training + testing data
  feature_vector_train, cv = preprocess_training_data(x_train, y_train, preprocess)
  feature_vector_test = preprocess_test_data(x_test, cv)

  print(preprocess, ":", run_logistic_regression(feature_vector_train, y_train, feature_vector_test, y_test))
