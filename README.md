# Sentiment Analysis using NLP

### Problem Setup
In this experiment, I load the positive and negative datasets into two different arrays and then
combine them into one array while labeling each sentence with 1 for positive and 0 for negative.
The order of the data is then randomized and training data is manually split by 80%. Using
NLTK’s provided libraries, each sentence undergoes standard preprocessing where all special
characters and stop words are removed, and the sentences are tokenized. Custom
preprocessors were created that stem the words using the Porter Stemmer and lemmatized
using the WordNet Lemmatizer. For every sentence, a feature vector was created using the
CountVectorizer library from scikit-learn. This created a #sentences x #features matrix with
unigram or bigram frequency counts. Unigrams/bigrams with frequency counts strictly less than
two were removed by the CountVectorizer. The same processing is done on the test data, and
then transformed to fit within the same features generated from training. Logistic regression is
then run on the data and the F1 score is used to measure the performance of the model with the
appropriate preprocessing used. F1 score was used because it’s a weighted average of
Precision and Recall, and so it takes both false positives and false negatives into account,
making it a more useful performance measure as opposed to just accuracy.

### Range of Parameter Settings
In this experiment the train-test split, elimination of special characters and stop words was kept
constant, along with the ‘min_df’ parameter which removes all unigram and bigram frequency
counts that are strictly less than two. Two parameters were changed to experiment with the
preprocessing, first is the ‘ngram_range’ that is set to either (1,1) for counting unigrams or (2,2)
for counting bigrams. The second is the ‘preprocessor’ that can be set to either lemmatization or
stemming. All combinations of these parameters were experimented and their evaluation results
are tabulated.

### Results
The following table shows the performance of the logistic regression model with different
parameter settings, measured by the F1 score (F1 = 2 * (precision * recall) / (precision + recall))

![image](https://user-images.githubusercontent.com/38390514/149220926-4f392605-7d48-40c4-adb3-556b18d74d00.png)


### Conclusion
From the results, we can observe that a unigram model that uses stemming for preprocessing
performs the best. However, lemmatization in the unigram model only has a 0.005 difference in
the F1 score therefore proving to still be a good preprocessor. The bigram models on the other
hand, did not perform as well as the unigram models with almost 0.1 difference in the F1 scores.
We observe that lemmatization had better performance with the bigram models as opposed to
stemming; perhaps since more words map to the same lemmatized form rather than the
stemmed form.

### Limitations
In each run of the experiment, our Logistic Regression model is trained with different data
between each parameter setting because of the randomized split of data. Hence, we could
observe slightly different performance results in some splits of the data. Another limitation could
be the setting of the ‘min_df’ parameter; perhaps different values of ‘min_df’ could have
achieved better results for the unigram and/or bigram models.
