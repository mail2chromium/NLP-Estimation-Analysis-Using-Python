import re
import time
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


class PredictiveAnalysis(object):

    def __init__(self):
        self.corpus = []

    ################################################################

    @staticmethod
    def fetch_data(file_name):

        """
        Importing files(.csv, .tsv, .excel etc...)
        quoting = 3 is for ignoring "" for our safety.

        :param file_name:

        :return: data_set
        """

        data_set = pd.read_csv(file_name, delimiter='\t', quoting=3)
        return data_set

    ################################################################

    def clean_data(self, data_set):

        """
        Cleaning the text
        stopwords is a list of unwanted words like the,and,of,etc...
        corpus is a collection of text.
        :param data_set:
        :return: corpus
        """

        for i in range(0, 1000):

            """
            Removing unnecessary punctuations and numbers except letters
             and replacing removed words with space.
            """

            review = re.sub('[^a-zA-z]', ' ', data_set['Review'][i])

            review = review.lower()
            review = review.split()

            """
            Loop through all words and keep those which are not in stopwords list.
            set is much faster than a list and is considered when the review is very large eg.
             an article, a book.
            """
            ps = PorterStemmer()

            review = [ps.stem(word) for word in review
                      if word not in set(stopwords.words('english'))]

            review = ' '.join(review)

            self.corpus.append(review)
        return self.corpus

    ################################################################

    @staticmethod
    def container(data_set, clean_data_set):
        
        """
        Container is a Bag of Words Model
        Bag of Words Model is a sparse matrix where

            1- Each row is the review and
            2- Each column is a unique word from the reviews.

        Tokenizing - process of taking all unique words of reviews and creating columns for each word.
        Since this a problem of classification we have dependent and independent variables and each
        unique word/column is like an independent variable and the review(good/bad) depends on these

        max_features keeps most frequent words and removes least frequent words (extra cleaning)
        max_feature reduces;

            1- sparsity,
            2- increases precision,
            3- better learning and
            4- hence better prediction

        :param data_set:
        :param clean_data_set:

        :return: X(input), y(output)
        """

        cv = CountVectorizer(max_features=1500)
        x = cv.fit_transform(clean_data_set).toarray()

        y = data_set.iloc[:, 1].values
        return x, y

    ################################################################

    @staticmethod
    def classification(x, y):

        """
        splitting the clean data into

            1- Training data set
            2- Test data set

        :param x: input
        :param y: output

        :return: x_train, x_test, y_train, y_test
        """

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

        return x_train, x_test, y_train, y_test

    ################################################################

    @staticmethod
    def model(trained_x, trained_y, tested_x):

        """
        Fitting the predictive model for processed data
        RandomForestClassifier can be used for predictions

        :param trained_x:
        :param trained_y:
        :param tested_x:

        :return: y_prediction (estimated value)
        """

        try:

            classifier = GaussianNB()
            classifier.fit(trained_x, trained_y)
            y_prediction = classifier.predict(tested_x)
            return y_prediction

        except Exception as ex:
            print('Model:', ex)

    ################################################################

    @staticmethod
    def predictive_estimation(tested_y, predictive_y):

        """

        To know the Accuracy of your model, confusion matrix is
        better choice.

        Confusion Matrix = ( 2 X 2 )

        :param tested_y:
        :param predictive_y:
        :return: cm

        It value yield 4 types of result in matrix form, which are;

        TRUE POSITIVE : measures the proportion of actual positives that are correctly identified.
        TRUE NEGATIVE : measures the proportion of actual positives that are not correctly identified.
        FALSE POSITIVE : measures the proportion of actual negatives that are correctly identified.
        FALSE NEGATIVE : measures the proportion of actual negatives that are not correctly identified.

        """

        cm = confusion_matrix(tested_y, predictive_y)
        return cm

    ################################################################

    def main_processing(self, file_name):

        start_time = int(round(time.time() * 1000))
        print('Start Time:', start_time)

        fetched = PredictiveAnalysis.fetch_data(file_name)
        values = self.clean_data(fetched)

        x_input, y_output = PredictiveAnalysis.container(data_set=fetched, clean_data_set=values)
        train_x, test_x, train_y, test_y = self.classification(x_input, y_output)
        estimation = PredictiveAnalysis.model(train_x, train_y, test_x)
        result = PredictiveAnalysis.predictive_estimation(test_y, estimation)
        print('-------------------------------')
        print('Estimation:\n', result)
        print('-------------------------------')

        end_time = int(round(time.time() * 1000))
        print('End Time: ', end_time)
        print('\nTime Efficiency: ', (end_time - start_time) / 1000)
