import sys
import pandas as pd
import nltk

nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.corpus import stopwords
import re
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import pos_tag
import pickle
from langdetect import detect

# custom transformer class. detects whether the given text is in english language or not
# using transform method
class IsEnglish(BaseEstimator, TransformerMixin):
    def detect_english(self, text):
        '''
        this function takes in a text and detects if it is in english language

        :param text: (str)the text for which language needs to be identified
        :return: (boolean) if the language of the text is english or not
        '''
        try:
            if (detect(text) == 'en'):
                return True
            else:
                return False
        except:
            return False

    def fit(self, x, y=None):
        '''
        this function just returns the instance of the class

        :param x: independent variables train dataset
        :param y: (optional) response variable of the train dataset
        :return: instance of the class
        '''
        return self

    def transform(self, X):
        '''
        takes a series of texts and returns a dataframe identifying if the text is in kanguage english or not

        :param X: independent variables train dataset
        :return: dataframe of one column with boolean identifying if the text in that row is english or not
        '''
        X_tagged = pd.Series(X).apply(lambda x: self.detect_english(x))
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    '''

    :param database_filepath:
    :return:
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    conn = engine.connect()
    df = pd.read_sql_table(table_name='disaster_response', con=conn)
    X = df['message']
    Y = df.loc[:, 'related':]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''
    takes a text and performs cleaning operations and then tokenizes the text

    :param text: the text that needs to be cleaned and tokenized
    :return: cleaned tokens
    '''

    # common, often repeated, inconsequential words are removed using stopwords as the removal of
    # those words affords ml algorithms better clarity and meaty words to focus on.
    stop_words = stopwords.words("english")

    # lemmatizer cuts the words to its root so that ml algorithms have fewer words
    # to focus on
    lemmat = WordNetLemmatizer()

    # removing few other words that are just greetings but doesn't have any siginificance
    # with respect to the task at hand. removing them by just adding those words to stopwords list
    cust_stop_words_list = ['hello', 'good morning', 'good evening', 'good afternoon', 'sir', 'madam']
    for words in cust_stop_words_list:
        stop_words.append(words)

    clean_tokens = []

    # all text is converted to lowercase so that 'S' and 's' doesn't seem different to ml
    # and all web urls are removed and a place holder is left in its place
    text = text.lower()
    text = re.sub(r"(https*:|www\.)\S+", r"urlplaceholder", text)

    # here all characters except a to z are removed, cause punctuations, numbers could be
    # different for what is essentially the same text. For instance,
    # text 'a borrowed $1000 from b' and 'c borrowed $3000 from d' are essentially the same
    #  and talks about borrowing but the numbers involved are different. removing the
    # numbers involved will make both the texts look the same to ml
    text = re.sub(r"[^a-z]", " ", text)

    # the text is tokenised for countvectorizer to create BoW matrix.
    tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
    tokens = [token for token in tokens if token not in stop_words]
    for tok in tokens:

        clean_tok = lemmat.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    this function returns a model after constructing a pipeline and choosing parameters using grid search
    :return: model to classify the text
    '''

    forest = RandomForestClassifier(n_estimators=100, random_state=42)

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            # feature union is used to add additional feature created from original data and not in the
            # sequence of the pipeline
            ('isenglish', IsEnglish())
        ])),
        # MultiOutputClassifier is used because the response variable is a multiple class output
        ('clf', MultiOutputClassifier(forest, n_jobs=-1))
    ])

    # parameters for gridsearch is constructed
    parameters = {
        'features__text_pipeline__vect__decode_error': ['strict', 'ignore'],
        'features__text_pipeline__vect__ngram_range': [(1, 1), (1, 2)],
        'features__text_pipeline__vect__max_df': [0.7, 0.9],
        'features__text_pipeline__tfidf__sublinear_tf': [True, False]
    }

    # gridsearch is constructed using pipelines and parameters
    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    this function takes a model and then predicts the response variable using test dataset
    and then prints the accuracy, precision and f1 score of the model

    :param model:
    :param X_test: independent variable of the test dataset
    :param Y_test: response variable of the test dataset
    :param category_names: column names of the response variables
    :return: None
    '''

    # since the model is a gridsearchcv the best_estimator is taken out for prediction
    model = model.best_estimator_
    Y_pred = model.predict(X_test)

    # precision, accuracy and the f1 score of the model is printed out
    for idx, column in enumerate(category_names):
        print('the category being evaluated is: --', column)
        print(classification_report(Y_test[column], Y_pred[:, idx]))


def save_model(model, model_filepath):
    '''
    this function takes in a model and stores them as a pickle in the given filepath
    :param model: model to be stored
    :param model_filepath: the filepath of the pickle file
    :return: None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()