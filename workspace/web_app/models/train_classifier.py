import sys
import pandas as pd
import nltk
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
import pickle
nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    '''
    This function takes in the name of a sqlite database and loads the data
    in the tablename 'disaster_response' in a dataframe. Then splits
    the dataframe into independent(X) and response variables(Y)

    :param database_filepath: the name of the database
    :return: indepedent variable, dependent variable, and column names of the dependent variable
    '''

    # sqlite engine is created and data read from the database from the table 'disaster_response'
    # and stored in a dataframe
    engine = create_engine('sqlite:///'+database_filepath)
    conn = engine.connect()
    df = pd.read_sql_table(table_name='disaster_response', con=conn)

    # dataframe is split into independent and dependent variables using column names
    X = df['message']
    Y = df.loc[:, 'related':]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''
    This functions is a text preprocessor called by countvectorizer to clean the text
    before BoW matrix is formulated and then subsequently fed to
    term frequency-document frequency matrix.

    :param text: the text that needs to be cleaned for machine learning
    :return: cleaned text in the form of tokens.
    '''

    # common inconsequential words are removed using stopwords as the removal of
    # those words affords ml algorithms better clarity and meaty words to
    # focus on.
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
    # all text is converted to lowercase and all web urls are removed
    # and a place holder is left in its place
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
    this function builds a pipeline for text classification

    :return: the pipeline
    '''

    # initializing a random forest classifier
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        # MultiOutputClassifier is used as the output is a multi class output.
        ('clf', MultiOutputClassifier(forest, n_jobs=-1))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    this function evaluates the model on three parameters and prints the results.
    1) precision, 2) recall and 3) f1 score.

    :param model: the model that is to be evaluated
    :param X_test: the test set for prediction
    :param Y_test: the test sets true response variable
    :param category_names: names of the columns of the response variable
    :return: None
    '''
    Y_pred = model.predict(X_test)
    for idx, column in enumerate(category_names):
        print('the category being evaluated is: --', column)
        print(classification_report(Y_test[column], Y_pred[:, idx]))


def save_model(model, model_filepath):
    '''
    this function takes a model and saves it as a pickle in the given name

    :param model: model to be saved as a pickle
    :param model_filepath: the filepath besides the name of the pickle
    :return: None
    '''
    pickle.dump(model, open(model_filepath,'wb'))
    

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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()