import sys
import pandas as pd
import nltk
nltk.download(['stopwords','punkt','wordnet','averaged_perceptron_tagger'])
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

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    conn = engine.connect()
    df = pd.read_sql_table(table_name='disaster_response', con=conn)
    X = df['message']
    Y = df.loc[:, 'related':]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    stop_words = stopwords.words("english")
    lemmat = WordNetLemmatizer()

    cust_stop_words_list = ['hello', 'good morning', 'good evening', 'good afternoon', 'sir', 'madam']
    for words in cust_stop_words_list:
        stop_words.append(words)

    clean_tokens = []
    text = text.lower()
    text = re.sub(r"(https*:|www\.)\S+", r"urlplaceholder", text)
    text = re.sub(r"[^a-z]", " ", text)
    tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
    tokens = [token for token in tokens if token not in stop_words]
    for tok in tokens:
        clean_tok = lemmat.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(forest, n_jobs=-1))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for idx, column in enumerate(category_names):
        print('the category being evaluated is: --', column)
        print(classification_report(Y_test[column], Y_pred[:, idx]))


def save_model(model, model_filepath):   
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