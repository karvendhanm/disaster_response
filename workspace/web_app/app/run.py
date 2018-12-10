import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from langdetect import detect
import pickle
from sklearn.base import BaseEstimator, TransformerMixin


app = Flask(__name__)

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

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)


model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals

 # Original code comment begins here
#     genre_counts = df.groupby('genre').count()['message']
#     genre_names = list(genre_counts.index)

#     # create visuals
#     # TODO: Below is an example - modify to create your own visuals
#     graphs = [
#         {
#             'data': [
#                 Bar(
#                     x=genre_names,
#                     y=genre_counts
#                 )
#             ],

#            'layout': {
#                 'title': 'Distribution of Message Genres',
#                 'yaxis': {
#                     'title': "Count"
#                 },
#                 'xaxis': {
#                     'title': "Genre"
#                 }
#             }
#         }
#     ]
# original code comment ends here

## my code begins here
    genre_counts = df.groupby('genre').count()['message']
    genre_counts_list = genre_counts.tolist()
    genre_names = genre_counts.index
    genre_names_list = genre_names.tolist()

    graph_one = []
    graph_one.append(
       Bar(
            x = genre_names_list,
            y = genre_counts_list,
        )
    )

    layout_one = dict(title = 'Distribution of Message Genres',
                      xaxis = dict(title = 'Genre',),
                      yaxis = dict(title = 'Count'),
                     )

    basic_necessity_list = ['water', 'shelter', 'food']
    perc_list = []
    for col in basic_necessity_list:
        perc_list.append(round(((df[col].sum()/len(df)) * 100),2))

    graph_two = []
    graph_two.append(
       Bar(
            x = basic_necessity_list,
            y = perc_list,
        )
    )

    layout_two = dict(title = 'Texts on Basic Necessities as a percentage of the Whole',
                      xaxis = dict(title = 'Type of Basic Necessity',),
                      yaxis = dict(title = 'Percentage of Total'),
                     )

    figures = []
    figures.append(dict(data = graph_one, layout = layout_one))
    figures.append(dict(data = graph_two, layout = layout_two))

    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html',ids=ids,figuresJSON=figuresJSON)

## my code ends here

#     # encode plotly graphs in JSON
#     ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
#     graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

#     # render web page with plotly graphs
#     return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
