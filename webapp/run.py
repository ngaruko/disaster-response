import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar 
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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
df = pd.read_sql_table('messages', engine)
#print(df.columns[4:])
#
# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # # extract data needed for visuals
    # # TODO: Below is an example - modify to extract data for your own visuals
    # genre_counts = df.groupby('genre').count()['message']
    # genre_names = list(genre_counts.index)
    # print(genre_names)
    # category_names = df.columns[4:]
    # print(category_names)
    # category_counts =df.groupby(['request','offer','money','storm']).count()['message']

    # # create visuals
    # # TODO: Below is an example - modify to create your own visuals
    # graphs = [
    #     {
    #         'data': [
    #             Bar(
    #                 x=category_names,
    #                 y=category_counts
    #             )
    #         ],

    #         'layout': {
    #             'title': 'Distribution of Message Categories',
    #             'yaxis': {
    #                 'title': "Count"
    #             },
    #             'xaxis': {
    #                 'title': "category"
    #             }
    #         }
    #     }
    # ]
    
    # # encode plotly graphs in JSON
    # ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    # graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # # render web page with plotly graphs
    # return render_template('master.html', ids=ids, graphJSON=graphJSON)
    return "Hello Heroku Flask 0101"


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

@app.route('/plot')
def plot():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    print(genre_names)
    category_names = df.columns[4:]
    print(category_names)
    category_counts =df.groupby(['request','offer','money','storm']).count()['message']

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

def getApp():   
    return app

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)
   

if __name__ == '__main__':
    main()

