
# import statements
import sys
import re
import pickle
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#database
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine

#constants
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_database(database_filename):
    #engine = create_engine('sqlite:///DisasterResponse.db')
    df =pd.read_sql_table('messages', 'sqlite:///' + database_filename)  
    X = df.message.values
    Y = df.medical_help
    category_names = df.columns[4:]
    #Y = df.drop(labels = ['id','message','original','genre'], axis=1).values
   
    return X, Y, category_names
  

def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

        # Remove punctuation characters
    #text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() 
        clean_tokens.append(clean_tok)
     
    return clean_tokens

def remove_stopwords(tokens):
    new_tokens =[]
    for token in tokens:
        token= [word for word in token if word not in stopwords.words("english")]
        new_tokens.append(token)
    return tokens
# def display_results(y_test, y_pred):
#     labels = np.unique(y_pred)
#     confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
#     accuracy = (y_pred == y_test).mean()

#     print("Labels:", labels)
#     print("Confusion Matrix:\n", confusion_mat)
#     print("Accuracy:", accuracy)





def build_model(X,y):
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),# strings to token integer counts
    ('tfidf', TfidfTransformer()), # integer counts to weighted TF-IDF scores
    ('clf', RandomForestClassifier()), # train on TF-IDF vectors w/ Naive Bayes classifier
    ])
    '''
    Split data into train and test sets
    Train pipeline
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
    # train classifier
    pipeline.fit(X_train,y_train)
    # evaluate all steps on test set
    predicted = pipeline.predict(X_test)
    return pipeline, predicted, X_test, y_test

def test_model(ytest, predictions) :
    print(confusion_matrix(ytest, predictions))
    print(classification_report(ytest, predictions)) 

def evaluate_model(model, X_test, Y_test, category_names):
    pass    

def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))
 
def main():
    if len(sys.argv) == 3:

        database_filepath, model_filepath = sys.argv[1:]

        ##PIPELINE CODE
        print('Loading database...')
        #df = load_data(messages_filepath, categories_filepath)
        X, y, category_names = load_database(database_filepath)

        print('Tokenizing...')
        #tokenize(message for message in X)
        #X = [tokenize(message) for message in X]
        print('Building model')
        pipeline, predicted, X_test, y_test = build_model(X,y)
        
        print('Testing model')
        test_model(y_test,predicted)

        print('Evaluating model...')
        evaluate_model(pipeline, X_test, y_test, category_names)

        print('saving model')
        save_model(pipeline, model_filepath)

    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()  

