# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys 

#database
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    #merge    
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    # create a dataframe of the 36 individual category columns

    categories = df['categories'].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [word[:-2] for word in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    '''Convert category values to just numbers 0 or 1.
    Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). For example, related-0 becomes 0, related-1 becomes 1. Convert the string to a numeric value.
    You can perform normal string actions on Pandas Series, like indexing, by including .str after the Series. You may need to first convert the Series to be of type string, which you can do with astype(str).
    '''
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda word: word[-1:])
    
    # convert column from string to numeric
    #TODO 

    #Replace categories column in df with new category columns.¶
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],axis =1)

    # drop duplicates
    df.drop_duplicates(keep=False, inplace=True)
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, if_exists='replace', index=False)  
    #print('saved to ' + filename)

## continue with load data
def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
    
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')

        
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

