import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
     This function takes in two .csv files, and merges them both on 'id' column.

    :param messages_filepath: file path of the message.csv file
    :param categories_filepath: file path of the categories.csv file
    :return: the merged dataframe
    '''

    # reading in both the .csv files and converting them to pandas dataframe
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merging both the dataframe on column 'id'
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    '''
    This function takes in an dataframe and then clean up a column called categories.
    The single column categories is spilt into 36 new columns and the original
    categories column is dropped before the newly created 36 columns are concatenated
    with the original dataframe.

    :param df: the dataframe that needs to be cleaned. must have a column called 'categories'
    :return: cleaned dataframe
    '''

    # split the contents of the column 'categories' using the delimiter ';'
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # just slicing the last character of the column and converting it into int datatype.
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)

    # dropping the original categories column and concatenating
    # the original dataframe with the newly created columns
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    '''
    This function saves a dataframe in a sql database

    :param df: The dataframe that needs to be stored in an sqlite database.
    :param database_filename: the name of the sqlite database to be created to hold the dataframe
    :return: None
    '''

    # creating a sqlite engine and storing the dataframe with the table name disaster_response
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_response', engine, index=False)  


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
