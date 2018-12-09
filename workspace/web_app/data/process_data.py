import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    this function takes two .csv files and merges(inner join) them on column 'id'

    :param messages_filepath: path of the message.csv file
    :param categories_filepath: path of the categories.csv file
    :return: dataframe combining both the aforementioned .csv files
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merges both messages and categories dataframe
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    '''
    takes a dataframe and splits the column 'categories' using the delimiter ';' and expands it to 36 new columns.
    then the original 'categories' column is dropped but the newly formed columns are concatenated with the orignal dataframe.
    and the dataframe is further manipulated to leave only integer values for response variable  

    :param df: dataframe that needs to be cleaned: independent and response variable put in usable form
    :return: returns the cleaned dataframe
    '''
    categories = df['categories'].str.split(';', expand=True)
    
    # the first row of the dataframe in taken 
    row = categories.iloc[0, :]
    
    # to get the column names last two characters from the first row is omitted
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # the last character is each column is sliced and the datatype converted to integer 
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)

    # original 'categories' column in dropped
    df = df.drop('categories', axis=1)
    
    # newly formed columns are concatenated with original dataframe
    df = pd.concat([df, categories], axis=1)

    # duplicate rows are dropped from the dataframe
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    '''
    this function creates a sqlite database on the 'database_filename' location and saves the dataframe 'df' 
    under the table name 'disaster_response'    

    :param df: the dataframe that needs to be saved
    :param database_filename: the filepath of the database that needs to be created
    :return: None
    '''
    
    # sqlite engine created
    engine = create_engine('sqlite:///' + database_filename)
    
    # the dataframe df is saved in the sqlite database under the table name 'disaster_response'
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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()