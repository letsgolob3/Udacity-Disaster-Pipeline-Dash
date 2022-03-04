import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    messages_filepath - string for location of messages data
    categories_filepath - string for location of categories data
    
    OUTPUT
    df - merged dataframe of messages and categories
    '''
    
    messages = pd.read_csv(f'{messages_filepath}')
    
    categories = pd.read_csv(f'{categories_filepath}')
    
    # Merging categories and messages into single dataframe
    df = pd.merge(categories,messages,on='id',how='inner')    
    
    return df


def clean_data(df):
    '''
    INPUT
    df - dataframe of messages with categories. Output from load_data
    
    OUTPUT
    df - cleaned dataframe of messages and categories
    '''
    
    # Splitting categories column into multiple columns by the ;
    categories = df['categories'].str.split(';',expand=True)
    
    # Rename the categories columns 
    row = categories.iloc[0,:]
    category_colnames = row.str.split('-').str[0]
    
    categories.columns = category_colnames
    

    # Convert values in each category column to binary 1/0
    for column in categories:

        categories[column] = categories[column].astype(str)
        categories[column] = categories[column].str[-1]
    
    categories[column] = pd.to_numeric(categories[column],downcast='integer')
    
    # Remove categories column from df
    df.drop(columns=['categories'],inplace=True)

    # Concatenate original df with expanded categories
    df = pd.concat([df,categories],axis=1)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)    
    
    return df


def save_data(df, database_filename):
    '''
    INPUT
    df - cleaned dataframe of messages and categories. Output from clean_data
    
    OUTPUT
    None - function does not return output but the tablename is saved locally
    '''
    
    engine = create_engine(f'sqlite:///{database_filename}')
    
    df.to_sql('Disaster_Resp', engine, index=False, if_exists='replace')
    
    pass  


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