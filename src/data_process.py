import pandas as pd
import re
import os
from clean_text import CleanText, RemoveNonEnglish


def count_total_entries(): 
    """
    Get the total number of entries 
    """
    pass

def read_data(path_to_data, delimiter='\t'): 
    """
    Get the DataFrame of the labeled data. 
    """
    try: 
        df = pd.read_csv(path_to_data, delimiter=delimiter, encoding='utf8')
    except UnicodeDecodeError: 
        try: 
            df = pd.read_csv(path_to_data, delimiter=delimiter, encoding='gbk')
        except UnicodeDecodeError: 
            raise

    return df


def remove_missing_and_blank_values(df): 
    """
    Remove the missing values and blank values. 
    """
    df.dropna(inplace=True)

    blanks = []

    for index, text, _, _ in df.itertuples():
        if type(text) == str and text.isspace():
            blanks.append(index)

    df.drop(labels=blanks, inplace=True)
    return df


def read_labeled_data(path_to_data):
    """
    Read the data based on a given path.
    Here we have to handle various cases. 
    """

    filename = os.path.basename(path_to_data)
    suffix = os.path.splitext(filename)[1][1:]
    
    delimiter = ',' if suffix == 'csv' else '\t'

    df = read_data(path_to_data=path_to_data, delimiter=delimiter)

    df = remove_missing_and_blank_values(df)

    df['sentiment_for_product'] = df['sentiment_for_product'].astype('str')
    df['sentiment_for_video'] = df['sentiment_for_video'].astype('str')

    mapping = {
        '1.0': 'positive',
        '0.0': 'neutral',
        '-1.0': 'negative',
        '1': 'positive',
        '0': 'neutral',
        '-1': 'negative',
        'positive': 'positive', 
        'neutral': 'neutral',
        'negative': 'negative'
    }

    # Apply the mapping to the specified column
    df['sentiment_for_product'] = df['sentiment_for_product'].map(mapping)
    df['sentiment_for_video'] = df['sentiment_for_video'].map(mapping)

    df['sentiment_for_product'] = df['sentiment_for_product'].str.strip()
    df['sentiment_for_video'] = df['sentiment_for_video'].str.strip()

    # df['sentiment_for_product'] = df['sentiment_for_product'].astype('str')
    # df['sentiment_for_video'] = df['sentiment_for_video'].astype('str')

    label2id = {"negative": int(0), "neutral": int(1), "positive": int(2)}
    
    df['sentiment_for_product'] = df['sentiment_for_product'].map(label2id)
    df['sentiment_for_video'] = df['sentiment_for_video'].map(label2id)

    df['sentiment_for_product'] = df['sentiment_for_product'].astype('Int64')
    df['sentiment_for_video'] = df['sentiment_for_video'].astype('Int64')

    return df

def read_all_labeled_data(): 
    """
    Read all data files under './data/comment_labeled/' directory.   
    """
    dir = './data/comment_labeled'
    
    files = os.listdir(dir)
    df_list = []

    for file in files: 
        df = read_labeled_data(os.path.join(dir, file))
        df_list.append(df)

    combined_df = pd.concat(df_list)

    combined_df = combined_df.reset_index(drop=True)

    return combined_df



def save_df(df, file_name): 
    """
    Save dataframe as a csv file.
    """
    dir = './data'

    file_path = os.path.join(dir, file_name)
    
    # Save the DataFrame as a CSV file
    df.to_csv(file_path, index=False, sep='\t')

def load_df(file_name): 
    """
    Load the processed dataset based on the filename. 
    """
    dir = './data'

    file_path = os.path.join(dir, file_name)

    df = read_data(file_path, delimiter='\t')

    df['sentiment_for_product'] = df['sentiment_for_product'].astype('Int64')
    df['sentiment_for_video'] = df['sentiment_for_video'].astype('Int64')

    df = df.reset_index(drop=True)

    return df
    

        
def check_df(df): 
    print(f'The size of dataset: {df.shape}', end='\n\n')
    print(df.head(), end='\n\n')
    print(df['sentiment_for_product'].value_counts(), end='\n\n')
    print(df['sentiment_for_video'].value_counts(), end='\n\n')

if __name__ == '__main__': 
    df = load_df('combined_text_clean.csv')
    # df = RemoveNonEnglish(df).get_df()
    # df = CleanText(df, do_stemming=False).get_df()
    check_df(df)
    # save_df(df, 'combined_text_clean.csv')

