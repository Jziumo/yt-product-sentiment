import pandas as pd

def count_total_entries(): 
    """
    Get the total number of entries 
    """
    pass

def read_labeled_data(path_to_data, delimiter='\t'): 
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






if __name__ == '__main__': 
    df = read_labeled_data('./data/comment_labeled/labeled_yamaha_p_225_piano_review_better_music.txt')
    print(df['sentiment_for_video'].value_counts())
    # df = read_labeled_data('./data/comment_labeled/labeled_almost_best_phone_of_2025_vivo_x200_ultra_review_comments.csv', delimiter=',')