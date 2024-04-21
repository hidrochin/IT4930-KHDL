import pandas as pd

data_files = [
    'data/dth-tiki/result/comments_data_ncds.csv',
    'data/nhacua-tiki/result/comments_data_ncds.csv',
    'data/laptop-tiki/result/comments_data_ncds.csv',
    'data/book-tiki/result/comments_data_ncds.csv'
]

dfs = []  
df1 = []

for file in data_files:
    df = pd.read_csv(file)
    df_negative = df[(df['rating'] <= 3) & df['content'].notnull()]
    dfs.append(df_negative)

merged_df = pd.concat(dfs)

merged_df = merged_df.drop('customer_id', axis=1)

merged_df = merged_df.drop('id', axis=1)

merged_df = merged_df.drop('title', axis=1)

non_duplicates_df = merged_df.drop_duplicates(subset='content')

non_duplicates_df.to_json('data/data-negative.json', orient='records', force_ascii=False)

for file in data_files:
    df = pd.read_csv(file)
    df_positive = df[(df['rating'] > 3) & df['content'].notnull()]
    df1.append(df_positive)

merged_df = pd.concat(df1)

merged_df = merged_df.drop('customer_id', axis=1)

merged_df = merged_df.drop('id', axis=1)

merged_df = merged_df.drop('title', axis=1)

non_duplicates_df = merged_df.drop_duplicates(subset='content')

non_duplicates_df.to_json('data/data-positive.json', orient='records', force_ascii=False)
