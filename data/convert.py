import pandas as pd

data_files = [
    'data/dth-tiki/result/comments_data_ncds.csv',
    'data/nhacua-tiki/result/comments_data_ncds.csv',
    'data/laptop-tiki/result/comments_data_ncds.csv',
    'data/book-tiki/result/comments_data_ncds.csv'
]

dfs = []  

for file in data_files:
    df = pd.read_csv(file)
    filtered_df = df[(df['rating'] <= 3) & df['content'].notnull()]
    dfs.append(filtered_df)

merged_df = pd.concat(dfs)

merged_df = merged_df.drop('customer_id', axis=1)

non_duplicates_df = merged_df.drop_duplicates(subset='content')

non_duplicates_df.to_csv('data/data.csv', index=False)