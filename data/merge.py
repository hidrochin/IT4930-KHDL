import pandas as pd

# Đọc các tệp Excel
df1 = pd.read_excel('data_split_1.xlsx')
df2 = pd.read_excel('data_split_2.xlsx')
df3 = pd.read_excel('data_split_3.xlsx')
df5 = pd.read_excel('data_split_5.xlsx')

# Hợp nhất DataFrames
df_merged = pd.concat([df1, df2, df3, df5], ignore_index=True)

# Lưu vào tệp Excel mới
df_merged.to_excel('data_merged.xlsx', index=False)