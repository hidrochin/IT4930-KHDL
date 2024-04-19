import requests
import time
import pandas as pd
import random
from tqdm import tqdm

cookies = {'_trackity':'f42ea536-5842-867f-587e-1a47852be7b4',
           'TOKENS':'{%22access_token%22:%22kax4WoQfzPUY2mhKri1sFbCJVDe0X8w9%22%2C%22expires_in%22:157680000%2C%22expires_at%22:1829937876436%2C%22guest_token%22:%22kax4WoQfzPUY2mhKri1sFbCJVDe0X8w9%22}',
           '_ga':'GA1.2.1017622613.1672257935',
           'gid': 'GA1.2.656675261.1672257935',
           'tiki_client_id': '1017622613.1672257935',
           '_gat': '1'
}
header = {'accept': 'application/json, text/plain, /',
          'accept-encoding': 'gzip, deflate, br',
          'accept-language': 'vi-VN,vi,;q=0.8,en-US,en;q=0.6',
          'referer': 'https://tiki.vn/dien-thoai-may-tinh-bang/c1789',
          'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
          'x-guest-token': 'kax4WoQfzPUY2mhKri1sFbCJVDe0X8w9',
          'TE': 'Trailers',
          }

params = (
    ('platform', 'web'),
    ('spid', 273947988)
)

def parser_product(json):
    d = dict()
    d['id'] = json.get('id')
    d['category'] = json.get('categories').get('name')
    d['name'] = json.get('name')
    d['rating_avg'] = json.get('rating_average')
    d['num_review'] = json.get('review_count')
    d['review_text'] = json.get('review_text')
    spec = json.get('specifications')[0].get('attributes')
    for i in spec:
        d[i.get('name')] = i.get('value')

    return d

df_id = pd.read_csv('data/dth-tiki/result/product_id_ncds.csv')
p_ids = df_id.id.to_list()
print(p_ids)
result = []
for idx, pid in  enumerate(p_ids):
    response = requests.get('https://tiki.vn/api/v2/products/{}'.format(pid), headers=header, params=params, cookies=cookies)
    if response.status_code == 200:
        if (
                response.status_code != 204 and
                response.headers["content-type"].strip().startswith("application/json")
        ):
            try:
                result.append(response.json())
            except ValueError:
                print(idx, pid)
        # decide how to handle a server that's misbehaving to this extent
    else:
        print('[WARNING] request failed')
    if idx % 5 == 0:
        print(idx)
        df = pd.DataFrame(result)
        df.to_csv("data/dth-tiki/result/crawled_data_ncds.csv",index=False)
        print('saving successfully')
df_product = pd.DataFrame(result)
df_product.to_csv('data/dth-tiki/result/crawled_data_ncds.csv', index=False)