from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pandas as pd
import re
import csv
import pathlib

def remove_html_tags(text):
    clean_text = re.sub('<.*?>', '', text)
    return clean_text

# Sentence BERTモデルのロード
model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

current_dir = pathlib.Path(__file__).parent

# CSVファイルの読み込み
data = pd.read_csv(current_dir / '元データ/論文小樽－帯広（2023~2018） .csv')
# 3列目の文章データを取得
text_data = data.iloc[1:, 2].dropna().values.tolist()
text_data = [remove_html_tags(text) for text in text_data]
affi_data = data.iloc[:, 1].dropna().values.tolist()
name_data = data.iloc[:, 0].dropna().values.tolist()

# 文書のエンコード
encoded_inputs = tokenizer(text_data, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    embeddings = model(**encoded_inputs).last_hidden_state[:, 0, :]

output_file = current_dir / "類似度処理前ファイル/小樽-帯広_ver2.csv"

with open(output_file, 'w', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(['A_Nunber','A_Name', 'A_Affi', 'A_Text', 'B_Nunber','B_Name', 'B_Affi', 'B_Text', 'Similarity'])
    
    for i in range(len(text_data)):
        for j in range(i + 1, len(text_data)):
            if i != j:
                if affi_data[i] != affi_data[j]:
                    similarity = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))
                    if similarity[0][0] >= 0.3: 
                        # カンマを取り除く
                        text_a = re.sub(r',', '', text_data[i])
                        text_b = re.sub(r',', '', text_data[j])
                        row_data = ([i,name_data[i], affi_data[i], text_a, j,name_data[j], affi_data[j], text_b, similarity])
                        f.write(','.join(map(str, row_data)) + '\n')

print("終了")

'''
2024~2010
if not ((0 <= i <= 4446 and 0 <= j <= 4446) or (4447 <= i <= 11247 and 4447 <= j <= 11247) or (11248 <= i <= 13109 and 11249 <= j <= 13109)):
'''
'''
2023~2018
if not ((0 <= i <= 728 and 0 <= j <= 728) or (729 <= i <= 3286 and 729 <= j <= 3286) or (3287 <= i <= 4635 and 3287 <= j <= 4635)):
'''