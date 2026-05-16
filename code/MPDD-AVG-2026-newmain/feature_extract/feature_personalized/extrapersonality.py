"""
用 RoBERTa-large 对个性化描述文本编码，生成 1024 维嵌入向量
输入：descriptions.csv（ID, Descriptions）
输出：descriptions_embeddings_with_ids.npy
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel

# ── 配置 ──────────────────────────────────────────────
INPUT_CSV   = r'D:\postgraduate-1\restruct_MPDD\MPDD\descriptions.csv'
OUTPUT_FILE = r'D:\postgraduate-1\restruct_MPDD\MPDD\descriptions_embeddings_with_ids.npy'
MODEL_NAME  = 'roberta-large'
# ──────────────────────────────────────────────────────

df = pd.read_csv(INPUT_CSV)
print(f'共 {len(df)} 条描述')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

print('加载 RoBERTa-large 模型...')
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()
print('模型加载完成')

embeddings_with_ids = []

with torch.no_grad():
    for _, row in df.iterrows():
        pid = str(int(row['ID']))
        desc = row['Descriptions']

        encoded = tokenizer(desc, return_tensors='pt', padding=True,
                            truncation=True, max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        output = model(**encoded)
        embedding = output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        embeddings_with_ids.append({'id': pid, 'embedding': embedding})
        print(f'  ID={pid}, shape={embedding.shape}')

np.save(OUTPUT_FILE, embeddings_with_ids, allow_pickle=True)
print(f'\n完成，已保存到 {OUTPUT_FILE}')
