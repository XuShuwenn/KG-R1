import pandas as pd
import os

file_path = '/netcache/yuanchenhao/KG-R1/data_kg/cwq_kgqa_agent_format/train.parquet'
if os.path.exists(file_path):
    df = pd.read_parquet(file_path)
    print(f"Columns: {df.columns}")
    if 'prompt' in df.columns:
        print("First prompt sample:")
        print(df['prompt'].iloc[0])
    elif 'messages' in df.columns:
        print("First messages sample:")
        print(df['messages'].iloc[0])
else:
    print(f"File not found: {file_path}")
