# import torch
# print(torch.cuda.is_available())  # 應該返回 True
# print(torch.cuda.current_device())  # 應該返回 GPU ID
from datasets import load_dataset
import pandas as pd
import numpy as np

# process train split, similar to other splits
data = {}
datasets = load_dataset('jyansir/excelformer') # load 96 small-scale datasets in default
# datasets = load_dataset('jyansir/excelformer', 'large') # load 21 large-scale datasets with specification
dataset = datasets['train'].to_dict()
for table_name, table, task in zip(dataset['dataset_name'], dataset['table'], dataset['task']):
    data[table_name] = {
        'X_num': None if not table['X_num'] else pd.DataFrame.from_dict(table['X_num']),
        'X_cat': None if not table['X_cat'] else pd.DataFrame.from_dict(table['X_cat']),
        'y': np.array(table['y']),
        'y_info': table['y_info'],
        'task': task,
    }
