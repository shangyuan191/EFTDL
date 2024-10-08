import json
import time
import random
# import dgl
import ast
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
# from sklearn.metrics.pairwise import cosine_similarity
# from catboost import CatBoostClassifier, CatBoostRegressor
import pickle
import os
import shutil

from datasets import load_dataset
# processing huggingface small datasets
import os

def gen_dataset_zip_file(path,dataset_name,train_data,val_data,test_data,task_type):
    info_dic={}
    info_dic['name']=dataset_name
    info_dic['basename']=dataset_name
    info_dic['split']=0
    info_dic['task_type']=task_type
    info_dic['n_num_features']=train_data['X_num'].shape[1] if train_data['X_num'] is not None else 0
    info_dic['n_cat_features']=train_data['X_cat'].shape[1] if train_data['X_cat'] is not None else 0
    info_dic['train_size']=train_data['y'].shape[0]
    info_dic['val_size']=val_data['y'].shape[0]
    info_dic['test_size']=test_data['y'].shape[0]
    with open(f'{path}/info.json','w',encoding='utf-8') as f:
        json.dump(info_dic,f,indent=4,ensure_ascii=False)

    # with open(f'{path}/info.json','r',encoding='utf-8') as f:
    #     info_dic=json.load(f)


    if train_data['X_num'] is not None:
        numpy_array = train_data['X_num'].to_numpy()
        np.save(f'{path}/N_train.npy', numpy_array)

    if train_data['X_cat'] is not None:
        numpy_array = train_data['X_cat'].to_numpy()
        np.save(f'{path}/C_train.npy', numpy_array)

    if train_data['y'] is not None:
        numpy_array = train_data['y']
        np.save(f'{path}/y_train.npy', numpy_array)

    if val_data['X_num'] is not None:
        numpy_array = val_data['X_num'].to_numpy()
        np.save(f'{path}/N_val.npy', numpy_array) 
    
    if val_data['X_cat'] is not None:
        numpy_array = val_data['X_cat'].to_numpy()
        np.save(f'{path}/C_val.npy', numpy_array)

    if val_data['y'] is not None:
        numpy_array = val_data['y']
        np.save(f'{path}/y_val.npy', numpy_array)

    if test_data['X_num'] is not None:
        numpy_array = test_data['X_num'].to_numpy()
        np.save(f'{path}/N_test.npy', numpy_array)
    
    if test_data['X_cat'] is not None:
        numpy_array = test_data['X_cat'].to_numpy()
        np.save(f'{path}/C_test.npy', numpy_array)
    
    if test_data['y'] is not None:
        numpy_array = test_data['y']
        np.save(f'{path}/y_test.npy', numpy_array)


    # 要壓縮的資料夾路徑
    folder_path = path  # 請替換成你的資料夾路徑
    # 壓縮檔的儲存路徑（不包括檔案後綴）
    output_path = f"{path}/{dataset_name}"  # 請替換成你想儲存的檔名

    # 壓縮成 zip 格式
    shutil.make_archive(output_path, 'zip', folder_path)




def create_directory_if_not_exists(directory_path: str):
    # 判定資料夾是否存在
    if not os.path.exists(directory_path):
        # 若不存在，則創建資料夾
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def get_dataset(dataset_type):
    # def data_process(dataset):
    #     data={"binclass":{},"multiclass":{},"regression":{}}
    #     for table_name, table, task in tqdm(zip(dataset['dataset_name'], dataset['table'], dataset['task']), total=len(dataset['dataset_name'])):
    #         table = ast.literal_eval(table)
    #         data[task][table_name] = {
    #             'X_num': None if not table['X_num'] else pd.DataFrame.from_dict(table['X_num']),
    #             'X_cat': None if not table['X_cat'] else pd.DataFrame.from_dict(table['X_cat']),
    #             'y': np.array(table['y']),
    #             'y_info': table['y_info'],
    #             'task': task,
    #         }
    #     return data


    # datasets=load_dataset('jyansir/excelformer',dataset_type)
    # train_data,val_data,test_data=datasets['train'].to_dict(),datasets['val'].to_dict(),datasets['test'].to_dict()
    # with open(f'{dataset_type}_train_data.json','w',encoding='utf-8') as f:
    #     json.dump(train_data,f,indent=4,ensure_ascii=False)
    # with open(f'{dataset_type}_val_data.json','w',encoding='utf-8') as f:
    #     json.dump(val_data,f,indent=4,ensure_ascii=False)
    # with open(f'{dataset_type}_test_data.json','w',encoding='utf-8') as f:
    #     json.dump(test_data,f,indent=4,ensure_ascii=False)

    # with open(f'{dataset_type}_train_data.json','r',encoding='utf-8') as f:
    #     train_data=json.load(f)

    # with open(f'{dataset_type}_val_data.json','r',encoding='utf-8') as f:
    #     val_data=json.load(f)

    # with open(f'{dataset_type}_test_data.json','r',encoding='utf-8') as f:  
    #     test_data=json.load(f)

    # print(train_data['task'])
    # train_data = data_process(train_data)
    # val_data = data_process(val_data)
    # test_data = data_process(test_data)
    
    # with open(f'{dataset_type}_datasets_info.txt','w') as f:
    #     for task_type in train_data.keys():
    #         f.write(f"{task_type}\n")

        
    #         for dataset_name in train_data[task_type].keys():
    #             f.write(f"{dataset_name}\n")
    #             train_size=train_data[task_type][dataset_name]['y'].shape[0]
    #             val_size=val_data[task_type][dataset_name]['y'].shape[0]
    #             test_size=test_data[task_type][dataset_name]['y'].shape[0]
    #             total_size=train_size+val_size+test_size
    #             num_feature_num=train_data[task_type][dataset_name]['X_num'].shape[1] if train_data[task_type][dataset_name]['X_num'] is not None else 0
    #             cat_feature_num=train_data[task_type][dataset_name]['X_cat'].shape[1] if train_data[task_type][dataset_name]['X_cat'] is not None else 0
    #             total_feature_num=num_feature_num+cat_feature_num
    #             f.write(f"total_size : {total_size}\n")
    #             f.write(f"train_size : {train_size}\n")
    #             f.write(f"val_size : {val_size}\n")
    #             f.write(f"test_size : {test_size}\n")
    #             f.write(f"train_size/total_size : {train_size/total_size}\n")
    #             f.write(f"val_size/total_size : {val_size/total_size}\n")
    #             f.write(f"test_size/total_size : {test_size/total_size}\n")
    #             f.write(f"num_feature_num : {num_feature_num}\n")
    #             f.write(f"cat_feature_num : {cat_feature_num}\n")
    #             f.write(f"total_feature_num : {total_feature_num}\n")
    #             f.write("\n")


    # with open(f'{dataset_type}_preprocessed_train_data.pkl', 'wb') as f:
    #     pickle.dump(train_data, f)

    # with open(f'{dataset_type}_preprocessed_val_data.pkl', 'wb') as f:
    #     pickle.dump(val_data, f)

    # with open(f'{dataset_type}_preprocessed_test_data.pkl', 'wb') as f:
    #     pickle.dump(test_data, f)

    with open(f'{dataset_type}_preprocessed_train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    with open(f'{dataset_type}_preprocessed_val_data.pkl', 'rb') as f:
        val_data = pickle.load(f)   
    with open(f'{dataset_type}_preprocessed_test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    

    for task_type in train_data.keys():
        print(f"{task_type}")

    
        for dataset_name in train_data[task_type].keys():
            print(f"{dataset_name}")
            path=f"../data/{dataset_type}_datasets/{dataset_name}"
            # create_directory_if_not_exists(path)
            # train_size=train_data[task_type][dataset_name]['y'].shape[0]
            # val_size=val_data[task_type][dataset_name]['y'].shape[0]
            # test_size=test_data[task_type][dataset_name]['y'].shape[0]
            # total_size=train_size+val_size+test_size
            # num_feature_num=train_data[task_type][dataset_name]['X_num'].shape[1] if train_data[task_type][dataset_name]['X_num'] is not None else 0
            # cat_feature_num=train_data[task_type][dataset_name]['X_cat'].shape[1] if train_data[task_type][dataset_name]['X_cat'] is not None else 0
            # total_feature_num=num_feature_num+cat_feature_num
            # print(f"total_size : {total_size}")
            # print(f"train_size : {train_size}")
            # print(f"val_size : {val_size}")
            # print(f"test_size : {test_size}")
            # print(f"train_size/total_size : {train_size/total_size}")
            # print(f"val_size/total_size : {val_size/total_size}")
            # print(f"test_size/total_size : {test_size/total_size}")
            # print(f"num_feature_num : {num_feature_num}")
            # print(f"cat_feature_num : {cat_feature_num}")
            # print(f"total_feature_num : {total_feature_num}")
            # print()
            gen_dataset_zip_file(path,dataset_name,train_data[task_type][dataset_name],val_data[task_type][dataset_name],test_data[task_type][dataset_name],task_type)





if __name__=="__main__":
    dataset_types=['small','large']
    for dataset_type in dataset_types:
        get_dataset(dataset_type)