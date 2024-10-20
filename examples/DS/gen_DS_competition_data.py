import csv
import os
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
import numpy as np


# def read_small_datasets(filename):
#     with open(filename,'r',encoding='utf-8') as f:
#         lines=f.readlines()
#     datasets=[]
#     current_dataset=[]

#     for line in lines:
#         if "binclass" in line:
#             continue
#         if line.strip()=="" and len(current_dataset)==11:
#             datasets.append(current_dataset)
#             current_dataset=[]
#         elif line.strip():
#             current_dataset.append(line.strip())

#     if len(current_dataset)==11:
#         datasets.append(current_dataset)

#     for dataset in datasets:
#         print(dataset)
#         print()
#     print(len(datasets))

if __name__=="__main__":
    # read_small_datasets("../small_datasets_info.txt")


    # splits = {'train': 'train-small.parquet', 'val': 'val-small.parquet', 'test': 'test-small.parquet'}
    # df_train = pd.read_parquet("hf://datasets/jyansir/excelformer/" + splits["train"])
    # df_val=pd.read_parquet("hf://datasets/jyansir/excelformer/" + splits["val"])
    # df_test=pd.read_parquet("hf://datasets/jyansir/excelformer/" + splits["test"])
    # df_binclass_train=df_train[df_train['task']=='binclass']
    # df_binclass_val=df_val[df_val['task']=='binclass']
    # df_binclass_test=df_test[df_test['task']=='binclass']

    # print(df_binclass_train)
    # print(df_binclass_val)
    # print(df_binclass_test)
    with open(f'../small_preprocessed_train_data.pkl', 'rb') as f:
            train_data = pickle.load(f)
        
    with open(f'../small_preprocessed_val_data.pkl', 'rb') as f:
        val_data = pickle.load(f)   
    with open(f'../small_preprocessed_test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)

    train_data=train_data['binclass']
    val_data=val_data['binclass']
    test_data=test_data['binclass']


    directory_all=f'./all'
    directory_without_y_test=f'./Competition_data'
    if not os.path.exists(directory_all):
            os.makedirs(directory_all)
    if not os.path.exists(directory_without_y_test):
        os.makedirs(directory_without_y_test)
    folder_names=[]
    for idx,key in enumerate(list(train_data.keys())):
        print(idx,key)
        # folder_name=key.replace("[", "").replace("]", "_").replace(" ", "_").replace("(", "_").replace(")", "").replace("-", "_").replace("&","_")
        folder_names.append(key)
    folder_names=sorted(folder_names)
    for idx,key in enumerate(folder_names):
        folder_name=f"Dataset_{idx+1}"
        all = f'./all/Dataset_{idx+1}'
        without_y_test=f'./Competition_data/Dataset_{idx+1}'
        if not os.path.exists(all):
            os.makedirs(all)
        if not os.path.exists(without_y_test):
            os.makedirs(without_y_test)
        X_num_train=train_data[key]['X_num'] if train_data[key]['X_num'] is not None else pd.DataFrame()
        X_cat_train=train_data[key]['X_cat'] if train_data[key]['X_cat'] is not None else pd.DataFrame()
        y_train = pd.DataFrame(train_data[key]['y'], columns=['target'])  # 設定 y_train 的列名
        X_num_val=val_data[key]['X_num'] if val_data[key]['X_num'] is not None else pd.DataFrame()
        X_cat_val=val_data[key]['X_cat'] if val_data[key]['X_cat'] is not None else pd.DataFrame()
        y_val = pd.DataFrame(val_data[key]['y'], columns=['target'])  # 設定 y_val 的列名

        X_num_test=test_data[key]['X_num'] if test_data[key]['X_num'] is not None else pd.DataFrame()
        X_cat_test=test_data[key]['X_cat'] if test_data[key]['X_cat'] is not None else pd.DataFrame()
        y_test = pd.DataFrame(test_data[key]['y'], columns=['target'])  # 設定 y_test 的列名

        # print(f"X_num_train.shape : {X_num_train.shape if X_num_train is not None else None}")
        # print(f"X_cat_train.shape : {X_cat_train.shape if X_cat_train is not None else None}")
        # print(f"y_train.shape : {y_train.shape}")
        # print(f"X_num_val.shape : {X_num_val.shape if X_num_val is not None else None}")
        # print(f"X_cat_val.shape : {X_cat_val.shape if X_cat_val is not None else None}")
        # print(f"y_val.shape : {y_val.shape}")
        # print(f"X_num_test.shape : {X_num_test.shape if X_num_test is not None else None}")
        # print(f"X_cat_test.shape : {X_cat_test.shape if X_cat_test is not None else None}")
        # print(f"y_test.shape : {y_test.shape}")
        X_num_result = pd.concat([X_num_train,X_num_val,X_num_test], axis=0)
        X_cat_result = pd.concat([X_cat_train,X_cat_val,X_cat_test], axis=0)
        y_result = pd.concat([y_train,y_val,y_test], axis=0)
        # print(X_num_result)
        # print(X_cat_result)
        # print(y_result)
        # print(f"X_num_result.shape : {X_num_result.shape if X_num_result is not None else None}")
        # print(f"X_cat_result.shape : {X_cat_result.shape if X_cat_result is not None else None}")
        X_result=pd.concat([X_num_result,X_cat_result],axis=1)
        # 修改欄位名稱為 feature_1, feature_2, ..., feature_n
        X_result.columns = [f'Feature_{i+1}' for i in range(len(X_result.columns))]
        # print(f"X_result.shape : {X_result.shape}")
        # print(f"y_result.shape : {y_result.shape}")
        data=pd.concat([X_result,y_result],axis=1)
        print(data)
        print(data.shape)
        



        X_train,X_test,y_train,y_test = train_test_split(data.drop(columns=['target']),data['target'],test_size=0.4,random_state=42)

        # X_train=pd.concat([pd.concat([X_num_train,X_cat_train],axis=1),pd.concat([X_num_val,X_cat_val],axis=1)],axis=0)
        # y_train=pd.concat([y_train,y_val],axis=0)
        print(f"X_train.shape : {X_train.shape}")
        print(f"y_train.shape : {y_train.shape}")
        # X_test=pd.concat([X_num_test,X_cat_test],axis=1)
        print(f"X_test.shape : {X_test.shape}")
        print(f"y_test.shape : {y_test.shape}")
        print(f"type(X_train) = {type(X_train)}")
        print(f"type(y_train) = {type(y_train)}")
        print(f"type(X_test) = {type(X_test)}")
        print(f"type(y_test) = {type(y_test)}")
        # 儲存 X_train 和 y_train
        X_train.to_csv(f'./all/{folder_name}/X_train.csv', index=False,header=True)
        y_train.to_csv(f'./all/{folder_name}/y_train.csv', index=False,header=True)
        
        
        # 計算一半的樣本數量
        half_size = len(y_test) // 2

        # 建立一個標籤列表，前一半是 1 (public)，後一半是 0 (private)
        public_private_split = np.array([1] * half_size + [0] * (len(y_test) - half_size))

        # 打亂 public 和 private 樣本的排列
        np.random.seed(191)  # 固定隨機種子以確保可重現性
        np.random.shuffle(public_private_split)
        # 將 public_private_split 加入到 y_test 作為新的一列
        y_test = pd.DataFrame({'y_test': y_test, 'leaderboard': public_private_split})

        # 將 public 的標籤標記為 1，private 的標籤標記為 0
        y_test['leaderboard'] = y_test['leaderboard'].map({0: 'private', 1: 'public'})

        # 儲存 X_test 和 y_test
        X_test.to_csv(f'./all/{folder_name}/X_test.csv', index=False,header=True)
        y_test.to_csv(f'./all/{folder_name}/y_test.csv', index=False,header=True)

        X_train.to_csv(f'./Competition_data/{folder_name}/X_train.csv', index=False,header=True)
        y_train.to_csv(f'./Competition_data/{folder_name}/y_train.csv', index=False,header=True)
        X_test.to_csv(f'./Competition_data/{folder_name}/X_test.csv', index=False,header=True)

        print("\n\n\n")






