import os
import csv
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
# dataset_names=[]
# X_trains=[]
# y_trains=[]
# X_tests=[]
# 設置權重
alpha = 0.3
beta = 0.4
gamma = 0.3
public_scores=[]
private_scores=[]
for folder_name in os.listdir("./all"):
    print(folder_name)
    y_test=pd.read_csv(f"./all/{folder_name}/y_test.csv",header=0)
    y_test_public = y_test[y_test['leaderboard'] == 'public']
    y_test_private = y_test[y_test['leaderboard'] == 'private']
    y_pred=pd.read_csv(f"./Competition_data/{folder_name}/y_predict.csv",header=0)
    y_pred_public = y_pred[y_test['leaderboard'] == 'public']
    y_pred_private = y_pred[y_test['leaderboard'] == 'private']

    # 計算 public leaderboard 的準確率和分類報告
    # 計算正樣本和負樣本的數量
    positive_count = y_test_public['y_test'].sum()  # 計算正樣本的數量
    negative_count = len(y_test_public) - positive_count  # 計算負樣本的數量

    # 計算比例
    total_count = len(y_test_public)
    positive_ratio = positive_count / total_count
    negative_ratio = negative_count / total_count
    acc_public=accuracy_score(y_test_public['y_test'], y_pred_public)
    precision_public=precision_score(y_test_public['y_test'], y_pred_public)
    f1_public=f1_score(y_test_public['y_test'], y_pred_public)
    score_public=alpha*acc_public+beta*precision_public+gamma*f1_public
    public_scores.append(score_public)
    # 印出結果
    print(f"num of total sample: {total_count}")
    print(f"num of pos sample: {positive_count}")
    print(f"num of neg sample: {negative_count}")
    print(f"ratio of pos sample: {positive_ratio:.2%}")
    print(f"ratio of neg sample: {negative_ratio:.2%}")
    print("Public Leaderboard Results:")
    print(f"Public Accuracy: {acc_public}")
    print(f"Public Precision: {precision_public}")
    print(f"Public F1 Score: {f1_public}")
    print(f"Public Score: {score_public}")
    print()



    # 計算 private leaderboard 的準確率和分類報告
    acc_private=accuracy_score(y_test_private['y_test'], y_pred_private)
    precision_private=precision_score(y_test_private['y_test'], y_pred_private)
    f1_private=f1_score(y_test_private['y_test'], y_pred_private)
    score_private=alpha*acc_private+beta*precision_private+gamma*f1_private
    private_scores.append(score_private)
    print("Private Leaderboard Results:")
    print(f"Private Accuracy: {acc_private}")
    print(f"Private Precision: {precision_private}")
    print(f"Private F1 Score: {f1_private}")
    print(f"Private Score: {score_private}")
    print("\n\n")
print("Public Leaderboard Average Score:",np.mean(public_scores))
print("Private Leaderboard Average Score:",np.mean(private_scores))