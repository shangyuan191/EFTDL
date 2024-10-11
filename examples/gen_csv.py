# 讀取文本文件內容
with open('Best_result.txt', 'r') as file:
    data = file.read()

# 拆分並處理文本資料
lines = data.strip().split('\n\n')  # 根據空行分塊
records = []

for block in lines:
    # 解析每一個 block
    lines = block.splitlines()
    if len(lines)<9
    dataset_name = lines[0].split(' : ')[1].strip()
    dataset_size = lines[1].split(' : ')[1].strip()
    task_type = lines[2].split(' : ')[1].strip()
    
    ratio_1 = lines[3].split(' : ')[1].strip()
    val_auc_1 = float(lines[4].split(': ')[1].strip())
    test_auc_1 = float(lines[5].split(': ')[1].strip())
    
    ratio_2 = lines[6].split(' : ')[1].strip()
    val_auc_2 = float(lines[7].split(': ')[1].strip())
    test_auc_2 = float(lines[8].split(': ')[1].strip())
    
    # 將數據存入列表
    records.append([dataset_name, dataset_size, task_type, ratio_1, val_auc_1, test_auc_1, ratio_2, val_auc_2, test_auc_2])

# 將數據轉換為 DataFrame
df = pd.DataFrame(records, columns=['Dataset Name', 'Dataset Size', 'Task Type', 
                                    'Train/Val/Test Split Ratio 1', 'Best Val AUC 1', 'Best Test AUC 1', 
                                    'Train/Val/Test Split Ratio 2', 'Best Val AUC 2', 'Best Test AUC 2'])

# 將 DataFrame 存成 CSV 文件
df.to_csv('output.csv', index=False)

print("CSV file saved successfully!")