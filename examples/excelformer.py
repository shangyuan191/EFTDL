"""Reported (reproduced) accuracy (for multi-classification task), auc
(for binary classification task) and rmse (for regression task)
based on Table 1 of the paper https://arxiv.org/abs/2301.02819.
ExcelFormer uses the same train-validation-test split as the Yandex paper.
The reproduced results are based on Z-score Normalization, and the
reported ones are based on :class:`QuantileTransformer` preprocessing
in the Sklearn Python package. The above preprocessing is applied
to numerical features.

california_housing: 0.4587 (0.4550) mixup: feature, num_layers: 3,
gamma: 1.00, epochs: 300
jannis : 72.51 (72.80) mixup: feature
covtype: 97.17 (97.02) mixup: hidden
helena: 38.20 (37.68) mixup: feature
higgs_small: 80.75 (79.27) mixup: hidden
"""
import argparse
import os.path as osp

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import AUROC, Accuracy, MeanSquaredError
from tqdm import tqdm

from torch_frame.data.loader import DataLoader
from torch_frame.datasets.yandex import Yandex
from torch_frame.nn import ExcelFormer
from torch_frame.transforms import CatToNumTransform, MutualInformationSort
from tqdm import tqdm
import json
import os
# classification_datasets = {
#     'adult', 'aloi', 'covtype', 'helena', 'higgs_small', 'jannis'
# }
# regression_datasets = {'california_housing', 'microsoft', 'yahoo', 'year'}


# dataset_strs = [
#     'adult', 'aloi', 'covtype', 'helena', 
#     'higgs_small', 'jannis', 
#     'california_housing', 'microsoft', 
#     'yahoo', 'year'
# ]
train_val_test_split_ratios=[[0.64,0.16,0.2],[0.05,0.15,0.8]]
dataset_info={}
with open(f'dataset_info.json','r',encoding='utf-8') as f:
    dataset_info=json.load(f)
with open(f'./result/Best_result.txt','w') as f:
    for dataset_name in dataset_info.keys():
        dataset_type=dataset_info[dataset_name]['dataset_type']
        task_type=dataset_info[dataset_name]['task_type']
        f.write(f"Dataset name : {dataset_name}\n")
        f.write(f"Dataset size : {dataset_type}\n")
        f.write(f"Task type : {task_type}\n")
        all_train_loss=[]
        all_train_metric=[]
        all_val_metric=[]
        all_test_metric=[]
        for train_val_test_split_ratio in train_val_test_split_ratios:
            parser = argparse.ArgumentParser()
            parser.add_argument('--dataset_size',type=str,default=f'{dataset_type}_datasets')
            parser.add_argument('--dataset', type=str, default=f"{dataset_name}")
            parser.add_argument('--mixup', type=str, default=None,
                                choices=[None, 'feature', 'hidden'])
            parser.add_argument('--channels', type=int, default=256)
            parser.add_argument('--batch_size', type=int, default=512)
            parser.add_argument('--num_heads', type=int, default=4)
            parser.add_argument('--num_layers', type=int, default=5)
            parser.add_argument('--lr', type=float, default=0.001)
            parser.add_argument('--gamma', type=float, default=0.95)
            parser.add_argument('--epochs', type=int, default=200)
            parser.add_argument('--compile', action='store_true')
            args = parser.parse_args()

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(device)
            print(args.dataset)
            path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',args.dataset_size,
                            args.dataset)
            print(path)
            dataset = Yandex(root=path, name=args.dataset,train_val_test_split_ratio=train_val_test_split_ratio)
            dataset.materialize()
            print(len(dataset))
            print(type(dataset))
            train_dataset, val_dataset, test_dataset = dataset.split()
            print(len(train_dataset))
            print(type(train_dataset))
            print(len(val_dataset))
            print(type(val_dataset))
            print(len(test_dataset))
            print(type(test_dataset))
            train_tensor_frame = train_dataset.tensor_frame
            val_tensor_frame = val_dataset.tensor_frame
            test_tensor_frame = test_dataset.tensor_frame

            # CategoricalCatBoostEncoder encodes the categorical features
            # into numerical features with CatBoostEncoder.
            categorical_transform = CatToNumTransform()
            categorical_transform.fit(train_dataset.tensor_frame, train_dataset.col_stats)

            train_tensor_frame = categorical_transform(train_tensor_frame)
            val_tensor_frame = categorical_transform(val_tensor_frame)
            test_tensor_frame = categorical_transform(test_tensor_frame)
            col_stats = categorical_transform.transformed_stats

            # MutualInformationSort sorts the features based on mutual
            # information.
            mutual_info_sort = MutualInformationSort(task_type=dataset.task_type)

            mutual_info_sort.fit(train_tensor_frame, col_stats)
            train_tensor_frame = mutual_info_sort(train_tensor_frame)
            val_tensor_frame = mutual_info_sort(val_tensor_frame)
            test_tensor_frame = mutual_info_sort(test_tensor_frame)

            train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                                    shuffle=True)
            val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
            test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

            is_classification = dataset.task_type.is_classification

            if is_classification:
                out_channels = dataset.num_classes
            else:
                out_channels = 1

            is_binary_class = is_classification and out_channels == 2

            if is_binary_class:
                metric_computer = AUROC(task='binary')
            elif is_classification:
                metric_computer = Accuracy(task='multiclass', num_classes=out_channels)
            else:
                metric_computer = MeanSquaredError()
            metric_computer = metric_computer.to(device)

            model = ExcelFormer(
                in_channels=args.channels,
                out_channels=out_channels,
                num_layers=args.num_layers,
                num_cols=train_tensor_frame.num_cols,
                num_heads=args.num_heads,
                residual_dropout=0.,
                diam_dropout=0.3,
                aium_dropout=0.,
                mixup=args.mixup,
                col_stats=mutual_info_sort.transformed_stats,
                col_names_dict=train_tensor_frame.col_names_dict,
            ).to(device)
            model = torch.compile(model, dynamic=True) if args.compile else model
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            lr_scheduler = ExponentialLR(optimizer, gamma=args.gamma)


            def train(epoch: int) -> float:
                model.train()
                loss_accum = total_count = 0

                for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
                    tf = tf.to(device)
                    # Train with FEAT-MIX or HIDDEN-MIX
                    pred_mixedup, y_mixedup = model(tf, mixup_encoded=True)
                    if is_classification:
                        # Softly mixed one-hot labels
                        loss = F.cross_entropy(pred_mixedup, y_mixedup)
                    else:
                        loss = F.mse_loss(pred_mixedup.view(-1), y_mixedup.view(-1))
                    optimizer.zero_grad()
                    loss.backward()
                    loss_accum += float(loss) * len(y_mixedup)
                    total_count += len(y_mixedup)
                    optimizer.step()
                return loss_accum / total_count


            @torch.no_grad()
            def test(loader: DataLoader) -> float:
                model.eval()
                metric_computer.reset()
                for tf in loader:
                    tf = tf.to(device)
                    pred = model(tf)
                    if is_binary_class:
                        metric_computer.update(pred[:, 1], tf.y)
                    elif is_classification:
                        pred_class = pred.argmax(dim=-1)
                        metric_computer.update(pred_class, tf.y)
                    else:
                        metric_computer.update(pred.view(-1), tf.y.view(-1))

                if is_classification:
                    return metric_computer.compute().item()
                else:
                    return metric_computer.compute().item()**0.5


            if is_classification:
                metric = 'Acc' if not is_binary_class else 'AUC'
                best_val_metric = 0
                best_test_metric = 0
            else:
                metric = 'RMSE'
                best_val_metric = float('inf')
                best_test_metric = float('inf')
            now_train_loss=[]
            now_train_metric=[]
            now_val_metric=[]
            now_test_metric=[]

            
            epochs=range(1,args.epochs+1)
            for epoch in range(1, args.epochs + 1):
                train_loss = train(epoch)
                train_metric = test(train_loader)
                val_metric = test(val_loader)
                test_metric = test(test_loader)
                
                now_train_loss.append(train_loss)
                now_train_metric.append(train_metric)
                now_val_metric.append(val_metric)
                now_test_metric.append(test_metric)

                if is_classification and val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_test_metric = test_metric
                elif not is_classification and val_metric < best_val_metric:
                    best_val_metric = val_metric
                    best_test_metric = test_metric

                print(f'Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, '
                    f'Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}')
                lr_scheduler.step()

            print(f'Best Val {metric}: {best_val_metric:.4f}, '
                f'Best Test {metric}: {best_test_metric:.4f}')
            
            
            f.write(f"Train_val_test_split_ratio : {train_val_test_split_ratio}\n")
            f.write(f'Best Val {metric}: {best_val_metric:.4f}, 'f'Best Test {metric}: {best_test_metric:.4f}\n')
            all_train_loss.append(now_train_loss)
            all_train_metric.append(now_train_metric)
            all_val_metric.append(now_val_metric)
            all_test_metric.append(now_test_metric)
        # 創建圖表，2 行（每行兩張圖），每個比例會有兩個子圖
        fig, axes = plt.subplots(len(train_val_test_split_ratios), 2, figsize=(12, 8))

        for i, ratio in enumerate(train_val_test_split_ratios):
            # 第一張子圖：訓練損失
            axes[i, 0].plot(epochs, all_train_loss[i], label=f'Train Loss (Ratio: {ratio})')
            axes[i, 0].set_xlabel('Epoch')
            axes[i, 0].set_ylabel('Loss')
            axes[i, 0].set_title(f'Training Loss Over Epochs (Ratio: {ratio})')
            axes[i, 0].legend()

            # 第二張子圖：metric（包括訓練、驗證、測試的 metric）
            axes[i, 1].plot(epochs, all_train_metric[i], label=f'Train {metric} (Ratio: {ratio})')
            axes[i, 1].plot(epochs, all_val_metric[i], label=f'Val {metric} (Ratio: {ratio})')
            axes[i, 1].plot(epochs, all_test_metric[i], label=f'Test {metric} (Ratio: {ratio})')
            axes[i, 1].set_xlabel('Epoch')
            axes[i, 1].set_ylabel(f'{metric}')
            axes[i, 1].set_title(f'{metric} Over Epochs (Ratio: {ratio})')
            axes[i, 1].legend()

        # 調整布局，避免重疊
        plt.tight_layout()
        result_path=f"./result/{dataset_type}_datasets/{task_type}/{dataset_name}/"
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        plt.savefig(f"{result_path}comparison_{args.dataset}.png")

        f.write('\n\n')
