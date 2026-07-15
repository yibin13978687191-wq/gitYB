from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas
import openpyxl
import numpy

from ann_project.models import ANN, DeviceManager, ModelTrainer, TrainingConfigurator, set_seed
from torch.utils.data import DataLoader, TensorDataset

if __name__ == '__main__':
    # 创建数据集
    file_path = PROJECT_ROOT / "data/raw/ANN测试原数据2026.1.15.xlsx"
    data = pandas.read_excel(file_path, sheet_name="Sheet1", header=0)
    # print(f'data.index:{data.index}')
    # print(f'data.columns:{data.columns}')
    # print(f'data.values:{data.values}')
    # print(f'data.values.shape:{data.values.shape}')
    # print(f'data.values.dtype:{data.values.dtype}')
    data.set_index("试验号", inplace=True)

    x_train=data.iloc[:-2, :-1]
    y_train=data.iloc[:-2, -1]
    train_dataset = TensorDataset(torch.tensor(x_train.values,dtype=torch.float32),
                                  torch.tensor(y_train.values,dtype=torch.float32))
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    x_val=data.iloc[-2:, :-1]
    y_val=data.iloc[-2:, -1]
    val_dataset = TensorDataset(torch.tensor(x_val.values,dtype=torch.float32),
                                torch.tensor(y_val.values,dtype=torch.float32))
    val_dataloader = DataLoader(val_dataset, batch_size=5, shuffle=False)


    set_seed(42)
    input_dim=x_train.shape[1]
    output_dim=1
    device_manager = DeviceManager(device_id=None)
    device = device_manager.get_device()
    model = ANN(input_dim,(6,12,18,6),output_dim,task_type='regression',activation='sigmoid')


    # print(list(model.children()))
    # model.print_model_parameters()
    # model.print_network_structure()

    print("\n=== 训练配置 ===")
    # 学习率和权重衰减设置
    training_config = TrainingConfigurator(model, device, learning_rate=0.01, weight_decay=0.0001)
    criterion, optimizer, scheduler, model = training_config()
    # 初始化模型训练器对象
    trainer = ModelTrainer(model, criterion, optimizer, scheduler, device, epochs=200)
    train_dict = trainer.train(train_dataloader,val_dataloader, print_interval=3)
    print(trainer.get_training_results(output_format='str'))
    trainer.visualize_training_results(save_path=PROJECT_ROOT / "outputs/figures/visualization.png")
    print(trainer.get_last_y_pred())



