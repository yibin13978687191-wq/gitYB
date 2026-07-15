import torch
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # 调用父类nn.Module的初始化方法

        # 定义神经网络的各层结构
        # linear1: 输入层到第一个隐藏层 (输入特征数=3, 输出特征数=3)
        self.linear1 = nn.Linear(in_features=3, out_features=3)
        # linear2: 第一个隐藏层到第二个隐藏层 (输入特征数=3, 输出特征数=2)
        self.linear2 = nn.Linear(in_features=3, out_features=2)
        # output: 输出层 (输入特征数=2, 输出特征数=2)
        self.output = nn.Linear(in_features=2, out_features=2)

        '''
        使用Kaiming初始化方法初始化权重和偏置项
        Kaiming初始化适用于ReLU激活函数
        '''
        nn.init.kaiming_normal_(self.linear1.weight)  # 对linear1层权重进行Kaiming正态分布初始化
        nn.init.zeros_(self.linear1.bias)             # 将linear1层偏置初始化为0
        nn.init.kaiming_normal_(self.linear2.weight)  # 对linear2层权重进行Kaiming正态分布初始化
        nn.init.zeros_(self.linear2.bias)             # 将linear2层偏置初始化为0

    def forward(self, x):
        """
        前向传播函数定义数据如何通过网络层流动

        参数:
            x (Tensor): 输入张量

        返回:
            Tensor: 经过网络处理后的输出张量
        """
        x = self.linear1(x)      # 第一层线性变换: wx + b
        x = torch.relu(x)        # 应用ReLU激活函数引入非线性因素

        x = self.linear2(x)      # 第二层线性变换
        x = torch.relu(x)        # 再次应用ReLU激活函数

        x = self.output(x)       # 输出层线性变换
        x = torch.softmax(x, dim=-1)  # 应用Softmax函数得到概率分布
        return x

def train():
    """
    模型训练与预测演示函数
    """
    my_net = Net()  # 实例化自定义神经网络模型

    # 创建随机输入数据 (batch_size=5, feature_dim=3)
    data = torch.randn(5, 3)
    print(f'data: {data}, data.shape: {data.shape}')

    # 执行前向传播获取输出结果
    output = my_net(data)
    print(f'output: {output}, output.shape: {output.shape}')
    print(output.requires_grad)  # 查看输出是否需要梯度计算(用于反向传播)
def dm01():
    """
    反向传播演示函数 - 展示如何使用优化器和损失函数进行参数更新
    """
    # 创建一个可学习的参数w，并设置requires_grad=True以便自动求导
    w = torch.tensor([1], requires_grad=True, dtype=torch.float)

    # 定义简单的二次损失函数 L = (1/2)*w^2
    criterion = w**2 * 0.5

    # 使用随机梯度下降(SGD)优化器，学习率设为0.01，动量为0.9
    optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)

    # 清除之前的梯度信息
    optimizer.zero_grad()

    # 计算损失并执行反向传播计算梯度
    criterion.sum().backward()

    # 更新参数 (w = w - learning_rate * gradient)
    optimizer.step()

    # 打印当前参数值、梯度及损失值
    print(f'w: {w}, w.grad: {w.grad}, criterion: {criterion}, optimizer: {optimizer}')
# 定义等间隔学习率衰减方法
def lr_decay():
    """
    演示等间隔学习率衰减的函数

    该函数通过模拟简单的线性回归训练过程，展示StepLR学习率调度器的工作原理。
    每隔一定epoch数，学习率会按照指定比例衰减。

    参数:
        无

    返回值:
        无

    功能说明:
        - 创建简单的线性模型进行训练
        - 使用StepLR调度器每隔50个epoch将学习率乘以0.5
        - 绘制学习率随epoch变化的曲线图
    """
    epoch = 200
    iteration = 10
    lr = 0.1
    y_true = torch.tensor([1.0])
    x = torch.tensor([0.0])
    bias = torch.tensor([0.0], requires_grad=True, dtype=torch.float)
    weight = torch.tensor([0.0], requires_grad=True, dtype=torch.float)
    optimizer = optim.SGD(params=[weight], lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.5)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)



    lr_list = []
    epoch_list = []

    # 训练循环：执行200个epoch
    for i in range(epoch):
        epoch_list.append(i)
        lr_list.append(optimizer.param_groups[0]['lr'])

        # 每个epoch内执行10次迭代训练
        for j in range(iteration):
            y_pred = weight * x + bias
            loss = torch.nn.MSELoss()(y_pred, y_true)
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

        # 更新学习率调度器
        scheduler.step()
        print(f'epoch:{i + 1},lr:{optimizer.param_groups[0]["lr"]}，')

    # 绘制学习率变化曲线
    plt.plot(epoch_list, lr_list)
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.grid()
    plt.show()

def dropout():
    

if __name__ == '__main__':
    # train()   # 运行模型训练演示
    # dm01()    # 运行反向传播演示
    lr_decay()
