import torch.nn as nn
import torch
torch.random.manual_seed(42)

def text1():
#随机数初始化
    liner=nn.Linear(5, 3)
    nn.init.uniform_(liner.weight)
    nn.init.uniform_(liner.bias)
    print(liner.weight,liner.bias)
def text2():
#固定值初始化
    liner=nn.Linear(5, 3)
    nn.init.constant_(liner.weight,1)
    nn.init.constant_(liner.bias,0)
    print(liner.weight,liner.bias)
def text1():
#初始化
    liner=nn.Linear(5, 3)
    nn.init.uniform_(liner.weight)
    nn.init.uniform_(liner.bias)
    print(liner.weight,liner.bias)




if __name__ == '__main__':
    # text1()
    text2()