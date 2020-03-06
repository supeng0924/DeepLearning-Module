# PyTorch教程

## 0. 教程介绍

参考PyTorch官方教程中文版[link](http://pytorch123.com/)

## 1. PyTorch入门

PyTorch是有Facebook的人工智能小组开发，能够实现强大的GPU加速，同时还支持动态神经网络。  
PyTorch提供了两个高级功能：**GPU加速的张量计算** 和 **自动求导系统**  

```
TensorFlow和caffe都是命令式的编程语言，而且是静态的，
	首先构建一个神经网络，，然后一次又一次使用相同的结构，如果要想改变网络的结构，必须从头开始
PyTorch通过反向求导技术，可以零延迟地任意改变神经网络的行为，而且速度快。
PyTorch的缺点：
	不支持快速傅里叶，沿维度翻转张量，检查无穷与非数值张量，针对移动端、嵌入式部署以及高性能
	服务器端的不熟其性能有待提升
```

### 1.1 Tensors张量

Tensors类似于numpy中的ndarrays，同时Tensors可以使用GPU进行计算

```
构造张量
1) 构造5x3矩阵，不初始化
x=torch.empty(5,3)
2) 构造随机初始化矩阵
x=torch.rand(5,3)
3) 构造全为0矩阵，数据类型是long
x=torch.zeros(5,3,dtype=torch.long)
4) 使用数据直接就构造张量
x=torch.tensor([5.5, 3])
5) 获取张量维度信息 
print(x.size())  # torch.Size([5,3])   (torch.Size是一个元组，所以它支持左右的元组操作)
```

```
加法操作
x=torch.rand(5,3)
y=x.rand_like(x)
1) print(x+y)
2) print(torch.add(x,y))
3) result=torch.empty(5,3)
   torch.add(x,y,out=result)
4) y.add_(x)   # 将加法后结果保存到y
   任何是张量变化的操作都有一个前缀。例如：x.copy(y),x.t_() 都会改变x
```

```
改变Tensors的大小或形状
x=torch.rand(4,4)  
y=x.view(16)
z=x.view(-1,8)
print(x.size(),y.size(),z.size())
>>torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```

```
获取张量元素的value
如果有一个元素tensor，可以使用.item()来获得这个value
x=torch.randn(1)
print(x,x.item())
>>tensor([0.2348]) 0.23476029932498932
```

### 1.2 自动微分

autograd是pytorch中所有神经网络的核心，该包为Tensors上所有操作提供自动微分。它是一个由运行定义的框架，意味着以代码运行方式定义反向传播，而且每次迭代可以不同。

```
torch.Tensor是包的核心类，如果将其属性 .requires_grad 设置为True，则会开始跟踪针对tensor的所有操作，完成
计算后，可以调用 .backward() 来自动计算所有梯度。该tensor的梯度将累积到 .grad 属性中

要停止tensor历史记录的跟踪：
	可以调用 .detach() 将其与计算历史分离，并防止将来的计算被跟踪
	也可以将代码块使用 with torch.no_grad() 包装起来。在评估模型是特别有用，因为在评估阶段不需要梯度
还有一个类对于autograd实现非常重要，那就是Function。Tensor和Function互相连接并构建一个非循环图，它
保存整个完整的计算过程的历史信息。每个张量都有一个 .grad_fn 属性保存着创建了张量的Function的引用(用户自己
创建的张量，则 grad_fn 是None)

如果要计算导数，可以调用 tensor.backward()。如果tensor是标量(即只包含一个元素数据),则不需要制定任何参数。
如果它有多个元素，则需要指定一个gradient参数来指定张量的形状。

import torch
x=torch.ones(2,2,requires_grad=True)
y=x+2
print(y)
>>tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)
        
z=y*y*3
out=z.mean()
print(z,out)
>>tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward0>)
        
out.backward()
print(x.grad)
>>tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
```

