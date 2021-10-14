import torch
import torch.nn as nn
import torch.optim as opt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图像参数
img_width = 128
img_height = 128
means_3channels = [0.4853868, 0.45147458, 0.4142927]
std_3channels = [0.22896309, 0.22360295, 0.22433776]

# 超参数
per_batch_size = 25  # 批量载入图片的个数
epochs = 100  # 训练的轮次
InputDim = img_width
OutDim = 2
Neurons = 150
Layers = 3

# 数据路径
train_data_path = '../data/train'
validation_data_path = '../data/validation'
test_data_path = '../data/test'

# 数据预处理，提升泛化能力.
# ToTensor已经将图像数据归一化到[0, 1]
# Normalize将图像数据进一步调整到[-1, 1]
train_transform = transforms.Compose([
    transforms.RandomResizedCrop([img_width, ]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(means_3channels, std_3channels)
])
validation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means_3channels, std_3channels)
])

# ImageFolder结合DataLoader导入数据集
# DataLoader本质上是一个可迭代对象
train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
validation_dataset = torchvision.datasets.ImageFolder(
    root=validation_data_path, transform=validation_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=per_batch_size, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=per_batch_size, shuffle=False)


# 定义模型
class ImageRnn(nn.Module):
    def __init__(self, batch_size, input_dim, out_dim, neurons, layers):
        super(ImageRnn, self).__init__()
        self.batch_size = batch_size  # 批量输入的大小
        self.layers = layers  # RNN的层数
        self.neurons = neurons  # 隐藏层权重数量
        self.out_dim = out_dim  # 输出维度（也就是分类个数）
        self.input_dim = input_dim  # 输入维度

        self.basic_rnn = nn.LSTM(self.input_dim*3, self.neurons, num_layers=self.layers)
        self.FC = nn.Linear(self.neurons, self.out_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.zeros(self.layers, self.batch_size, self.neurons).to(device)

    # 是对RNN的一层封装，自定义组合RNN。
    def forward(self, data):
        self.batch_size = data.size(0)
        data = data.permute(1, 0, 2)  # 第一维度与第二维度换位

        # 前向传播，rnn_out是128个序列的输出值。
        rnn_out, self.hidden = self.basic_rnn.forward(data)
        # print(rnn_out)
        # 取RNN的最后一个序列，然后求出每一类概率.rnn_out的size为[75, 150] out为75*2
        out = self.FC(rnn_out[-1])
        return out.view(-1, self.out_dim)


model = ImageRnn(per_batch_size, InputDim, OutDim, Neurons, Layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = opt.Adam(model.parameters(), lr=0.001)

# 初始化模型的weight, eye返回一个二维张量，对角线为1，其他地方为0。
# model.basic_rnn.weight_hh_l0.data = torch.eye(n=Neurons, m=Neurons).to(device)
# model.basic_rnn.weight_hh_l1.data = torch.eye(n=Neurons, m=Neurons).to(device)
# model.basic_rnn.weight_hh_l2.data = torch.eye(n=Neurons, m=Neurons).to(device)


def get_accuracy(log_it, target, batch_size):
    corrects = (torch.max(log_it, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects / batch_size
    return accuracy.item()


def validation_accuracy():
    test_acc = 0.0
    for index1, (inputs, labels1) in enumerate(validation_loader):
        labels1 = labels1.to(device)
        inputs = inputs.view(-1, 128, 128).to(device)
        outputs1 = model(inputs)
        batch_acc = get_accuracy(outputs1, labels1, per_batch_size)
        print("Batch:{:0>2d}, Accuracy : {:<6.4f}%".format(index1, batch_acc))
        test_acc = test_acc + batch_acc
    return test_acc / index1


# 循环epoch个轮次
for epoch in range(epochs):
    train_running_loss = 0.0
    train_acc = 0.0
    model.train()  # 网络进入训练模式

    # enumerate内置函数，同时遍历索引和元素
    for index, (input_data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        model.hidden = model.init_hidden()

        input_data = input_data.view(input_data.size(0), 128, -1).to(device)
        labels = labels.to(device)
        outputs = model.forward(input_data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_running_loss = train_running_loss + loss.detach().item()
        train_acc = train_acc + get_accuracy(outputs, labels, per_batch_size)
    model.eval()
    msg = 'Epoch : {:0>2d} | Loss : {:<6.4f} | Train Accuracy : {:<6.2f}%'
    print(msg.format(epoch, train_running_loss / index, train_acc / index))

print('Validation Accuracy : {:<6.4f}%'.format(validation_accuracy()))
