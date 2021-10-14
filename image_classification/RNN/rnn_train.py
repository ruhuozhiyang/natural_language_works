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
# means_3channels = [0.4853868, 0.45147458, 0.4142927]
# std_3channels = [0.22896309, 0.22360295, 0.22433776]
means_3channels = [0.4853868]
std_3channels = [0.22896309]

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
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomResizedCrop([img_width, ]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(means_3channels, std_3channels)
])
validation_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
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

        self.basic_rnn = nn.RNN(self.input_dim, self.neurons, num_layers=self.layers)
        self.FC = nn.Linear(self.neurons, self.out_dim)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.zeros(self.layers, self.batch_size, self.neurons).to(device)

    def forward(self, x):
        self.batch_size = x.size(0)
        x = x.permute(1, 0, 2)  # 第一维度与第二维度换位

        rnn_out, self.hidden = self.basic_rnn(x, self.hidden)  # 前向传播
        out = self.FC(rnn_out[-1])  # 取RNN的最后一层，然后求出每一类概率. out为25*2
        return out.view(-1, self.out_dim)


model = ImageRnn(per_batch_size, InputDim, OutDim, Neurons, Layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = opt.Adam(model.parameters(), lr=0.001)

# 初始化模型的weight
model.basic_rnn.weight_hh_l0.data = torch.eye(n=Neurons, m=Neurons, out=None).to(device)
model.basic_rnn.weight_hh_l1.data = torch.eye(n=Neurons, m=Neurons, out=None).to(device)
model.basic_rnn.weight_hh_l2.data = torch.eye(n=Neurons, m=Neurons, out=None).to(device)


def get_accuracy(log_it, target, batch_size):
    corrects = (torch.max(log_it, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects / batch_size
    return accuracy.item()


# 循环epoch个轮次
for epoch in range(epochs):
    train_running_loss = 0.0
    train_acc = 0.0
    model.train()  # 网络进入训练模式

    # enumerate内置函数，同时遍历索引和元素
    for index, item in enumerate(train_loader):
        optimizer.zero_grad()
        # reset hidden states
        model.hidden = model.init_hidden()

        # get inputs
        input_data, labels = item  # 元素带着数据信息和标签信息 个数为batch_size
        # input_data.size()为[25, 3, 128, 128]
        input_data = input_data.view(-1, 128, 128).to(device)
        labels = labels.to(device)
        # forward+backward+optimize
        outputs = model.forward(input_data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_running_loss = train_running_loss + loss.detach().item()
        train_acc = train_acc + get_accuracy(outputs, labels, per_batch_size)
    model.eval()
    msg = 'Epoch : {:0>2d} | Loss : {:<6.4f} | Train Accuracy : {:<6.2f}%'
    print(msg.format(epoch, train_running_loss / index, train_acc / index))

test_acc = 0.0
for i, data in enumerate(validation_loader, 0):
    inputs, labels = data
    labels = labels.to(device)
    inputs = inputs.view(-1, 128, 128).to(device)
    outputs = model(inputs)
    thisBatchAcc = get_accuracy(outputs, labels, per_batch_size)
    print("Batch:{:0>2d}, Accuracy : {:<6.4f}%".format(i, thisBatchAcc))
    test_acc = test_acc + thisBatchAcc
print('============平均准确率===========')
print('Test Accuracy : {:<6.4f}%'.format(test_acc / i))
