import torch
import torchvision.transforms
from torchvision.datasets import ImageFolder


def compute_means_std(train_data):
    """
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    """
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


train_data_path = '../data/train'
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

if __name__ == '__main__':
    train_dataset = ImageFolder(root=train_data_path, transform=train_transform)
    print(compute_means_std(train_dataset))
