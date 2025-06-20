from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

def gray_loader(path):
    return Image.open(path).convert('L')

def get_loaders(train_dir, test_dir, transform_train, transform_test, batch_size=64):
    train_set = ImageFolder(root=train_dir, transform=transform_train, loader=gray_loader)
    test_set = ImageFolder(root=test_dir, transform=transform_test, loader=gray_loader)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, train_set.classes
