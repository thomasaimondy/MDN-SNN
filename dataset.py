from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from nettalk import NETtalk
from nmnist import nmnist
from tidigits import Tidigits
from timit import Timit
from tensorboardX import SummaryWriter

def loaddata(task, if_tb):
    writer = None
    if task == 'M':
        if if_tb:
            writer = SummaryWriter(comment = '-Mni')
        hyperparams = [100, 784, 10, 1e-3, 20, 'MNIST', 1e-3]
        train_dataset = dsets.MNIST(root = './data/mnist', train = True, transform = transforms.ToTensor(), download = True)
        test_dataset = dsets.MNIST(root = './data/mnist', train = False, transform = transforms.ToTensor())
        train_loader = DataLoader(dataset = train_dataset, batch_size = hyperparams[0], shuffle = True)
        test_loader = DataLoader(dataset = test_dataset, batch_size = hyperparams[0], shuffle = False)
    elif task == 'F':
        if if_tb:
            writer = SummaryWriter(comment = '-Fas')
        hyperparams = [100, 784, 10, 1e-3, 20, 'FashionMNIST', 1e-3]
        train_dataset = dsets.FashionMNIST(root = './data/fashion', train = True, transform = transforms.ToTensor(), download = True)
        test_dataset = dsets.FashionMNIST(root = './data/fashion', train = False, transform = transforms.ToTensor())
        train_loader = DataLoader(dataset = train_dataset, batch_size = hyperparams[0], shuffle = True)
        test_loader = DataLoader(dataset = test_dataset, batch_size = hyperparams[0], shuffle = False)
    elif task == 'N':
        if if_tb:
            writer = SummaryWriter(comment = '-Net')
        hyperparams = [5, 189, 26, 1e-3, 20, 'NETtalk', 1e-3]
        train_dataset = NETtalk('train', transform=transforms.ToTensor())
        test_dataset = NETtalk('test', transform=transforms.ToTensor())
        train_loader = DataLoader(dataset = train_dataset, batch_size = hyperparams[0], shuffle = True)
        test_loader = DataLoader(dataset = test_dataset, batch_size = hyperparams[0], shuffle = False)
    elif task == 'C10':
        if if_tb:
            writer = SummaryWriter(comment = '-Cif')
        hyperparams = [100, 3072, 10, 1e-4, 20, 'Cifar10', 1e-3]
        train_dataset = dsets.CIFAR10(root = './data/cifar10', train = True, transform = transforms.ToTensor(), download = True)
        test_dataset = dsets.CIFAR10(root = './data/cifar10', train = False, transform = transforms.ToTensor())
        train_loader = DataLoader(dataset = train_dataset, batch_size = hyperparams[0], shuffle = True)
        test_loader = DataLoader(dataset = test_dataset, batch_size = hyperparams[0], shuffle = False)
    elif task == 'NM':
        if if_tb:
            writer = SummaryWriter(comment = '-Nmn')
        hyperparams = [100, 2592, 10, 1e-3, 20, 'NMNIST', 1e-3]
        train_dataset = nmnist(datasetPath = 'nmnist/Train/', sampleFile = 'nmnist/Train.txt', samplingTime = 1.0, sampleLength = 20)
        test_dataset = nmnist(datasetPath = 'nmnist/Test/', sampleFile = 'nmnist/Test.txt', samplingTime = 1.0, sampleLength = 20)
        train_loader = DataLoader(dataset = train_dataset, batch_size = hyperparams[0], shuffle = True)
        test_loader = DataLoader(dataset = test_dataset, batch_size = hyperparams[0], shuffle = False)
    elif task == 'T':
        if if_tb:
            writer = SummaryWriter(comment = '-Tid')
        hyperparams = [10, 30, 10, 1e-2, 30, 'TiDigits', 1e-4, 1, 30, transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))])]
        train_dataset = Tidigits('train', hyperparams[7], hyperparams[8], hyperparams[4], transform = hyperparams[9])
        test_dataset = Tidigits('test', hyperparams[7], hyperparams[8], hyperparams[4], transform = hyperparams[9])
        train_loader = DataLoader(dataset = train_dataset, batch_size = hyperparams[0], shuffle = True)
        test_loader = DataLoader(dataset = test_dataset, batch_size = hyperparams[0], shuffle = False, drop_last = True)
    elif task == 'TM':
        if if_tb:
            writer = SummaryWriter(comment = '-Tim')
        hyperparams = [32, 520, 2, 1e-3, 20, 'Timit', 1e-3]
        train_dataset = Timit('TRAIN')
        test_dataset = Timit('TEST')
        train_loader = DataLoader(dataset = train_dataset, batch_size = hyperparams[0], shuffle = True)
        test_loader = DataLoader(dataset = test_dataset, batch_size = hyperparams[0], shuffle = False)

    return writer, hyperparams, train_dataset, test_dataset, train_loader, test_loader
