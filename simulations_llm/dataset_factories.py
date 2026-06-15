import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import os

from torch.utils.data import Sampler


def CIFAR10(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=1)
    
    num_classes = 10

    return num_classes, trainset, trainloader, testset, testloader


def CIFAR100(batch_size):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=False, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    num_classes = 100

    return num_classes, trainset, trainloader, testset, testloader

def MNIST(batch_size):
    
    transform_train = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

    transform_test = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=False, transform=transform_train)   
    trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    num_classes = 10

    return num_classes, trainset, trainloader, testset, testloader


def CIFAR10Scratch(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='/scratch0/wenchhan/9982345', train=True, download=False, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='/scratch0/wenchhan/9982345', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    num_classes = 10

    return num_classes, trainset, trainloader, testset, testloader


def create_val_img_folder(dataset_dir):
    '''
    This method is responsible for separating validation images into separate sub folders
    '''
    val_dir = os.path.join(dataset_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')

    fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))


def ImageNetScratch(batch_size):
    job_id = os.environ["JOB_ID"]
    base_folder = "/scratch0/wenchhan/" + job_id + "/"
    train_dataset = \
    torchvision.datasets.ImageFolder(base_folder + "tiny-imagenet-200/train",
                            transform=transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                            ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    create_val_img_folder(dataset_dir=base_folder + "tiny-imagenet-200")
    val_dataset = torchvision.datasets.ImageFolder(base_folder + "tiny-imagenet-200/val/images",
                            transform=transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                            ]))

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    num_classes = 200

    return num_classes, train_dataset, train_loader, val_dataset, val_loader


def ImageNet(batch_size):
    train_dataset = \
    torchvision.datasets.ImageFolder("/cluster/project2/gcreduce_data/data/tiny-imagenet-200/train",
                            transform=transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                            ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    create_val_img_folder(dataset_dir="/cluster/project2/gcreduce_data/data/tiny-imagenet-200")

    val_dataset = torchvision.datasets.ImageFolder("/cluster/project2/gcreduce_data/data/tiny-imagenet-200/val/images",
                            transform=transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                            ]))

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    num_classes = 200

    return num_classes, train_dataset, train_loader, val_dataset, val_loader



from torch.utils.data import Sampler

class PermutedDistributedSampler(Sampler):
    def __init__(self, dataset, permutation, num_replicas=None, rank=None, shuffle=False):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        
        self.dataset = dataset
        self.permutation = permutation
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.num_samples = len(permutation) // num_replicas
        self.total_size = self.num_samples * num_replicas

    def __iter__(self):
        indices = self.permutation

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(0)
            indices = torch.tensor(indices)[torch.randperm(len(indices), generator=g)].tolist()

        # Add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples
