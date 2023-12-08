import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import random
import os

# Dataset Class
class ImageDataset(Dataset):
    def __init__(self, path_label, transform=None):
        self.path_label = path_label
        self.transform = transform

    def __len__(self):
        return len(self.path_label)

    def __getitem__(self, idx):
        path, label = self.path_label[idx]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label
    
# Create dataset from raw structure obtained from Kaggle
# (1): Download dataset fron Kaggle: https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species?select=train
# (2): Unzip the downloaded dataset to './data/butterfly'
# (3): Run the function
def create_butterfly_dataset(path = './data/butterfly/', img_reshape=(3, 224, 224)):
    C, H, W = img_reshape
    transform = transforms.Compose([
                                    transforms.Resize((H, W)),
                                    transforms.ToTensor()])
      
    train_path = path + 'train'
    test_path = path + 'test'
    
    class_names=sorted(os.listdir(train_path))
    N=list(range(len(class_names)))
    normal_mapping=dict(zip(class_names,N)) 
    reverse_mapping=dict(zip(N,class_names))

    paths0=[]
    # store an image for each class for adversarial attack
    sample_img_dataset = torch.zeros((len(class_names), C, H, W))
    seen = list()
    for dirname, _, filenames in os.walk(train_path):
        for filename in filenames:
            if filename[-4:]=='.jpg':
                path=os.path.join(dirname, filename)
                label=dirname.split('\\')[-1]
                if label == '.ipynb_checkpoints':
                    continue
                paths0+=[(path,normal_mapping[label])]
            if label not in seen:
                image = Image.open(path).convert('RGB')
                image = transform(image)
                sample_img_dataset[normal_mapping[label], :, :, :] = image
                seen.append(label)
            
    tpaths0=[]
    for dirname, _, filenames in os.walk(test_path):
        for filename in filenames:
            if filename[-4:]=='.jpg':
                path=os.path.join(dirname, filename)
                label=dirname.split('\\')[-1]
                if label == '.ipynb_checkpoints':
                    continue
                tpaths0+=[(path,normal_mapping[label])]
                
    random.seed(123)
    random.shuffle(paths0)            
    random.shuffle(tpaths0)  

    trainset = ImageDataset(paths0, transform)
    testset = ImageDataset(tpaths0, transform)
    
    return trainset, testset, normal_mapping, reverse_mapping, sample_img_dataset

def create_imagenet_dataset(path='./data/tiny_imagenet/tiny-imagenet-200/', img_reshape=(3, 224, 224), split_ratio = 0.8):
    C, H, W = img_reshape
    transform = transforms.Compose([
                                    transforms.Resize((H, W)),
                                    transforms.ToTensor()])
      
    data_path = path + 'train'
    
    class_names = sorted(os.listdir(data_path))
    N = list(range(len(class_names)))
    normal_mapping = dict(zip(class_names, N)) 
    reverse_mapping = dict(zip(N, class_names))

    paths0 = []
    sample_img_dataset = torch.zeros((len(class_names), C, H, W))
    seen = list()

    for label in class_names:
        label_path = os.path.join(data_path, label, 'images')
        if not os.path.isdir(label_path):
            continue
        for filename in os.listdir(label_path):
            if filename.endswith('.JPEG'):
                path = os.path.join(label_path, filename)
                paths0.append((path, normal_mapping[label]))
                if label not in seen:
                    image = Image.open(path).convert('RGB')
                    image = transform(image)
                    sample_img_dataset[normal_mapping[label], :, :, :] = image
                    seen.append(label)
                
    random.seed(123)
    random.shuffle(paths0) 
    
    data = ImageDataset(paths0, transform)

    total_size = len(data)
    train_size = int(split_ratio * total_size)
    test_size = total_size - train_size

    trainset, testset = random_split(data, [train_size, test_size])

    return trainset, testset, normal_mapping, reverse_mapping, sample_img_dataset
