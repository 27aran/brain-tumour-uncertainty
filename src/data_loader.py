import kagglehub
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import os

#Load Data
data_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

"""
Augmented training data
"""
#Transformations
train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

train_path_aug = os.path.join(data_path, "Training")
train_data_aug = ImageFolder(
    root=train_path_aug,
    transform= train_transform
)

"""
Plain training data and test data
"""
#Transformations
eval_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

train_path_plain = os.path.join(data_path, "Training")
train_data_plain = ImageFolder(root=train_path_plain, transform=eval_transform)


test_path = os.path.join(data_path, "Testing")
test_data = ImageFolder(root=test_path, transform=eval_transform)

#Split training data
train_size = int(len(train_data_plain) * 0.7)
val_size = int(len(train_data_plain) * 0.15)
cal_size = len(train_data_plain) - (train_size + val_size)

train, val, cal = random_split(train_data_plain, [train_size, val_size, cal_size], generator = torch.Generator().manual_seed(42)
)
train.dataset = train_data_aug

train_data_loader = DataLoader(train, batch_size=32, num_workers=2, pin_memory=True, shuffle=True)
val_data_loader = DataLoader(val, batch_size=32, num_workers=2, pin_memory=True, shuffle=False)
cal_data_loader = DataLoader(cal, batch_size=32, num_workers=2, pin_memory=True, shuffle=False)
test_data_loader = DataLoader(test_data, batch_size=32, num_workers=2, pin_memory=True, shuffle=False)

print(train_data_aug.classes)
print(len(train_data_aug))
print(len(train)+len(val)+len(cal))
print(len(train), len(val), len(cal), len(test_data))