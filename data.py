from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import glob 
import numpy as np 

class CelebAGenderDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.female_path = path + '/female/'
        self.male_path = path + '/male/'

        self.female_img_list = glob.glob(self.female_path + '/*.jpg')
        self.male_img_list = glob.glob(self.male_path + '/*.jpg')

        self.transform = transform

        self.img_list = self.female_img_list + self.male_img_list
        self.Image_list = []  
        for img_path in self.img_list:  
            self.Image_list.append(Image.open(img_path))  

        self.class_list = [0] * len(self.female_img_list) + [1] * len(self.male_img_list) 
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = self.Image_list[idx]  
        label = self.class_list[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

def make_celeba_dataloaders(path, dataset_length, train_ratio=0.8, val_ratio=0.1, \
                            resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], \
                            batch_size=4, num_workers=0): 

    transform = transforms.Compose(
        [
        transforms.Resize(resize),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean)
        ]
    )

    dataset = CelebAGenderDataset(path, transform=transform)
    dataset = Subset(dataset, np.random.choice(len(dataset), dataset_length))

    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    validation_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - validation_size

    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

    datasets = {'train': train_dataset, 'val': validation_dataset, 'test': test_dataset}

    print(f"Training Data Size : {len(train_dataset)}")
    print(f"Validation Data Size : {len(validation_dataset)}")
    print(f"Testing Data Size : {len(test_dataset)}")

    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers)
                for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}

    return dataloaders, dataset_sizes

