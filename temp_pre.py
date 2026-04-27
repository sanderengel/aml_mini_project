import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


class DriverDataset(Dataset):
    def __init__(self, metadata, root_dir="data/imgs/train"):
        self.metadata = metadata.reset_index(drop=True)
        self.root_dir = root_dir

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(0.2, 0.2),
            transforms.ToTensor(),
            
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        img_name = row["img"]
        class_name = row["classname"]

        img_path = os.path.join(self.root_dir, class_name, img_name)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = int(class_name[1])

        return image, label



def get_data_loader(metadata, batch_size=32, shuffle=True):
    dataset = DriverDataset(metadata)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )