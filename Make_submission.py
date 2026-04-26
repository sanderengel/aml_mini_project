import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import get_model
from utils import get_device


TEST_DIR = "data/imgs/test"
MODEL_PATH = "best_driver_model.pth"
OUTPUT_CSV = "submission.csv"

classes = [f"c{i}" for i in range(10)]


# -------------------
# Dataset
# -------------------
class TestDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.images = [
            f for f in sorted(os.listdir(img_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, img_name


def main():
    device = get_device()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load model
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Dataset + DataLoader
    dataset = TestDataset(TEST_DIR, transform)

    loader = DataLoader(
        dataset,
        batch_size=64,     
        shuffle=False,
        num_workers=4,      
        pin_memory=True
    )

    rows = []

    with torch.no_grad():
        for images, img_names in tqdm(loader, desc="Inference"):
            images = images.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for i in range(len(img_names)):
                row = {"img": img_names[i]}
                for cls, prob in zip(classes, probs[i]):
                    row[cls] = prob
                rows.append(row)

    submission = pd.DataFrame(rows)
    submission = submission[["img"] + classes]
    submission.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    main()