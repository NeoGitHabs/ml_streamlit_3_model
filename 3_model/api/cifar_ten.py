from fastapi import UploadFile, File, HTTPException, APIRouter
from torchvision import transforms
from db.database import get_db
from db.models import CifarModel
from sqlalchemy.orm import Session
from fastapi import Depends
import torch.nn as nn
from PIL import Image
import torch
import io

cifar_ten_router = APIRouter(prefix='/cifar_10', tags=['Cifar_10'])


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class CifarmClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, image):
        image = self.first(image)
        image = self.second(image)
        return image

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CifarmClassification()
model.load_state_dict(torch.load('cifar_10_model.pth', map_location=device))
model.to(device)
model.eval()


@cifar_ten_router.post('/predict/')
async def check_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=400, detail='Файл кошулган жок')

        img = Image.open(io.BytesIO(image_data))
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)
            pred = y_pred.argmax(dim=1).item()

        db_mnist = CifarModel(image=image_data.hex(), label=str(pred))
        db.add(db_mnist)
        db.commit()
        db.refresh(db_mnist)

        return {
            'Класстын саны': pred,
            'Класстын аталышы': class_names[pred]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))