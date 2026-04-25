from fastapi import UploadFile, File, HTTPException, APIRouter
from torchvision import transforms
from db.database import get_db
from db.models import MnistModel
from sqlalchemy.orm import Session
from fastapi import Depends
import torch.nn as nn
from PIL import Image
import torch
import io

class_names = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

mnist_router = APIRouter(prefix='/mnist_numbers', tags=['Mnist_numbers'])
class CheckImage(nn.Module):
    def __init__(self):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckImage()
model.load_state_dict(torch.load('mnist_numbers.pth', map_location=device))
model.to(device)
model.eval()

@mnist_router.post('/predict/')
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

        db_mnist = MnistModel(image=image_data.hex(), label=str(pred))
        db.add(db_mnist)
        db.commit()
        db.refresh(db_mnist)

        return {
            'Класстын саны': pred,
            'Класстын аталышы': class_names[pred]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))