# app/predict.py
import torch
from torchvision import transforms
from PIL import Image
from utils import SimpleCNN

device = torch.device("cpu")
model = SimpleCNN()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict_image(image_path):
    img = Image.open(image_path).convert('L')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        prediction = output.argmax(dim=1).item()
    return prediction
