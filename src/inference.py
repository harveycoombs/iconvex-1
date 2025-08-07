import torch
from torchvision import transforms
from PIL import Image
from model import IConvexOne

model = IConvexOne(num_classes=4)
model.load_state_dict(torch.load("iconvexone.pth", map_location="cpu"))
model.eval()

classes = ["birds", "cats", "dogs", "humans"]

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

image_path = input("Image path: ")
prediction = predict_image(image_path)
print(f"Prediction: {prediction}")