import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import TrafficSignCNN
from streamlit_cropper import st_cropper

st.set_page_config(page_title = "Traffic Sign Detection")

st.title("Traffic Sign Detection")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TrafficSignCNN().to(device)

model.load_state_dict(torch.load("traffic_sign_detection.pth",map_location = device))

model.eval()

transform = transforms.Compose([
  transforms.Resize((32,32)),
  transforms.ToTensor(),
  
])

CLASS_NAMES = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
    'No passing', 'No passing for vehicles over 3.5 metric tons',
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
    'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
    'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
    'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
    'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
    'Keep left', 'Roundabout mandatory', 'End of no passing',
    'End of no passing by vehicles over 3.5 metric tons'
]

uploaded = st.file_uploader(
  "Upload Traffic Sign Image",
  type = ["jpg", "jpeg", "png"]
)
camera_image = st.camera_input("Capture Sign")

image = None

if uploaded is not None:
  image = Image.open(uploaded).convert(RGB)
elif camera_image is not None:
  image = Image.open(camera_image).convert(RGB)

if image is not None:
  st.image(image, caption="Input Sign")
  img_tensor = transform(image).unsqueeze(0).to(device)  

  with torch.no_grad():
    
    output = model(img_tensor)
    probs = torch.softmax(output, 1)
    confidence, pred = torch.max(probs,1)

  conf = confidence.item()
  class_name = CLASS_NAMES[pred.item()]

  st.success(f"Prediction: {class_name}")
  st.write(f"Confidence:{conf * 100:.2f}%")
