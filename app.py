import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os

st.set_page_config(page_title="Ohio Bird ID", page_icon="🐦")
st.title("🐦 Bird Species Identifier")

# Note: For deployment, ensure you upload a 'classes.txt' or hardcode the list
# For now, we use a placeholder list of 200 species
classes = [f"Species {i}" for i in range(200)] 

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 200)
    if os.path.exists('bird_model.pth'):
        model.load_state_dict(torch.load('bird_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
preprocess = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Upload a bird photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    inputs = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)
    st.success(f"Prediction: {classes[pred.item()]}")
