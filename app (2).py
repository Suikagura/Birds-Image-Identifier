import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os
import gdown

st.set_page_config(page_title="Ohio Bird ID", page_icon="🐦")
st.title("🐦 Bird Species Identifier")

# --- GOOGLE DRIVE CONNECTION ---
# Extracted ID from your provided link
GOOGLE_DRIVE_FILE_ID = '1pOijydumjJeo_rO8DNiuFZFjpbEEN3ne'
MODEL_PATH = 'bird_model.pth'

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Downloading model from Google Drive... This may take a minute.'):
            url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
    
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 200)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        st.error("Model file not found. Please ensure the Google Drive link is set to 'Anyone with the link can view'.")
        return None

# --- CLASS LABELS ---
bird_labels = ['Black footed Albatross', 'Laysan Albatross', 'Sooty Albatross', 'Groove billed Ani', 'Crested Auklet', 'Least Auklet', 'Parakeet Auklet', 'Rhinoceros Auklet', 'Brewer Blackbird', 'Red winged Blackbird', 'Rusty Blackbird', 'Yellow headed Blackbird', 'Bobolink', 'Indigo Bunting', 'Lazuli Bunting', 'Painted Bunting', 'Cardinal', 'Spotted Catbird', 'Gray Catbird', 'Yellow breasted Chat', 'Eastern Towhee', 'Chuck will widow', 'Brandt Cormorant', 'Red faced Cormorant', 'Pelagic Cormorant', 'Bronzed Cowbird', 'Shiny Cowbird', 'Brown Creeper', 'American Crow', 'Fish Crow', 'Black billed Cuckoo', 'Mangrove Cuckoo', 'Yellow billed Cuckoo', 'Gray crowned Rosy Finch', 'Purple Finch', 'Northern Flicker', 'Acadian Flycatcher', 'Great Crested Flycatcher', 'Least Flycatcher', 'Olive sided Flycatcher', 'Scissor tailed Flycatcher', 'Vermilion Flycatcher', 'Willow Flycatcher', 'Yellow bellied Flycatcher', 'Frigatebird', 'Northern Fulmar', 'Gadwall', 'Anna Hummingbird', 'Ruby throated Hummingbird', 'Rufous Hummingbird', 'Green Kingfisher', 'Belted Kingfisher', 'Pied Kingfisher', 'Ringed Kingfisher', 'White breasted Kingfisher', 'Red legged Kittiwake', 'Black legged Kittiwake', 'Sayornis', 'American Pipit', 'Common Tern', 'Arctic Tern', 'Forsters Tern', 'Least Tern', 'Green tailed Towhee', 'Brown Thrasher', 'Sage Thrasher', 'Great Grey Shrike', 'Loggerhead Shrike', 'Great Crested Grebe', 'Horned Grebe', 'Eared Grebe', 'Pied billed Grebe', 'Western Grebe', 'Western Gull', 'Glaucous winged Gull', 'California Gull', 'Ring billed Gull', 'Ivory Gull', 'Heermann Gull', 'Herring Gull', 'Least Auklet', 'Bald Eagle', 'Common Raven', 'White necked Raven', 'American Crow', 'Fish Crow', 'Cliff Swallow', 'Barn Swallow', 'Bank Swallow', 'Tree Swallow', 'Violet green Swallow', 'Rough winged Swallow', 'American Goldfinch', 'House Finch', 'Common Redpoll', 'Pine Siskin', 'Evening Grosbeak', 'Pine Grosbeak', 'Rose breasted Grosbeak', 'Blue Grosbeak', 'Kentucky Warbler', 'Magnolia Warbler', 'Cape May Warbler', 'Myrtle Warbler', 'Nashville Warbler', 'Orange crowned Warbler', 'Palm Warbler', 'Pine Warbler', 'Prairie Warbler', 'Tennessee Warbler', 'Wilson Warbler', 'Worm eating Warbler', 'Yellow Warbler', 'Yellow breasted Chat', 'Yellow throated Warbler', 'Western Meadowlark', 'Eastern Meadowlark', 'Scott Oriole', 'Orchard Oriole', 'Hooded Oriole', 'Bullock Oriole', 'Baltimore Oriole', 'White throated Sparrow', 'White crowned Sparrow', 'Vesper Sparrow', 'Tree Sparrow', 'Song Sparrow', 'Savannah Sparrow', 'Lincoln Sparrow', 'Le Conte Sparrow', 'House Sparrow', 'Henslow Sparrow', 'Grasshopper Sparrow', 'Fox Sparrow', 'Field Sparrow', 'Chipping Sparrow', 'Cape Sable Sparrow', 'Brewers Sparrow', 'Black throated Sparrow', 'Baird Sparrow', 'American Tree Sparrow', 'Summer Tanager', 'Scarlet Tanager', 'Prothonotary Warbler', 'Parula Warbler', 'Northern Waterthrush', 'Louisiana Waterthrush', 'Lutescent Warbler', 'Hooded Warbler', 'Golden winged Warbler', 'Cerulean Warbler', 'Canada Warbler', 'Blue winged Warbler', 'Blackpoll Warbler', 'Blackburnian Warbler', 'Black throated Gray Warbler', 'Black throated Blue Warbler', 'Black and White Warbler', 'Bay breasted Warbler', 'American Redstart', 'American Pipit', 'Common Yellowthroat', 'Yellow headed Blackbird', 'Yellow breasted Chat', 'Yellow bellied Sapsucker', 'White breasted Nuthatch', 'Western Wood Pewee', 'Western Kingbird', 'Warbling Vireo', 'Vesper Sparrow', 'Upland Sandpiper', 'Tree Swallow', 'Townsend Solitaire', 'Tennessee Warbler', 'Swainson Thrush', 'Summer Tanager', 'Stilt Sandpiper', 'Spotted Sandpiper', 'Song Sparrow', 'Slaty backed Gull', 'Short billed Dowitcher', 'Sharp tailed Sparrow', 'Seaside Sparrow', 'Savannah Sparrow', 'Sanderling', 'Ruby crowned Kinglet', 'Royal Tern', 'Rough winged Swallow', 'Rose breasted Grosbeak', 'Rock Wren', 'Ring billed Gull', 'Red winged Blackbird', 'Red eyed Vireo', 'Red breasted Nuthatch', 'Red breasted Merganser', 'Pied billed Grebe', 'Philadelphia Vireo', 'Palm Warbler', 'Ovenbird', 'Orange crowned Warbler', 'Northern Waterthrush']

model = download_and_load_model()

if model:
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
            idx = pred.item()

        bird_name = bird_labels[idx]
        st.success(f"Prediction: **{bird_name}**")
