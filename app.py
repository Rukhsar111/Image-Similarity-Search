import streamlit as st
import torch
import numpy as np
from PIL import Image
import faiss
from transformers import (AutoImageProcessor, AutoModel, Dinov2Model, ViTModel, CLIPProcessor, CLIPModel, ResNetForImageClassification)
import matplotlib.pyplot as plt
import os

import sys
from pathlib import Path

# dir = Path(__file__).resolve()
# sys.path.append(str(dir.parent.parent))

BASE_DIR = Path(__file__).parent
# data_path = BASE_DIR / "data/merged"

# Base class for embedding extraction
class BaseEmbeddingExtractor:
    def __init__(self, image):
        self.image = image if isinstance(image, Image.Image) else Image.open(image)
        
    def extract_embedding(self) -> torch.Tensor:
        raise NotImplementedError

# Individual models for extracting embeddings
class DinoV2(BaseEmbeddingExtractor):
    def extract_embedding(self) -> torch.Tensor:
        image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = Dinov2Model.from_pretrained("facebook/dinov2-base")
        inputs = image_processor(self.image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(1)
        return embedding

class Clip(BaseEmbeddingExtractor):
    def extract_embedding(self) -> torch.Tensor:
        image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        inputs = image_processor(images=self.image, return_tensors='pt', padding=True)
        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
        return embedding

class VIT(BaseEmbeddingExtractor):
    def extract_embedding(self) -> torch.Tensor:
        image_processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")
        model = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k")
        inputs = image_processor(self.image, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(1)
        return embedding

class Resnet18(BaseEmbeddingExtractor):
    def extract_embedding(self) -> torch.Tensor:
        image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
        
        # Load your custom trained weights
        state_dict = torch.load("resnet_finetuned.pt", map_location=torch.device('cpu'))
        
        # Load the state dictionary
        model.load_state_dict(state_dict,strict=False)


        inputs = image_processor(self.image, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[-1]
        embedding = torch.mean(embedding, dim=[2,3])
        return embedding

# Function to get appropriate extractor
def get_embedding_extractor(model_name: str, image) -> BaseEmbeddingExtractor:
    extractors = {
        'resnet': Resnet18,
        'vit': VIT,
        'dino': DinoV2,
        'clip': Clip
    }
    model_name = model_name.lower()
    if model_name not in extractors:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(extractors.keys())}")
    
    return extractors[model_name](image)

# Function to extract embeddings
def extract_embedding(model_name: str, image) -> torch.Tensor:
    extractor = get_embedding_extractor(model_name, image)
    return extractor.extract_embedding()

# Load Faiss index and image paths
def load_faiss_index(index_path):
    # Load the Faiss index from file
    index = faiss.read_index(index_path)
    
    # Load the image paths from the corresponding '.paths' file
    with open(index_path + '.paths', 'r') as f:
        image_paths = [line.strip() for line in f]
    
    # Return both index and image paths
    return index, image_paths

# Retrieve similar images from Faiss
def retrieve_similar_images(query, model, index, image_paths, top_k=3):
    # Extract embedding for the query image
    embeddings = extract_embedding(model, query)
    print(embeddings.shape)

    
    # Perform Faiss search to get top-k similar images
    distances, indices = index.search(embeddings, top_k)
    
    # Retrieve image paths using the indices
    retrieved_images = [image_paths[int(idx)] for idx in indices[0]]
    
    return retrieved_images

# Function to visualize results
# def visualize_results(query, retrieved_images):
#     # Display query image
#     st.image(query, caption="Query Image", use_column_width=True)
    
#     # Display retrieved similar images
#     for idx, img_path in enumerate(retrieved_images):
#         img = Image.open(img_path)
#         st.image(img, caption=f"Match {idx + 1}", use_column_width=True)

import streamlit as st
from PIL import Image


import requests

def download_image(img_path):
    local_path = os.path.join(BASE_DIR, img_path)
    # https://raw.githubusercontent.com/<Rukhsar111>/<Image-Similarity-Search.git>/<main>/data/merged/
    github_repo_url = "https://raw.githubusercontent.com/<Rukhsar111>/<Image-Similarity-Search.git>/<main>/data/merged/"
    img_url = github_repo_url + img_path
    
    if not os.path.exists(local_path):
        response = requests.get(img_url)
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {img_path}")
    return local_path


def visualize_results(query_image, retrieved_images, container_width=5):
    """
    Visualize the query image and the retrieved similar images in a grid layout.

    Args:
    - query: Path or image object for the query image.
    - retrieved_images: List of paths or image objects for the retrieved images.
    - columns_per_row: Number of columns to display per row in the grid.
    """
    # Display the query image
    # github_repo_url = "https://raw.githubusercontent.com/<Rukhsar111>/<Image-Similarity-Search.git>/<main>/data/merged/"
    st.header(f" Retrived Simailar Images Using : {model.upper()}")

    # Display the retrieved images in a grid
    for i in range(0, len(retrieved_images), container_width):
        cols = st.columns(container_width)
        for j, col in enumerate(cols):
            if i + j < len(retrieved_images):
                img_path = retrieved_images[i + j]
                img_path=download_image(img_path=img_path)
                print('new_path', img_path)
                img = Image.open(img_path)
                col.image(img_path, caption=f"Match {i + j + 1}", use_container_width=True)

   
        

# Streamlit interface
def main():
    global model

    # Inject CSS for a black background
    st.markdown(
        """
        <style>
        /* Entire app background */
        .stApp {
            background-color: grey;
            color: white;
        }

        /* Optional: Adjust text and elements for better visibility */
        div, span, p, h1, h2, h3, h4, h5, h6 {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Image Similarity Search")
    
    #Choose each model and extract their feature vectors.
    available_models=['resnet','vit','dino','clip']

    # Upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        query_image = Image.open(uploaded_image)
        # query_image = query_image.resize((100, 100))
        st.header(f"Query Image")
        st.image(query_image, caption="Uploaded Image", use_container_width=True)
        
        # # Select model
        # model_choice = st.selectbox("Select a model for similarity search", ['resnet', 'vit', 'dino', 'clip'])
        
        # # Load Faiss index for the selected model
        # index_path = f'faiss_indexes/{model_choice}.index'
        # try:
        #     index, image_paths = load_faiss_index(index_path)
        #     st.success(f"Faiss index loaded for {model_choice} model.")
        # except Exception as e:
        #     st.error(f"Error loading Faiss index for {model_choice}: {str(e)}")
        #     return

        for model in available_models:
            print(f'\n start extracting embeddings for {model}')


            # model='resnet'
            OUTPUT_INDEX_PATH=f'faiss_indexes/{model}.index'
            index, image_paths = load_faiss_index(OUTPUT_INDEX_PATH)
            print('index_loaded_successfuy')


            # Perform the similarity search
            retrieved_images = retrieve_similar_images(query_image, model, index, image_paths, top_k=6)
            print('retrieved_images',  retrieved_images)
        
            # Visualize the results
            visualize_results(query_image, retrieved_images,container_width=5)




if __name__ == "__main__":
    #call the main function to execue the code.
    main()
