# Image-Similarity-Search

# Overview
This project implements an Image Similarity Search application that leverages powerful pre-trained deep learning models like ResNet, CLIP, DinoV2, and ViT to extract meaningful image embeddings. These embeddings are then used to perform similarity searches, helping users to find visually similar images from a custom dataset. The models have been fine-tuned on a custom dataset consisting of categories like watches and shoes, allowing for enhanced accuracy when identifying these specific objects.


# Features
  * Fine-tuned Models: Models like ResNet, CLIP, DinoV2, and ViT are fine-tuned on custom categories (watches and shoes).
  * Image Similarity Search: Search for visually similar images in the dataset based on a given query image.
  * Faiss Indexing: Efficient similarity search is achieved using Faiss for fast nearest neighbor retrieval.
  * Custom Dataset: The custom dataset includes images of watches and shoes, and has been curated to improve model performance in these categories.

![Screenshot (2921)](https://github.com/user-attachments/assets/29d72dec-3ebe-4b8a-ba6f-fac2c65b1071)
![Screenshot (2919)](https://github.com/user-attachments/assets/5146faa2-4c8d-4b2e-8968-4856ee31079f)
![Screenshot (2920)](https://github.com/user-attachments/assets/7b462d78-ad33-46f1-8806-9c56d828bfad)
![resnet](https://github.com/user-attachments/assets/197076d5-1e6e-4ee0-90ad-3031e150a848)

# Technologies Used
  * Deep Learning Models: ResNet, CLIP, DinoV2, ViT
  * Faiss: For fast similarity search and indexing.
  * Streamlit: For building the user interface.
  * PyTorch: For deep learning model handling and fine-tuning.
  * Transformers Library: For working with pre-trained models and processors.
  * PIL: For image loading and processing.
  * NumPy: For handling numerical operations.

# Setup and Installation
1. Clone the repository:
``` bash
   git clone https://github.com/yourusername/image-similarity-search.git

   cd image-similarity-search

```
2. Install the required dependencies:
``` bash
   pip install -r requirements.txt
```

3. Download and Preprocess Data:   
   Make sure your custom dataset (images of watches and shoes) is available. The models are fine-tuned on this dataset, and the embeddings are stored in Faiss indices.
   

5. Run the Streamlit Application:
``` bash
   streamlit run app.py
```
This will start the Streamlit app, allowing you to upload query images and perform similarity searches with the fine-tuned models.

# How It Works  
1. Embedding Extraction: The uploaded query image is passed through one of the fine-tuned models (ResNet, CLIP, DinoV2, or ViT) to extract its embedding.
2. Similarity Search: Using Faiss, the extracted embedding is compared against pre-computed embeddings of images in the dataset, and the most similar images are retrieved.
3. Visualization: The application displays the query image and the top-k most similar images, allowing users to visually compare them.

# Fine-Tuning Process  
1. Dataset Preparation: The dataset was curated by collecting images of watches and shoes.
2. Model Training: Each model (ResNet, CLIP, DinoV2, ViT) was fine-tuned using this custom dataset.
3. Saving Weights: The fine-tuned models were saved, and their weights are loaded during inference to ensure accurate results.


# Directory Structure
``` bash
├── app.py                # Streamlit application script
├── models/               # Directory containing fine-tuned models
│   ├── resnet_finetuned.pt
│   ├── clip_finetuned.pt
│   ├── dinov2_finetuned.pt
│   └── vit_finetuned.pt
├── data/                 # Custom dataset directory
│   ├── watches/
│   └── shoes/
├── faiss_indexes/        # Directory containing Faiss indices
│   ├── resnet.index
│   ├── clip.index
│   ├── dinov2.index
│   └── vit.index
└── requirements.txt      # Python dependencies

```
# Usage  
1. Upload an Image: Start by uploading a query image through the Streamlit interface.
2. Model Selection: Automatically Select  one of the pre-trained and fine-tuned models (ResNet, CLIP, DinoV2, or ViT) for similarity search.
3. View Results: View the top-k similar images retrieved from the dataset based on the query image’s embedding.

# Application Demo Link
https://image-similarity-search-ekwtcdyte3hcwgyqhsh25c.streamlit.app/


# Acknowledgments  
* Hugging Face Transformers: For providing the pre-trained models (ResNet, CLIP, DinoV2, ViT).
* Faiss: For enabling efficient similarity search and retrieval.
* Streamlit: For providing a simple interface to interact with the application.

# References
- Alexey Dosovitskiy, et al., AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE (2020), Arxiv
- Maxime Oquab, Timothée Darcet, Théo Moutakanni, et.al., DINOv2: Learning Robust Visual Features without Supervision (2023), Arxiv  
