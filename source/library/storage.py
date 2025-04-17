


import os
import json
import random
from typing import Dict, List, Any
import numpy as np
import torch

from sentence_transformers import SentenceTransformer

from faiss import IndexFlatIP
from langchain_community.vectorstores import FAISS

from source.library.retrieval import get_emebdding_model, embeddings_vector_store


def load_sentence(name: str, hf_tokens:str ,device=None):
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    # Convert model name into a valid directory name
    model_dir = "models/" + name.replace("/", "_")
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Check if model is already saved
    if os.path.exists(model_dir):
        print(f"Loading model from disk: {model_dir}")
        try:
            model = SentenceTransformer(model_dir, device=device, trust_remote_code=True)
            return model
        except Exception as e:
            print(f"Error loading model from {model_dir}, redownloading...\n{e}")

    # Download and save the model
    print(f"Downloading model: {name}")
    model = SentenceTransformer(name, device=device, trust_remote_code=True)
    model.save(model_dir)
    
    return model


def init_library(library_path, 
                 vector_store_path='vector_store', 
                 model_name="Alibaba-NLP/gte-large-en-v1.5") -> List[Dict[str, Any]]:
    """Initialize or load the existing library"""

    if os.path.exists(library_path):
        with open(library_path, 'r') as f:
            library= json.load(f)
        vector_store = FAISS.load_local(vector_store_path, embeddings=get_emebdding_model(model_name) , allow_dangerous_deserialization=True)
        return library,vector_store
   
    return [], embeddings_vector_store(model_name)


def save_library(library: List[Dict[str, Any]], 
                 vector_store, 
                 library_path, 
                 vector_store_path='vector_store',):
    """Save the current library to disk"""

    with open(library_path, 'w') as f:
        json.dump(library, f, indent=2)
    vector_store.save_local(vector_store_path)


