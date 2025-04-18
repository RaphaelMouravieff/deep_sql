


import os
import json
from langchain_community.vectorstores import FAISS
from source.library.retrieval import get_emebdding_model, embeddings_vector_store



def init_library(data_args, training_args):

    if os.path.exists(data_args.library_path):

        with open(data_args.library_path, 'r') as f:
            library= json.load(f)
        vector_store = FAISS.load_local(data_args.vector_store_path,
                                        embeddings=get_emebdding_model(training_args.embedding_model_name),
                                        allow_dangerous_deserialization=True)
        
        return library, vector_store
   
    return [], embeddings_vector_store(training_args.embedding_model_name)


def save_library(data_args,
                 library, 
                 vector_store):

    with open(data_args.library_path, 'w') as f:
        json.dump(library, f, indent=2)
    vector_store.save_local(data_args.vector_store_path)


