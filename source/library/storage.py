import os
import json
import fcntl
from langchain_community.vectorstores import FAISS
from source.library.retrieval import get_emebdding_model, embeddings_vector_store



def init_library(data_args):

    if os.path.exists(data_args.library_path):

        with open(data_args.library_path, 'r') as f:
            library= json.load(f)
        vector_store = FAISS.load_local(data_args.vector_store_path,
                                        embeddings=get_emebdding_model(data_args.embedding_model_name),
                                        allow_dangerous_deserialization=True)
        
        return library, vector_store
   
    return [], embeddings_vector_store(data_args.embedding_model_name)


def save_library(data_args,
                 library, 
                 vector_store):
    # Create a lock file path
    lock_file_path = f"{data_args.library_path}.lock"
    
    # Open the lock file
    with open(lock_file_path, 'w') as lock_file:
        try:
            # Acquire an exclusive lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            
            # Write to the library file
            with open(data_args.library_path, 'w') as f:
                json.dump(library, f, indent=2)
            
            # Save vector store
            vector_store.save_local(data_args.vector_store_path)
            
        finally:
            # Release the lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            # Remove the lock file
            try:
                os.remove(lock_file_path)
            except OSError:
                pass  # Ignore if file was already removed


