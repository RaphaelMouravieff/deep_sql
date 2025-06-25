


import os
import json
from langchain_community.vectorstores import FAISS
from source.library.retrieval import get_emebdding_model, embeddings_vector_store
from filelock import FileLock
import time
from langchain_core.documents import Document
from uuid import uuid4
from source.utils.logger import setup_logger
logger = setup_logger(__name__)

def init_library(data_args):
    vec_exists = os.path.exists(data_args.vector_store_path)

    if  vec_exists:
        vectorstore = FAISS.load_local(data_args.vector_store_path,
                                        embeddings=get_emebdding_model(data_args.embedding_model_name),
                                        allow_dangerous_deserialization=True)
        logger.info('🆕 Loaded existing library and vector store.')

        docstore = vectorstore.docstore
        total_docs = len(docstore._dict)
        logger.info("📦 Total documents: %d", total_docs)


        return vectorstore


    else:
        logger.info("🆕 Creating new vector store and empty library.")
        return embeddings_vector_store(data_args.embedding_model_name)


def save_library(data_args, entries):
    """
    Safely appends new entries to the JSON library file using file locking.
    
    Args:
        data_args: an object with a .library_path attribute.
        entries: a list of dicts to append to the library.
    """
    start_time = time.time()
    lock_path = f"{data_args.library_path}.lock"

    with FileLock(lock_path):
        # Step 1: Load existing library if it exists
        if os.path.exists(data_args.library_path):
            try:
                with open(data_args.library_path, 'r') as f:
                    existing_library = json.load(f)
            except json.JSONDecodeError:
                logger.warning("⚠️ Warning: JSON file is empty or corrupted. Starting fresh.")
                existing_library = []
        else:
            existing_library = []

        # Step 2: Append new entries
        updated_library = existing_library + entries

        # Step 3: Save the updated library
        with open(data_args.library_path, 'w') as f:
            json.dump(updated_library, f, indent=2)

    end_time = time.time()
    duration = end_time - start_time
    logger.info("🕒 save_library completed in %.3f seconds.", duration)



def save_vector_store(data_args, entries):
    """
    Safely appends new entries (as Documents) to the FAISS vector store.
    
    Args:
        data_args: contains .vector_store_path and .embedding_model_name
        entries: list of dicts, each containing at least a "question"
    """
    start_time = time.time()
    lock_path = f"{data_args.vector_store_path}.lock"

    with FileLock(lock_path):
        # Step 1: Load or initialize vector store
        if os.path.exists(data_args.vector_store_path):
            vector_store = FAISS.load_local(
                data_args.vector_store_path,
                embeddings=get_emebdding_model(data_args.embedding_model_name),
                allow_dangerous_deserialization=True
            )
        else:
            vector_store = embeddings_vector_store(data_args.embedding_model_name)

        # Step 2: Create Documents from entries
        new_documents = []
        ids = []

        for entry in entries:
            if "question" not in entry:
                continue
            new_documents.append(Document(entry["question"]))
            ids.append(str(uuid4()))

        if new_documents:
            vector_store.add_documents(documents=new_documents, ids=ids)
            vector_store.save_local(data_args.vector_store_path)
            logger.info("✅ Added %d documents to vector store.", len(new_documents))
        else:
            logger.warning("⚠️ No valid questions found in entries.")

    end_time = time.time()
    logger.info("🕒 save_vector_store completed in %.3f seconds.", end_time - start_time)