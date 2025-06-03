from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

from faiss import IndexFlatIP

import torch

def get_emebdding_model(model_name):

    device = "cpu"
    encode_kwargs = {"device":device,'normalize_embeddings':True}
    model_kwargs = {"trust_remote_code":True}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    return embeddings

def embeddings_vector_store(model_name):

    embeddings = get_emebdding_model(model_name)
    index = IndexFlatIP(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )

    return vector_store

