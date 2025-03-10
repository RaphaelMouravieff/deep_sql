


import os
import json
import random
from typing import List
import faiss
from source.library.utils import load_sentence
import numpy as np


class SQLLibrary:
    def __init__(self, data_args, model_args):

        library_path = data_args.library_path
        # Load from disk or init empty
        if os.path.exists(library_path):
            with open(library_path, "rb", encoding="utf-8") as f:
                self.storage = json.load(f)
        else:
            self.storage = {}

        self.library_path = library_path

        dim = 1024  #  the vector dimension 
        if len(self.storage)==0:
            self.vect_index = faiss.IndexFlatIP(dim)# ini

        else :self.vect_index = faiss.IndexFlatIP(dim)# to be done 

        self.selected_index=[] # list of random selected index of sql skills 
        self.selected_ret_index = [] # list of retrieved selected index of sql skills ]
        
        self.model = load_sentence(model_args.sentence_model_name_or_path, model_args.hf_tokens)
        
    def __repr__(self):
        return self.storage

    def __len__(self):
        return len(self.storage)


    def save(self):
        with open(self.library_path, "wb", encoding="utf-8") as f:
            json.dump(self.storage, f, indent=2)
    
    def get_sql(self,random_=True,num_q=5):
        n_skill=len(self.storage)
        if n_skill==0:
            return ["SELECT * \nFROM [table];" ]
        if random_: 
            self.selected_index = random.sample(list(self.storage.keys()),k=min(num_q,n_skill
                                                                )
                                                ) 
            selected_sql = [self.storage[i]["sql"] for  i in  self.selected_index]
        else:
            n_skill-=1
            selected_sql = [ self.storage[i] for i in range(n_skill,n_skill-num_q,-1) ]
            self.selected_index = list(range(n_skill,n_skill-num_q,-1) )
        
        return selected_sql

    def add_query(self, sql: str, python_func:str=None,save:bool=False)-> None:
        if False:

            embedding_vec = self.compute_embedding(sql)
            
            print(f"Embedding vector shape {embedding_vec.shape} : {embedding_vec}\n\n")
            self.storage[len(self.storage)] = {
                "sql" : sql,
                "embedding": embedding_vec,
                "python_func": python_func
            }
            self.vect_index.add(embedding_vec)
            if save or len(self.storage)%100==0:
                self.save()
            return None


    


    def get_queries(self, embedding_vec: List[float], top_k: int = 10, throushold:int=.9,) -> List[str]:
        """
        Return up to top_k skill names sorted by similarity (desc).
        """
        # If skill library is empty, return empty
        if not self.storage:
            return []
        
        self.selected_ret_index=[]
        # Compute similarity
        sims, ret_index = self.vect_index.search(embedding_vec, k=top_k)
        sims = sims[0]# because only one query
        self.selected_ret_index=ret_index[0]# because only one query
        self.selected_ret_index = [ self.selected_ret_index[i] for i in range(len(sims)
                                                  )  if (sims[i] > throushold )]
        ret_sql = [ self.storage[i]['sql'] for i in self.selected_ret_index  ] if len(self.selected_ret_index)>0 else []
        
        return ret_sql
    



    def compute_embedding(self,text: str) -> List[float]:
        """
        Compute embedding (e.g. using sentence-transformers or OpenAI embeddings).
        Return a vector as a list of floats.
        """
        # Example pseudocode:
        # from sentence_transformers import SentenceTransformer
        embedding = self.model.encode([text], convert_to_tensor=False
                        ,batch_size=32,show_progress_bar=False,normalize_embeddings=True)
        

        return np.array(embedding, dtype=np.float32)