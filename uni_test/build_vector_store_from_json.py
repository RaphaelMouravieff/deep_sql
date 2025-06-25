import os
import json
from uuid import uuid4
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from source.library.retrieval import get_emebdding_model, embeddings_vector_store



def init_library(vector_store_path, embedding_model_name):
    vec_exists = os.path.exists(vector_store_path)

    if  vec_exists:
        vectorstore = FAISS.load_local(vector_store_path,
                                        embeddings=get_emebdding_model(embedding_model_name),
                                        allow_dangerous_deserialization=True)
        print('🆕 Loaded existing library and vector store.')

        docstore = vectorstore.docstore
        total_docs = len(docstore._dict)
        print(f"📦 Total documents: {total_docs}")


        return vectorstore


    else:
        print("🆕 Creating new vector store and empty library.")
        return embeddings_vector_store(embedding_model_name)
    

def main():
    folder_path = "/home/raphael.gervillie/deep_sql/data/library/library_step0"
    vector_store_output_path = "/home/raphael.gervillie/deep_sql/data/library/vector_store_step"
    embedding_model_name = "Alibaba-NLP/gte-large-en-v1.5"

    
    
    total_files = 30
    total_questions = 0

    print(f"📁 Loading {total_files} JSON files from: {folder_path}\n")

    for i in tqdm(range(total_files), desc="📦 Processing files"):

        vector_store = init_library(vector_store_output_path, embedding_model_name)

        documents, ids = [], []
        file_name = f"library_step_chunk{i}_30.json"
        file_path = os.path.join(folder_path, file_name)

        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_name} — skipping.")
            continue

        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"❌ Error decoding JSON in: {file_name} — skipping.")
                continue

        valid_count = 0
        for entry in data:
            if "question" in entry:
                documents.append(Document(page_content=entry["question"]))
                ids.append(str(uuid4()))
                valid_count += 1

        print(f"✅ {file_name}: {valid_count} questions loaded.")
        total_questions += valid_count

        print("🧠 Adding documents to vector store...")
        vector_store.add_documents(documents=documents, ids=ids)
        vector_store.save_local(vector_store_output_path)

    print(f"\n🔢 Total questions collected: {total_questions}")

    if not documents:
        print("⚠️ No valid questions found across all chunks.")
        return



    print(f"\n✅ Finished! {len(documents)} documents saved.")
    print(f"📦 Vector store location: {vector_store_output_path}")

if __name__ == "__main__":
    main()

    