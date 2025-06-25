import os
from argparse import ArgumentParser
from source.library.storage import init_library
from source.utils.args import DataArguments



def main():
    parser = ArgumentParser()
    parser.add_argument("--vector_store_path", type=str,  default="data/library/vector_store_step_debug")
    parser.add_argument("--embedding_model_name", type=str, default="Alibaba-NLP/gte-large-en-v1.5")
    args = parser.parse_args()

    # Wrap args into DataArguments object (assumes it accepts a dict or similar)
    data_args = DataArguments()
    data_args.vector_store_path = args.vector_store_path
    data_args.embedding_model_name = args.embedding_model_name
    
    # ✅ Load the vector store using your official project logic
    vectorstore = init_library(data_args)

    print("🔍 Listing all documents in the vector store...", data_args.vector_store_path)

    if hasattr(vectorstore, "docstore"):
        docstore = vectorstore.docstore
        total_docs = len(docstore._dict)
        print(f"📦 Total documents: {total_docs}")

        for i, (doc_id, doc) in enumerate(docstore._dict.items(), 1):
            print(f"\n--- Document {i} (ID: {doc_id}) ---")
            print("Content:", doc.page_content)
   
    else:
        print("⚠️ Could not access internal docstore. No documents to show.")

if __name__ == "__main__":
    main()