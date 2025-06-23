

from source.library.storage import  init_library, save_library
from types import SimpleNamespace
from source.tools.semantic_retriver import SemanticRetrieverTool


# Simulate args
data_args0 = SimpleNamespace(
    library_path="data/uni_test_todel/test_library_chunk0.json",
    vector_store_path="data/uni_test_todel/test_vector_store",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)
data_args1 = SimpleNamespace(
    library_path="data/uni_test_todel/test_library_chunk1.json",
    vector_store_path="data/uni_test_todel/test_vector_store",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)

data_args2 = SimpleNamespace(
    library_path="data/uni_test_todel/test_library_chunk2.json",
    vector_store_path="data/uni_test_todel/test_vector_store",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)

for data_args in [data_args0,data_args1,data_args2]:

    library, vector_store = init_library(data_args)

    tool = SemanticRetrieverTool(vector_store, gamma_max=0.9)

    # Step 3: Try adding a question
    test_question = "What is the capital of France?"

    try:
        # This will raise an error if too similar
        tool.forward(test_question)

        # Add to library and vector store only if accepted
        doc_id = str(len(library))
        library.append(test_question)
        vector_store.add_texts([test_question], ids=[doc_id])
        print(f"✅ Added: {test_question}")

    except ValueError as e:
        print(f"❌ Duplicate Detected: {e}")

    # Step 4: Save updated library/vector store
    save_library(data_args, library, vector_store)


