from smolagents import Tool
from langchain_core.vectorstores import VectorStore

class SemanticRetrieverTool(Tool):
    name = "retriever_tool"
    description = (
        "Checks if the input question/query is too semantically similar to known queries in the database. "
        "If any retrieved query has a similarity score greater than gamma_max, a ValueError is raised. "
        "Otherwise, the input question/query is accepted. Input must be a string."
    )

    inputs = {
        "question": {
            "type": "string",
            "description": "The question or query to check against the knowledge base."
        }
    }

    output_type = "string"

    def __init__(self, vectordb: VectorStore, gamma_max: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb
        self.gamma_max = gamma_max
        self.too_similar_count = 0

    def forward(self, question: str) -> str:
        assert isinstance(question, str), "The input question must be a string."

        # If no documents are in the vector store, accept immediately
        if not self.vectordb.index_to_docstore_id:
            print(f'Question/Query: "{question}" is accepted (no existing records).')
            return question

        # Check for semantic similarity
        docs = self.vectordb.similarity_search_with_score(query=question, k=5)

        too_similar = [doc[0].page_content for doc in docs if doc[1] > self.gamma_max]

        if too_similar:
            self.too_similar_count += 1
            raise ValueError(
                f"Rejected question: '{question}' is too similar to existing queries: {too_similar}"
            )

        print(f'Question/Query: "{question}" is accepted.')
        return question


