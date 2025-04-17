
from smolagents import Tool
from langchain_core.vectorstores import VectorStore




class SemanticRetrieverTool(Tool):
    name = "retriever_tool"
    description = (
        "Checks semantic similarity of the input question/query against known queries in the knowledge base. "
        "If any retrieved question/query is too similar to the new question/query (score > gamma_max) or too different "
        "(score < gamma_min), it automatically checks and raises ValueError (already built in). Otherwise, it returns the question/query with an acceptance message." \
        "This input must be a string (question/query)"
    )

    inputs = {
        "question": {
            "type": "string",
            "description": (
                "The question/query to check against the existing knowledge base."
            ),
        }
    }
    output_type = "string"
    def __init__(self, vectordb: VectorStore, gamma_max: float = 0.95, gamma_min: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb
        self.gamma_max= gamma_max
        self.gamma_min= gamma_min
        self.count = 0

    def forward(
        self,
        question: str,
    ) -> str:

        assert isinstance(question, str), "Your search query must be a string."

        if not len(self.vectordb.index_to_docstore_id):
            print(f'Question/Query: "{question}" is accepted (no existing records).')
            return question

        docs = self.vectordb.similarity_search_with_score(
            query=question,
            k=5,
        )

        doc_min,doc_max=[],[]
        for doc in docs :
            if doc[1] > self.gamma_max:
                doc_max.append(doc[0].page_content)
        
        if doc_max:
            raise ValueError(f"Wrong for question: {question}, Retrieved queries are too similar to recent question/query (Retrieved questions/queries  : {doc_max}) rewrite the question/query.")
    
        print(f'Question/Query: "{question}" is accepted (no existing records).')
        return question


