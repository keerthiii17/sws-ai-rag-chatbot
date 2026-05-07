from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

query = "What is the leave policy?"

results = db.similarity_search(query, k=3)

for i, doc in enumerate(results):

    print("\n")
    print("=" * 50)

    print("Source:", doc.metadata["source"])
    print("Page:", doc.metadata["page"])

    print(doc.page_content[:500])