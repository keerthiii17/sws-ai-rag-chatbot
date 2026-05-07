import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = []

# Load PDFs
for file in os.listdir(DATA_PATH):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(DATA_PATH, file)

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        for i, doc in enumerate(docs):

            doc.metadata["source"] = file
            doc.metadata["page"] = i + 1

        documents.extend(docs)

print(f"Loaded {len(documents)} pages")

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

# Add chunk ids
for i, chunk in enumerate(chunks):

    chunk.metadata["chunk_id"] = i

print(f"Created {len(chunks)} chunks")

# Store in ChromaDB
db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_PATH
)

db.persist()

print("Chroma DB created successfully")