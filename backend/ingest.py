import os
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data"

documents = []

# Load all PDFs
for file in os.listdir(DATA_PATH):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(DATA_PATH, file)

        print(f"Loading: {file}")

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        # Add metadata
        for i, doc in enumerate(docs):

            doc.metadata["source"] = file
            doc.metadata["page"] = i + 1

        documents.extend(docs)

print(f"\nLoaded {len(documents)} pages")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

# Save chunks
chunk_data = []

for chunk in chunks:

    chunk_data.append({

        "text": chunk.page_content,

        "source": chunk.metadata["source"],

        "page": chunk.metadata["page"]

    })

# Save as JSON
with open("chunks.json", "w", encoding="utf-8") as f:

    json.dump(
        chunk_data,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\nchunks.json created successfully")