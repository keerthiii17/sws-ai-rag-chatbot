from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI

load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load ChromaDB
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# MEMORY STORAGE
conversation_history = []

class ChatRequest(BaseModel):
    question: str

@app.post("/api/chat")
async def chat(request: ChatRequest):

    global conversation_history

    # Retrieve relevant chunks
    docs = db.similarity_search(
        request.question,
        k=4
    )

    context = "\n\n".join([
        doc.page_content for doc in docs
    ])

    # Last few messages memory
    memory_context = "\n".join(conversation_history[-6:])

    prompt = f"""
You are an internal company policy assistant.

Answer ONLY from the provided context.

If answer is unavailable, say:
"I don't have that information in the company documents."

Conversation History:
{memory_context}

Document Context:
{context}

Question:
{request.question}
"""

    response = llm.invoke(prompt)

    # Save conversation
    conversation_history.append(
        f"User: {request.question}"
    )

    conversation_history.append(
        f"Assistant: {response.content}"
    )

    # Sources
    sources = list(set([
        doc.metadata["source"]
        for doc in docs
    ]))

    return {
        "answer": response.content,
        "sources": sources
    }