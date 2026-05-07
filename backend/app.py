from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

import requests
import os
import json

load_dotenv()

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load chunks
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Conversation memory
conversation_history = []

class ChatRequest(BaseModel):
    question: str

@app.post("/api/chat")
async def chat(request: ChatRequest):

    global conversation_history

    # User question words
    question_words = request.question.lower().split()

    # Score chunks
    scored_chunks = []

    for chunk in chunks:

        text = chunk["text"].lower()

        score = 0

        for word in question_words:

            if word in text:
                score += 1

        if score > 0:

            scored_chunks.append(
                (score, chunk)
            )

    # Sort by highest score
    scored_chunks.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # Retrieve top chunks
    matched_chunks = [
        item[1]
        for item in scored_chunks[:6]
    ]

    # Context
    context = "\n\n".join([
        chunk["text"]
        for chunk in matched_chunks
    ])

    # Conversation memory
    memory_context = "\n".join(
        conversation_history[-6:]
    )

    # Prompt
    prompt = f"""
You are an internal company policy assistant.

STRICT RULES:
- Answer ONLY using the provided document context.
- Do NOT invent or hallucinate information.
- If the answer is not available in the context, respond exactly:
"I don't have that information in the company documents."

ANSWER STYLE:
- Give professional and structured answers.
- Use bullet points or sections when useful.
- Include related policy details ONLY if they exist in the context.
- Keep the response concise but informative.
- Mention only facts from the provided documents.

Conversation History:
{memory_context}

Document Context:
{context}

Employee Question:
{request.question}
"""

    # OpenRouter request
    response = requests.post(

        url="https://openrouter.ai/api/v1/chat/completions",

        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },

        json={

            "model": "openai/gpt-3.5-turbo",

            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        }

    )

    result = response.json()

    answer = result["choices"][0]["message"]["content"]

    # Save conversation memory
    conversation_history.append(
        f"User: {request.question}"
    )

    conversation_history.append(
        f"Assistant: {answer}"
    )

    # Clean source names
    sources = list(set([

        chunk["source"]
        .replace(".pdf", "")
        .replace("SWS-AI-", "")
        .replace("-", " ")
        .title()

        for chunk in matched_chunks

    ]))

    return {
        "answer": answer,
        "sources": sources
    }