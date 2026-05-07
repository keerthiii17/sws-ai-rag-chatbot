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

    # Current question
    current_question = request.question.lower()

    # Previous user question
    previous_context = ""

    for item in reversed(conversation_history):

        if item.startswith("User:"):

            previous_context = item.replace(
                "User:",
                ""
            ).strip()

            break

    # Combined query
    combined_query = (
        previous_context + " " + current_question
    ).lower()

    question_words = combined_query.split()

    # Chunk scoring
    scored_chunks = []

    for chunk in chunks:

        text = chunk["text"].lower()

        score = 0

        for word in question_words:

            if word in text:
                score += 1

        # Smart boosting
        if "sick leave" in text and "sick" in combined_query:
            score += 5

        if "annual leave" in text and "leave" in combined_query:
            score += 3

        if "password" in text and "password" in combined_query:
            score += 5

        if "work from home" in text and "wfh" in combined_query:
            score += 5

        if score > 0:

            scored_chunks.append(
                (score, chunk)
            )

    # Sort highest relevance
    scored_chunks.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # Top retrieved chunks
    matched_chunks = [
        item[1]
        for item in scored_chunks[:6]
    ]

    # Context
    context = "\n\n".join([
        chunk["text"]
        for chunk in matched_chunks
    ])

    # Memory context
    memory_context = "\n".join(
        conversation_history[-6:]
    )

    # Prompt
    prompt = f"""
You are an internal company policy assistant.

STRICT RULES:
- Answer ONLY using the provided document context.
- Do NOT hallucinate or invent information.
- If information is unavailable, respond exactly:
"I don't have that information in the company documents."

ANSWER STYLE:
- Give professional and structured answers.
- Use bullet points when useful.
- Include related policy details ONLY if present in the context.
- Keep responses concise but informative.
- Mention only facts from company documents.

Conversation History:
{memory_context}

Document Context:
{context}

Employee Question:
{request.question}
"""

    # OpenRouter API request
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

    # Safe fallback
    try:
        answer = result["choices"][0]["message"]["content"]

    except:
        answer = "Error generating response."

    # Save memory
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