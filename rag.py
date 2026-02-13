import base64
from openai import OpenAI
from embeddings import (
    embed_query_for_text,
    embed_query_for_image
)
from vectore_store import query_text, query_image
from dotenv import load_dotenv
from langsmith import traceable

from cache_store import (
    get_cached_embedding,
    save_embedding,
    get_cached_answer,
    save_answer
)
import os 

os.environ['LANGCHAIN_PROJECT'] = "multimodal-rag"

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = """
You are a research assistant.
Answer strictly using ONLY the provided context.
If query is to explain or describe then describe everything so that the user understands clearly, especially if an image is provided.
and also answer in a way that is helpful for researchers, providing page numbers when referencing text.
and also if require answer in points whenever relevent.
If the answer is not found, say "I don't know".
If an image is provided, explain it clearly when relevant.
Be structured, concise, and factual.
Do not use external knowledge.
"""


def needs_image(query: str) -> bool:
    keywords = ["figure", "fig", "diagram", "image", "visual", "show"]
    return any(word in query.lower() for word in keywords)



@traceable
def answer(query: str, mode="fast"):

    # ----------------------------
    # 0️⃣ ANSWER CACHE CHECK
    # ----------------------------
    cached_answer = get_cached_answer(query)
    if cached_answer:
        return cached_answer  # (response_text, image_base64)


    # ----------------------------
    # 1️⃣ TEXT EMBEDDING (WITH CACHE)
    # ----------------------------
    text_query_vec = get_cached_embedding(query, "text")

    if not text_query_vec:
        text_query_vec = embed_query_for_text(query)
        save_embedding(query, text_query_vec, "text")

    text_results = query_text(text_query_vec)

    context = ""
    if text_results and text_results.get("metadatas"):
        for r in text_results["metadatas"][0]:
            context += f"(Page {r.get('page', 'N/A')}): {r.get('content', '')}\n"

    # If no context retrieved → stop
    if not context.strip():
        return "I don't know", None


    # ----------------------------
    # 2️⃣ IMAGE RETRIEVAL (SAFE)
    # ----------------------------
    image_query_vec = None
    image_base64 = None

    if needs_image(query):

        image_query_vec = get_cached_embedding(query, "image")

        if not image_query_vec:
            image_query_vec = embed_query_for_image(query)
            save_embedding(query, image_query_vec, "image")

        image_results = query_image(image_query_vec)

        if image_results and image_results.get("metadatas"):
            for r in image_results["metadatas"][0]:
                if "image_base64" in r:
                    image_base64 = r["image_base64"]
                    break


    # ----------------------------
    # 3️⃣ MODEL SELECTION
    # ----------------------------
    model_name = "gpt-4o-mini" if mode == "fast" else "gpt-4o"


    # ----------------------------
    # 4️⃣ MULTIMODAL CALL
    # ----------------------------
    if image_base64:

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Context:\n{context}\n\nQuestion:\n{query}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        )

        response_text = response.choices[0].message.content

        save_answer(query, response_text, image_base64)

        return response_text, image_base64


    # ----------------------------
    # 5️⃣ TEXT-ONLY CALL
    # ----------------------------
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ]
    )

    response_text = response.choices[0].message.content

    save_answer(query, response_text, None)

    return response_text, None


