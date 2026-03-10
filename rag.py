import os
from openai import OpenAI
from dotenv import load_dotenv
from langsmith import traceable

from embeddings import embed_query_for_text, embed_query_for_image
from vectore_store import query_text, query_image
from cache_store import (
    get_cached_embedding,
    save_embedding,
    get_cached_answer,
    save_answer
)

from schemas import RAGResponse, AnswerSchema


load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "multimodal-rag"


class MultimodalRAG:

    def __init__(self):

        self.client = OpenAI()

        self.system_prompt = """
You are a research assistant.

Answer ONLY using the provided context.

If an image is provided, analyze it and explain clearly.

If the answer is not in the context say "I don't know".
"""


    def embed_text(self, query):

        vec = get_cached_embedding(query, "text")

        if vec is None:
            vec = embed_query_for_text(query)
            save_embedding(query, vec, "text")

        return vec


    def embed_image(self, query):

        vec = get_cached_embedding(query, "image")

        if vec is None:
            vec = embed_query_for_image(query)
            save_embedding(query, vec, "image")

        return vec


    def retrieve_text(self, query):

        vec = self.embed_text(query)
        results = query_text(vec)
        context = ""
        pages = []

        if results and results.get("metadatas"):

            for r in results["metadatas"][0]:
                page = r.get("page", "N/A")
                content = r.get("content", "")
                context += f"(Page {page}): {content}\n"

                if page != "N/A":
                    pages.append(page)

        return context, pages

    def retrieve_image(self, query):

        vec = self.embed_image(query)
        results = query_image(vec)
        image_base64 = None

        if results and results.get("metadatas"):

            for r in results["metadatas"][0]:

                if "image_base64" in r:
                    image_base64 = r["image_base64"]
                    break

        return image_base64


    def generate_answer(self, query, context, image, mode):

        model_name = "gpt-4o-mini" if mode == "fast" else "gpt-4o"

        if image:

            response = self.client.chat.completions.parse(

                model=model_name,

                messages=[

                    {"role": "system", "content": self.system_prompt},

                    {
                        "role": "user",

                        "content": [

                            {
                                "type": "text",
                                "text": f"Context:\n{context}\n\nQuestion:{query}"
                            },

                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image}"
                                }
                            }

                        ]
                    }

                ],

                response_format=AnswerSchema
            )

        else:

            response = self.client.chat.completions.parse(

                model=model_name,

                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion:{query}"}
                ],

                response_format=AnswerSchema
            )

        return response.choices[0].message.parsed.answer


    @traceable
    def answer(self, query, mode="fast"):

        cached = get_cached_answer(query)

        if cached:
            return RAGResponse(**cached)

        context, pages = self.retrieve_text(query)

        if not context:
            return RAGResponse(answer="I don't know")

        image = self.retrieve_image(query)

        answer = self.generate_answer(query, context, image, mode)

        save_answer(query, answer, image)

        return RAGResponse(
            answer=answer,
            source_pages=pages,
            image_base64=image
        )