from openai import OpenAI
from embeddings import embed
from vectore_store import query_text
import base64
import io
from langsmith import traceable

from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

SYSTEM_PROMPT = """
You are a research assistant.
You must answer using ONLY the provided context.
If the answer is not in the context, say "I don't know".
and if the asked question is also related to text as well as image then answer with context of image as well
and answer in proper manner and structure and also explain about image , and if in any question
there is no need of image then dont show image, if not needed 
Do not use any external knowledge.
Be concise and factual.

and if there is no need of image then please dont fetch image only text
"""


def pil_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

@traceable
def answer(query, image_chunks):
    # 1. Text retrieval
    q_vec = embed(type("obj",(object,),{"content":query,"modality":"text"}))
    text_results = query_text(q_vec)

    context = ""
    if text_results["metadatas"]:
        for r in text_results["metadatas"][0]:
            context += f"(Page {r['page']}): {r['content']}\n"

    # 2. Decide if image is needed
    decision = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
Question: {query}

Context:
{context}

Is a visual figure needed to answer this?
Reply only YES or NO.
"""
            }
        ]
    )

    use_image = "YES" in decision.choices[0].message.content.upper()

    # 3. If yes → use vision
    if use_image and len(image_chunks) > 0:
        image = image_chunks[0].content
        image_b64 = pil_to_base64(image)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context + "\n" + query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content, True

    # 4. Otherwise → normal text answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": context + "\n" + query
            }
        ]
    )
    return response.choices[0].message.content,False
