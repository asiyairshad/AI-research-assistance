import torch
from transformers import CLIPProcessor, CLIPModel
from llama_index.embeddings.openai import OpenAIEmbedding

# ----------------------------
# OpenAI Text Embedding Model
# ----------------------------
text_embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)

# ----------------------------
# CLIP Model (for image space)
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)


# --------------------------------
# TEXT EMBEDDING (OpenAI)
# --------------------------------
def embed_text(text: str):
    return text_embed_model.get_text_embedding(text)


# --------------------------------
# IMAGE EMBEDDING (CLIP)
# --------------------------------
def embed_image(image):
    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)

    features = features[0]
    features = features / features.norm(p=2)

    return features.cpu().tolist()


# --------------------------------
# QUERY EMBEDDING (DUAL)
# --------------------------------
def embed_query_for_text(query: str):
    """
    Used for retrieving text chunks.
    """
    return embed_text(query)


def embed_query_for_image(query: str):
    """
    Used for retrieving relevant images.
    Query embedded in CLIP text space.
    """
    inputs = clip_processor(
        text=query,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)

    features = features[0]
    features = features / features.norm(p=2)

    return features.cpu().tolist()

def embed(chunk):
    if chunk.modality == "text":
        return embed_text(chunk.content)
    elif chunk.modality == "image":
        return embed_image(chunk.content)
    else:
        raise ValueError("Unknown modality")
