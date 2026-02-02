from transformers import CLIPProcessor, CLIPModel
import torch
from llama_index.embeddings.openai import OpenAIEmbedding

text_embed_model = OpenAIEmbedding()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def embed(chunk):
    if chunk.modality == "text":
        return text_embed_model.get_text_embedding(chunk.content)

    elif chunk.modality == "image":
        inputs = clip_processor(images=chunk.content, return_tensors="pt")
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
        return outputs[0].tolist()
