import chromadb
import uuid

# In-memory Chroma client (NO persistence folder)
client = chromadb.Client()

text_collection = client.get_or_create_collection("text_vectors")
image_collection = client.get_or_create_collection("image_vectors")


def add_vector(vector, metadata, image_data=None):

    if metadata["modality"] == "text":

        text_collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[vector],
            metadatas=[metadata]
        )

    elif metadata["modality"] == "image":

        if image_data is None:
            raise ValueError("Image data missing for image modality")

        image_collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[vector],
            metadatas=[{
                **metadata,
                "image_base64": image_data
            }]
        )

    else:
        raise ValueError("Unknown modality")


def query_text(vector, k=5):
    return text_collection.query(
        query_embeddings=[vector],
        n_results=k
    )


def query_image(vector, k=3):
    return image_collection.query(
        query_embeddings=[vector],
        n_results=k
    )


def clear_db():
    global text_collection, image_collection

    client.delete_collection("text_vectors")
    client.delete_collection("image_vectors")

    text_collection = client.get_or_create_collection("text_vectors")
    image_collection = client.get_or_create_collection("image_vectors")
