import chromadb
import uuid

client = chromadb.Client()

text_collection = client.get_or_create_collection("text_vectors")
image_collection = client.get_or_create_collection("image_vectors")

def add_vector(vector, metadata):
    if metadata["modality"] == "text":
        text_collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[vector],
            metadatas=[metadata]
        )
    else:
        image_collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[vector],
            metadatas=[metadata]
        )

def query_text(vector):
    return text_collection.query(
        query_embeddings=[vector],
        n_results=5
    )

def clear_db():
    global text_collection, image_collection

    try:
        client.delete_collection("text_vectors")
    except:
        pass

    try:
        client.delete_collection("image_vectors")
    except:
        pass

    text_collection = client.get_or_create_collection("text_vectors")
    image_collection = client.get_or_create_collection("image_vectors")
