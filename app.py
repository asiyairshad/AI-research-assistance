from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from ingest import parse_pdf_bytes, extract_images_bytes
from embeddings import embed
from vectore_store import add_vector, clear_db
from rag import answer

st.title("Multimodal Research Assistant")

all_image_chunks = []

uploaded_files = st.file_uploader(
    "Upload research PDFs",
    accept_multiple_files=True
)

if uploaded_files:
    clear_db()  # start fresh session
    all_image_chunks = []

    for uploaded in uploaded_files:
        pdf_bytes = uploaded.read()

        text_chunks = parse_pdf_bytes(pdf_bytes)
        image_chunks = extract_images_bytes(pdf_bytes)
        all_image_chunks.extend(image_chunks)

        for chunk in text_chunks:
            vec = embed(chunk)
            add_vector(vec, {
                "content": chunk.content,
                "page": chunk.page,
                "modality": "text",
                "source": uploaded.name
            })

    st.success("All documents indexed")
#mode selector
mode = st.radio(
    "Choose response mode",
    ["fast", "quality"],
    horizontal=True
)


query = st.text_input("Ask a question")

if query:
    response, used_image = answer(query, all_image_chunks, mode = mode)
    st.write(response)

    if used_image:
        st.subheader("Relevant figures")
        for chunk in all_image_chunks:
            st.image(chunk.content, caption=f"Page {chunk.page}")



