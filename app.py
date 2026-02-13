from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import io
import base64

from ingest import parse_pdf_bytes, extract_images_bytes
from embeddings import embed_text, embed_image
from vectore_store import add_vector, clear_db
from rag import answer

st.set_page_config(page_title="Multimodal Research Assistant")
st.title("Multimodal Research Assistant")

# ----------------------------
# File Upload
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload research PDFs",
    accept_multiple_files=True
)

if uploaded_files:
    clear_db()

    for uploaded in uploaded_files:
        pdf_bytes = uploaded.read()

        text_chunks = parse_pdf_bytes(pdf_bytes)
        image_chunks = extract_images_bytes(pdf_bytes)

        # ----------------------------
        # Index Text
        # ----------------------------
        for chunk in text_chunks:
            vec = embed_text(chunk.content)

            add_vector(
                vec,
                {
                    "content": chunk.content,
                    "page": chunk.page,
                    "modality": "text",
                    "source": uploaded.name
                }
            )

        # ----------------------------
        # Index Images
        # ----------------------------
        for img_chunk in image_chunks:
            img_vec = embed_image(img_chunk.content)

            buffer = io.BytesIO()
            img_chunk.content.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()

            add_vector(
                img_vec,
                {
                    "page": img_chunk.page,
                    "modality": "image",
                    "source": uploaded.name
                },
                img_b64
            )

    st.success("All documents indexed")

# ----------------------------
# Mode Selector
# ----------------------------
mode = st.radio(
    "Choose response mode",
    ["fast", "quality"],
    horizontal=True
)

# ----------------------------
# Ask Question
# ----------------------------
query = st.text_input("Ask a question")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        response, image_base64 = answer(query, mode=mode)

    st.subheader("Answer")
    st.write(response)

    if image_base64:
        st.subheader("Relevant Image")
        st.image(base64.b64decode(image_base64))
