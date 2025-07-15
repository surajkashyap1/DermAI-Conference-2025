# app/retriever/pdf_loader.py
"""
Streamlit front‑end for the DermAI RAG chatbot + image classifier.

❗  NOTE
    ▸  All heavy work now lives in `run_streamlit_app()`.
    ▸  Nothing runs on mere *import*, so unit‑tests that import this
       module (via `file_ingestor` or others) will no longer explode.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

import openai
import requests
import streamlit as st
import tiktoken
from gtts import gTTS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import speech_recognition as sr

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "OpenAI_Key")
FAISS_INDEX_PATH = Path("faiss_index")  # folder created by FAISS.save_local()

# ────────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────────
def extract_text_from_pdfs(pdf_paths: List[str | Path]) -> str:
    pages: List[str] = []
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        pages.extend(
            page.extract_text() or "" for page in reader.pages  # keep order
        )
    return "\n".join(filter(None, pages))


def split_text_into_chunks(text: str, max_tokens: int = 2_000) -> List[str]:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)

    chunks, current = [], []
    for token in tokens:
        current.append(token)
        if len(current) >= max_tokens:
            chunks.append(tokenizer.decode(current))
            current = []
    if current:
        chunks.append(tokenizer.decode(current))
    return chunks


def create_langchain_model(full_text: str):
    """Load or build the FAISS index, then return a ConversationalRetrievalChain."""
    if FAISS_INDEX_PATH.exists():
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        chunks = split_text_into_chunks(full_text)
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY
        )
        vectorstore = FAISS.from_texts(chunks, embeddings)
        vectorstore.save_local(str(FAISS_INDEX_PATH))

    retriever = vectorstore.as_retriever()
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY),
        retriever=retriever,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────────────────────────────────────
def run_streamlit_app() -> None:
    """Entry point **only** when the file is run with `streamlit run`."""
    # ----------------------------------------------------------------------------
    # Build / load index
    # ----------------------------------------------------------------------------
    pdf_files = ["Dermatology_An_Illustrated_Colour_Textbo.pdf"]
    text_content = extract_text_from_pdfs(pdf_files)
    qa_chain = create_langchain_model(text_content)

    # ----------------------------------------------------------------------------
    # Streamlit layout
    # ----------------------------------------------------------------------------
    st.set_page_config(page_title="Skin Disease Expert System", page_icon="🧴")
    st.title("🧑‍⚕️ Skin Disease Expert System")

    # Initialise chat history (per‑session)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.sidebar.title("Chat History")
    for i, (u, b) in enumerate(st.session_state.chat_history, start=1):
        st.sidebar.markdown(f"*You {i}:* {u}")
        st.sidebar.markdown(f"*Chatbot {i}:* {b}")

    # -------------------------------- Voice input ------------------------------
    def speech_to_text() -> str | None:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening…")
            try:
                audio = recognizer.listen(source, timeout=5)
                return recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                st.error("Couldn’t understand speech.")
            except sr.RequestError as e:
                st.error(f"Speech‑API error: {e}")
        return None

    # -------------------------------- Chat box --------------------------------
    user_query = st.text_input(
        "Ask about any skin disease:", key="user_query", placeholder="Type here…"
    )
    if st.button("🎤 Use voice"):
        voice = speech_to_text()
        if voice:
            user_query = voice
            st.success(f"You said: {voice}")

    if user_query:
        try:
            resp = qa_chain.invoke(
                {"question": user_query, "chat_history": st.session_state.chat_history}
            )
            answer = resp["answer"]
            st.session_state.chat_history.append((user_query, answer))

            # TTS
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                gTTS(answer).save(tmp.name)
                st.audio(tmp.name, format="audio/mp3")

            # Display entire thread
            for i, (u, b) in enumerate(st.session_state.chat_history, start=1):
                st.markdown(f"*You {i}:* {u}")
                st.markdown(f"*Chatbot {i}:* {b}")

        except Exception as e:
            st.error(f"LLM error: {e}")

    # ------------------------------ Image upload ------------------------------
    uploaded = st.file_uploader(
        "Upload a skin‑condition image:", type=["png", "jpg", "jpeg"]
    )
    if uploaded:
        try:
            resp = requests.post(
                "http://127.0.0.1:5000/predict",
                files={"pic": (uploaded.name, uploaded.getvalue(), "image/jpeg")},
                timeout=30,
            )
            if resp.ok:
                data = resp.json()
                st.image(uploaded, caption="Uploaded Image", use_column_width=True)
                st.markdown(f"**Diagnosis:** {data['class_name']}")
                st.markdown(f"Confidence: {data['class_probability']:.2f}%")
                st.markdown(f"**Description:** {data['description']}")
                st.markdown(
                    "*Disclaimer:* This is a demo; always consult a dermatologist."
                )
            else:
                st.error("Classifier API returned an error.")
        except Exception as e:
            st.error(f"Image‑API call failed: {e}")


# ────────────────────────────────────────────────────────────────────────────────
# Run only when executed directly (i.e. `streamlit run pdf_loader.py`)
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_streamlit_app()
