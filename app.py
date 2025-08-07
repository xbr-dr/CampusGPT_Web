
# app.py

import streamlit as st
import os
import pickle
import faiss
import nltk
import re
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from geopy.distance import geodesic
from difflib import get_close_matches
import folium
from streamlit_folium import st_folium
from groq import Groq
from langdetect import detect
import shutil

# Directories
DOCUMENTS_DIR = "data/documents"
STORAGE_DIR = "storage"
FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "faiss_index.faiss")
CORPUS_PATH = os.path.join(STORAGE_DIR, "corpus.pkl")
LOCATION_DATA_PATH = os.path.join(STORAGE_DIR, "locations.pkl")

# Admin Password
ADMIN_PASSWORD = "1234"

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

@st.cache_resource
def load_models_and_groq():
    embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    multi_embed_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
    groq_client = Groq(api_key="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")
    return embed_model, multi_embed_model, groq_client

embed_model, multi_embed_model, client = load_models_and_groq()

@st.cache_resource
def load_nltk_data():
    nltk.download("punkt", quiet=True)
    return True

nltk_loaded = load_nltk_data()

def detect_language(text):
    try:
        lang = detect(text)
        return {"en": "English", "ur": "Urdu", "hi": "Hindi"}.get(lang, "English")
    except:
        return "English"

def process_uploaded_files(uploaded_files):
    file_data = []
    locations_from_text = {}

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_path = os.path.join(DOCUMENTS_DIR, file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        text = ""
        if file_path.lower().endswith('.pdf'):
            reader = PdfReader(file_path)
            text = "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())
        elif file_path.lower().endswith('.txt'):
            text = uploaded_file.read().decode("utf-8")

        if text:
            file_data.append({'text': text, 'source': file_name})
            locations_from_text.update(extract_locations_from_text(text))

    return file_data, locations_from_text

def extract_locations_from_text(text):
    patterns = [
        r'([\w\s]{3,50}?)\s*-\s*Lat:\s*([-+]?\d{1,3}\.\d+),?\s*Lon:\s*([-+]?\d{1,3}\.\d+)',
        r'([\w\s]{3,50}?)\s+Latitude:\s*([-+]?\d{1,3}\.\d+),?\s*Longitude:\s*([-+]?\d{1,3}\.\d+)',
        r'([\w\s]{3,50}?)\s*\(\s*([-+]?\d{1,3}\.\d+),\s*([-+]?\d{1,3}\.\d+)\s*\)',
    ]
    locations = {}
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                name, lat, lon = match.groups()
                name = name.strip().lower()
                if name not in locations:
                    locations[name] = {'name': name, 'lat': float(lat), 'lon': float(lon)}
            except:
                continue
    return locations

def extract_sentences(text_data):
    all_sentences = []
    for data in text_data:
        text = data['text']
        if nltk_loaded:
            sentences = nltk.sent_tokenize(text)
        else:
            sentences = text.split(".")
        for s in sentences:
            if len(s.strip()) > 25:
                all_sentences.append({'sentence': s.strip(), 'source': data['source']})
    return all_sentences

def build_and_save_index(corpus, locations):
    sentences_to_embed = [item['sentence'] for item in corpus]
    embeddings = embed_model.encode(sentences_to_embed, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype="float32"))
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CORPUS_PATH, "wb") as f:
        pickle.dump(corpus, f)
    with open(LOCATION_DATA_PATH, "wb") as f:
        pickle.dump(locations, f)

def load_system_data():
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CORPUS_PATH, "rb") as f: corpus = pickle.load(f)
        with open(LOCATION_DATA_PATH, "rb") as f: locations = pickle.load(f)
        return index, corpus, locations
    return None, None, {}

def retrieve_chunks(query, corpus, index, language):
    model = multi_embed_model if language != "English" else embed_model
    query_embedding = model.encode([query])
    _, I = index.search(np.array(query_embedding, dtype="float32"), 5)
    return [corpus[i] for i in I[0] if i < len(corpus)]

def ask_chatbot(query, context_chunks, locations, language):
    context = "\n".join([chunk['sentence'] for chunk in context_chunks])
    system_prompt = {
        "English": "You are CampusGPT, a smart assistant. Answer concisely.",
        "Urdu": "آپ کیمپسGPT ہیں۔ مختصر اور مفید جواب دیں۔",
        "Hindi": "आप CampusGPT हैं। संक्षेप में और सहायक उत्तर दें।"
    }.get(language, "You are CampusGPT.")

    prompt = f"""{system_prompt}
CONTEXT: {context}
LOCATIONS: {locations}
QUESTION: {query}
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=1024
    )
    return response.choices[0].message.content

# Streamlit app
st.set_page_config("CampusGPT", "🏫")
st.title("🏫 CampusGPT")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

role = st.sidebar.radio("Select Role", ["User", "Admin"], horizontal=True)

index, corpus, location_map = load_system_data()
system_ready = index is not None

if role == "Admin":
    if not st.session_state.authenticated:
        password = st.text_input("Enter Admin Password", type="password")
        if st.button("Login"):
            if password == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    else:
        st.subheader("Upload Campus Documents")
        uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
        if st.button("Process & Build Index"):
            file_data, locs = process_uploaded_files(uploaded_files)
            corpus_sentences = extract_sentences(file_data)
            if corpus_sentences:
                build_and_save_index(corpus_sentences, locs)
                st.success("Index built successfully.")
else:
    if not system_ready:
        st.warning("System is not ready. Please contact Admin.")
    else:
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**CampusGPT:** {chat['bot']}")
        if user_input := st.chat_input("Ask me anything..."):
            lang = detect_language(user_input)
            results = retrieve_chunks(user_input, corpus, index, lang)
            loc_str = ", ".join([f"{v['name']}({v['lat']},{v['lon']})" for v in location_map.values()])
            response = ask_chatbot(user_input, results, loc_str, lang)
            st.session_state.chat_history.append({"user": user_input, "bot": response})
            st.rerun()
