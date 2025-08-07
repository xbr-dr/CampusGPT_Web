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
from langdetect import detect
import folium
from streamlit_folium import st_folium
from groq import Groq
import shutil
import io

# Paths
DOCUMENTS_DIR = "data/documents"
STORAGE_DIR = "storage"
INDEX_PATH = os.path.join(STORAGE_DIR, "faiss_index.faiss")
CORPUS_PATH = os.path.join(STORAGE_DIR, "corpus.pkl")
LOCATIONS_PATH = os.path.join(STORAGE_DIR, "locations.pkl")

# Admin password (for demo; ideally use env var or secrets)
ADMIN_PASSWORD = "1234"

# Ensure directories exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

# Load models and Groq
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    multi_embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    groq_client = Groq(api_key=st.secrets[GROQ_API_KEY])
    return embed_model, multi_embed_model, groq_client

embed_model, multi_embed_model, client = load_models()

# Download NLTK punkt
@st.cache_resource
def setup_nltk():
    nltk.download("punkt", quiet=True)
    return True

setup_nltk()

# Language detection helper
def detect_language(q):
    try:
        code = detect(q)
        return {"en": "English", "ur": "Urdu", "hi": "Hindi"}.get(code, "English")
    except:
        return "English"

# File processing
def extract_text(file_path):
    ext = file_path.lower()
    if ext.endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext.endswith(".txt"):
        return open(file_path, "r", encoding="utf-8").read()
    elif ext.endswith((".csv", ".xlsx", ".xls")):
        df = pd.read_excel(file_path) if ext.endswith((".xlsx", ".xls")) else pd.read_csv(file_path)
        return df.astype(str).to_csv(index=False)
    return ""

def extract_locations(text):
    patterns = [
        r"([\w\s]{3,50})\s*-\s*Lat\s*[:=]\s*([\d\.\-]+)\s*[,;]?\s*Lon\s*[:=]\s*([\d\.\-]+)",
        r"([\w\s]{3,50})\s*\(\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*\)"
    ]
    locs = {}
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            nm, lat, lon = m.groups()
            nm = nm.strip().lower()
            try:
                locs[nm] = {"name": nm, "lat": float(lat), "lon": float(lon)}
            except:
                pass
    return locs

def chunk_text(text, max_len=500):
    sents = nltk.sent_tokenize(text)
    chunks, tmp = [], ""
    for s in sents:
        if len(tmp) + len(s) <= max_len:
            tmp += " " + s
        else:
            chunks.append(tmp.strip()); tmp = s
    if tmp: chunks.append(tmp.strip())
    return chunks

def build_index_and_save(docs, locations):
    corpus = []
    for doc in docs:
        txt = extract_text(doc)
        for chunk in chunk_text(txt):
            corpus.append({"sentence": chunk, "source": os.path.basename(doc)})
    embeddings = embed_model.encode([c["sentence"] for c in corpus], show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype="float32"))
    faiss.write_index(index, INDEX_PATH)
    with open(CORPUS_PATH, "wb") as f: pickle.dump(corpus, f)
    with open(LOCATIONS_PATH, "wb") as f: pickle.dump(locations, f)

def load_index_and_data():
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        corpus = pickle.load(open(CORPUS_PATH, "rb"))
        locations = pickle.load(open(LOCATIONS_PATH, "rb"))
        return index, corpus, locations
    return None, None, {}

def retrieve_chunks(q, corpus, index, lang):
    model = multi_embed_model if lang != "English" else embed_model
    emb = model.encode([q])
    _, I = index.search(np.array(emb, dtype="float32"), 5)
    return [corpus[i] for i in I[0] if i < len(corpus)]

def ask_groq(q, context, locations_text, lang):
    prompt = f"You are a helpful assistant. Answer in {lang}.\n\nCONTEX: {context}\n\nLOCATIONS: {locations_text}\n\nQ: {q}"
    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=512,
        top_p=1
    )
    return resp.choices[0].message.content.strip()

def show_map(locs):
    if not locs: return
    center = [np.mean([l["lat"] for l in locs]), np.mean([l["lon"] for l in locs])]
    m = folium.Map(location=center, zoom_start=17)
    for l in locs:
        folium.Marker([l["lat"], l["lon"]], popup=l["name"].title()).add_to(m)
    st_folium(m, height=400)

# Streamlit app
st.set_page_config(page_title="CampusGPT", layout="wide")
st.title("CampusGPT — Your Smart Campus Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

role = st.sidebar.radio("Role:", ["User", "Admin"])

index, corpus, locations = load_index_and_data()
ready = index is not None

if role == "Admin":
    if not st.session_state.authenticated:
        pwd = st.text_input("Admin Password:", type="password")
        if st.button("Login"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.success("Logged in!")
            else:
                st.error("Wrong password.")
    else:
        st.header("Upload Documents")
        up = st.file_uploader("PDF/TXT/CSV/XLSX/XLS", accept_multiple_files=True, type=["pdf","txt","csv","xlsx","xls"])
        if st.button("Process & Build"):
            docs = []
            all_locs = {}
            for f in up:
                path = os.path.join(DOCUMENTS_DIR, f.name)
                with open(path, "wb") as fp: fp.write(f.read())
                docs.append(path)
                all_locs.update(extract_locations(extract_text(path)))
            if docs:
                build_index_and_save(docs, all_locs)
                index, corpus, locations = load_index_and_data()
                st.success(f"Indexed {len(corpus)} chunks and {len(locations)} locations.")
else:
    if not ready:
        st.info("System not ready. Admin needs to build index.")
    else:
        for turn in st.session_state.chat_history:
            st.markdown(f"**You:** {turn['user']}")
            st.markdown(f"**CampusGPT:** {turn['bot']}")
        if q := st.chat_input("Ask anything..."):
            lang = detect_language(q)
            ctx = retrieve_chunks(q, corpus, index, lang)
            locs = [v for k,v in locations.items() if k in q.lower()]
            loc_text = ", ".join([f"{l['name']}({l['lat']},{l['lon']})" for l in locs])
            resp = ask_groq(q, "\n".join(c["sentence"] for c in ctx), loc_text, lang)
            st.session_state.chat_history.append({"user": q, "bot": resp})
            st.experimental_rerun()
