import os
import streamlit as st
import torch
import faiss
import pickle
import re
import fitz  # PyMuPDF
import folium
from streamlit_folium import st_folium
from sentence_transformers import SentenceTransformer
from groq import Groq
from langdetect import detect
from PyPDF2 import PdfReader

# ----------------------- CONFIG -----------------------
st.set_page_config(
    page_title="CampusGPT",
    page_icon="🎓",
    layout="wide"
)

st.markdown("<h1 style='text-align:center;'>🎓 CampusGPT</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Your intelligent campus navigation assistant</p>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------- INITIAL SETUP -----------------------
DOCUMENTS_FOLDER = "data/documents"
RESULTS_FOLDER = "results"
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
index_path = os.path.join(RESULTS_FOLDER, "index.pkl")

# ----------------------- LOAD MODELS -----------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embed_model

embed_model = load_models()
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ----------------------- PDF PROCESSING -----------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, max_length=500):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) <= max_length:
            chunk += sentence + " "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + " "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def extract_locations(text):
    patterns = [
        r"([A-Za-z\s]+)[^\d]*(?:Latitude|Lat)[:\s]*([+-]?\d+\.\d+)[,;\s]*(?:Longitude|Lon)[:\s]*([+-]?\d+\.\d+)",
        r"([A-Za-z\s]+)\s*\(\s*([+-]?\d+\.\d+)\s*,\s*([+-]?\d+\.\d+)\s*\)",
    ]
    sentences = re.split(r'[.\n]', text)
    locations = []

    for sentence in sentences:
        if len(sentence.strip()) < 10:
            continue
        for pattern in patterns:
            try:
                for match in re.finditer(pattern, sentence, re.IGNORECASE):
                    name_raw = match.group(1).strip()
                    lat = float(match.group(2).strip())
                    lon = float(match.group(3).strip())
                    name = re.sub(r'\s+', ' ', name_raw).strip().lower()
                    if len(name) >= 3 and not name.isdigit():
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            locations.append({
                                "name": name,
                                "lat": lat,
                                "lon": lon,
                                "desc": sentence.strip()
                            })
            except:
                continue
    unique_locations = {loc['name']: loc for loc in locations}
    return unique_locations

# ----------------------- VECTOR STORE -----------------------
def build_vector_index(docs):
    chunks = []
    for doc in docs:
        text = extract_text_from_pdf(doc)
        chunks += chunk_text(text)
    embeddings = embed_model.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    with open(index_path, "wb") as f:
        pickle.dump((index, chunks), f)
    return chunks

def load_index():
    with open(index_path, "rb") as f:
        return pickle.load(f)

# ----------------------- LLM RESPONSE -----------------------
def query_llm(chat_history):
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=chat_history,
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False
    )
    return response.choices[0].message.content.strip()

# ----------------------- MAP RENDERING -----------------------
def show_map(location_data):
    location = [location_data["lat"], location_data["lon"]]
    m = folium.Map(location=location, zoom_start=17)
    folium.Marker(location, popup=location_data["desc"], tooltip=location_data["name"].title()).add_to(m)
    st_folium(m, height=400)

# ----------------------- UI LAYOUT -----------------------
role = st.sidebar.radio("Login as", ["User", "Admin"])

if role == "Admin":
    st.subheader("📄 Upload Documents")
    uploaded = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if st.button("Rebuild Knowledge Base") and uploaded:
        with st.spinner("Processing..."):
            for file in uploaded:
                with open(os.path.join(DOCUMENTS_FOLDER, file.name), "wb") as f:
                    f.write(file.read())
            all_docs = [os.path.join(DOCUMENTS_FOLDER, f) for f in os.listdir(DOCUMENTS_FOLDER)]
            build_vector_index(all_docs)
        st.success("Knowledge base updated!")

else:
    st.subheader("💬 Ask CampusGPT")
    query = st.text_input("You:", placeholder="e.g., Where is the admin block?")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "system", "content": "You are a helpful assistant for campus queries."}]

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Check if query matches any known location
        all_text = ""
        for f in os.listdir(DOCUMENTS_FOLDER):
            with open(os.path.join(DOCUMENTS_FOLDER, f), "rb") as pdf:
                all_text += extract_text_from_pdf(pdf)
        locations = extract_locations(all_text)
        match = None
        for name in locations:
            if name in query.lower():
                match = locations[name]
                break

        with st.spinner("Generating response..."):
            if os.path.exists(index_path):
                index, chunks = load_index()
                query_emb = embed_model.encode([query])
                D, I = index.search(query_emb, k=3)
                retrieved = "\n".join([chunks[i] for i in I[0]])
                st.session_state.chat_history.append({"role": "user", "content": f"{query}\n\nContext:\n{retrieved}"})

            response = query_llm(st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

        st.markdown(f"**Bot:** {response}")
        
        if match:
            st.success(f"📍 Found location: {match['name'].title()}")
            show_map(match)

    if st.session_state.chat_history:
        with st.expander("🕒 Chat History"):
            for msg in st.session_state.chat_history[1:]:
                st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")

# ----------------------- FOOTER -----------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
        <p>🏫 <strong>CampusGPT</strong> | Made with ❤️ using Streamlit</p>
        <p style='font-size: 0.9rem;'>Your intelligent campus navigation assistant</p>
    </div>
    """,
    unsafe_allow_html=True
)
