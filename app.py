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
import shutil
from langdetect import detect, LangDetectException
import html

nltk.download('punkt_tab')

# --------------- Configuration ---------------
DOCUMENTS_DIR = "data/documents"
STORAGE_DIR = "storage"
FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "faiss_index.faiss")
CORPUS_PATH = os.path.join(STORAGE_DIR, "corpus.pkl")
LOCATION_DATA_PATH = os.path.join(STORAGE_DIR, "locations.pkl")
ADMIN_PASSWORD = "1234"

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

# --------------- Initialization & Caching ---------------
@st.cache_resource
def load_models_and_groq():
    """Load sentence transformer model and initialize Groq client."""
    try:
        embed_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key:
            st.warning("⚠️ Groq API key not found. Please add it to your Streamlit secrets.", icon="🔒")
            return embed_model, None
        groq_client = Groq(api_key=api_key)
        return embed_model, groq_client
    except Exception as e:
        st.error(f"❌ Error loading models or initializing Groq: {e}")
        return None, None

embed_model, client = load_models_and_groq()

@st.cache_resource
def load_nltk_data():
    """Download and verify NLTK 'punkt' data for sentence tokenization."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
            nltk.data.find('tokenizers/punkt')
        except Exception as e:
            st.error(f"❌ Failed to download NLTK 'punkt' data: {e}. Sentence tokenization may be suboptimal.")
            return False
    return True

nltk_loaded = load_nltk_data()

# --------------- Data Processing Functions ---------------
def process_uploaded_files(uploaded_files):
    file_data, locations_from_csv = [], {}
    for uploaded_file in uploaded_files:
        try:
            file_name = uploaded_file.name
            file_path = os.path.join(DOCUMENTS_DIR, file_name)
            with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
            text = ""
            if file_path.lower().endswith('.pdf'):
                reader = PdfReader(file_path)
                text = "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())
            elif file_path.lower().endswith('.txt'):
                text = uploaded_file.read().decode("utf-8")
            if text: file_data.append({'text': text, 'source': file_name})
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
                df.columns = [col.lower().strip() for col in df.columns]
                name_col = next((c for c in ['name', 'location', 'place'] if c in df.columns), None)
                lat_col = next((c for c in ['lat', 'latitude', 'y'] if c in df.columns), None)
                lon_col = next((c for c in ['lon', 'longitude', 'x'] if c in df.columns), None)
                desc_col = next((c for c in ['description', 'desc', 'details'] if c in df.columns), 'name')
                if name_col and lat_col and lon_col:
                    for _, row in df.iterrows():
                        try:
                            name, lat, lon = str(row[name_col]).strip().lower(), float(row[lat_col]), float(row[lon_col])
                            desc = str(row.get(desc_col, f"Location: {name}"))
                            if name and -90 <= lat <= 90 and -180 <= lon <= 180:
                                locations_from_csv[name] = {'name': name, 'lat': lat, 'lon': lon, 'desc': desc}
                        except (ValueError, TypeError): continue
        except Exception as e: st.error(f"❌ Error processing {uploaded_file.name}: {e}")
    return file_data, locations_from_csv

def extract_sentences(text_data):
    all_sentences = []
    for data in text_data:
        text, source = data['text'], data['source']
        if not text: continue
        sentences = nltk.sent_tokenize(text) if nltk_loaded else re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) > 25: all_sentences.append({'sentence': s_clean, 'source': source})
    return all_sentences

def extract_locations_from_text(text):
    patterns = [r'([\w\s]{3,50}?)\s*-\s*Lat:\s*([-+]?\d{1,3}\.?\d+),?\s*Lon:\s*([-+]?\d{1,3}\.?\d+)', r'([\w\s]{3,50}?)\s+Latitude:\s*([-+]?\d{1,3}\.?\d+),?\s*Longitude:\s*([-+]?\d{1,3}\.?\d+)', r'([\w\s]{3,50}?)\s*\(\s*([-+]?\d{1,3}\.?\d+),\s*([-+]?\d{1,3}\.?\d+)\s*\)']
    locations = {}
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                name, lat, lon = match.groups()
                name = name.strip().lower()
                if name not in locations: locations[name] = {'name': name, 'lat': float(lat), 'lon': float(lon), 'desc': f"Found in document at {lat}, {lon}."}
            except (ValueError, IndexError): continue
    return locations

def build_and_save_data(corpus, locations):
    saved_sentences, saved_locations = 0, 0
    try:
        if corpus and embed_model:
            embeddings = embed_model.encode([item['sentence'] for item in corpus], show_progress_bar=True)
            index = faiss.IndexFlatL2(embeddings.shape[1]); index.add(np.array(embeddings, dtype="float32"))
            faiss.write_index(index, FAISS_INDEX_PATH)
            with open(CORPUS_PATH, "wb") as f: pickle.dump(corpus, f)
            saved_sentences = len(corpus)
        else:
            if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
            if os.path.exists(CORPUS_PATH): os.remove(CORPUS_PATH)
        if locations:
            with open(LOCATION_DATA_PATH, "wb") as f: pickle.dump(locations, f)
            saved_locations = len(locations)
        else:
            if os.path.exists(LOCATION_DATA_PATH): os.remove(LOCATION_DATA_PATH)
        return True, saved_sentences, saved_locations
    except Exception as e:
        st.error(f"❌ Error building/saving data: {e}")
        return False, 0, 0

def load_system_data():
    index, corpus, location_map = None, [], {}
    try:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CORPUS_PATH):
            index = faiss.read_index(FAISS_INDEX_PATH)
            with open(CORPUS_PATH, "rb") as f: corpus = pickle.load(f)
        if os.path.exists(LOCATION_DATA_PATH):
            with open(LOCATION_DATA_PATH, "rb") as f: location_map = pickle.load(f)
    except Exception as e:
        st.error(f"⚠️ Error loading system data: {e}"); return None, [], {}
    return index, corpus, location_map

# --------------- RAG & Chat Functions ---------------
def retrieve_chunks(query, corpus, index, top_k=5):
    if not all([query, corpus, index, embed_model]): return []
    try:
        query_embedding = embed_model.encode([query])
        _, I = index.search(np.array(query_embedding, dtype="float32"), top_k)
        return [corpus[i] for i in I[0] if i < len(corpus)]
    except Exception as e:
        st.warning(f"⚠️ Retrieval error: {e}"); return []

def match_locations(query, location_map):
    if not location_map: return []
    query_lower, found = query.lower(), []
    for name, loc in location_map.items():
        if name in query_lower: found.append(loc)
    if not found:
        query_words = re.findall(r'\b\w+\b', query_lower)
        for word in query_words:
            matches = get_close_matches(word, list(location_map.keys()), n=1, cutoff=0.8)
            if matches: found.append(location_map[matches[0]])
    return list({loc['name']: loc for loc in found}.values())

def compute_distance_info(locations):
    if len(locations) == 2:
        try:
            coord1, coord2 = (locations[0]["lat"], locations[0]["lon"]), (locations[1]["lat"], locations[1]["lon"])
            dist = geodesic(coord1, coord2)
            unit, val = ("km", f"{dist.kilometers:.2f}") if dist.kilometers >= 1 else ("meters", f"{dist.meters:.0f}")
            return f"The distance between {locations[0]['name'].title()} and {locations[1]['name'].title()} is approximately {val} {unit}."
        except: return ""
    return ""

def ask_chatbot(query, context_chunks, geo_context, distance_info):
    if not client: return "The AI assistant is currently offline."
    try:
        lang_code = detect(query)
        lang_map = {'en': 'English', 'ur': 'Urdu', 'hi': 'Hindi'}
        language = lang_map.get(lang_code, 'English')
    except LangDetectException: language = "English"

    context = "\n".join([chunk['sentence'] for chunk in context_chunks])
    system_prompt = "You are CampusGPT, a helpful campus assistant. Answer concisely and conversationally using the provided information. When you mention a specific location, include its coordinates like '(Lat: 34.05, Lon: -118.24)'. Use the context from documents to answer the user's question."
    prompt = f"""{system_prompt}
    ---
    CONTEXT FROM DOCUMENTS: {context if context else 'No relevant information was found in the documents.'}
    ---
    IDENTIFIED LOCATIONS: {geo_context if geo_context else 'No specific locations were mentioned or identified.'}
    ---
    CALCULATED DISTANCE: {distance_info if distance_info else 'Not applicable.'}
    ---
    USER'S QUESTION: {query}
    ---
    YOUR CONVERSATIONAL ANSWER (IN {language.upper()}):"""
    try:
        response = client.chat.completions.create(model="llama3-8b-8192", messages=[{"role": "user", "content": prompt}], temperature=0.6, max_tokens=1024)
        return response.choices[0].message.content
    except Exception as e: return f"I apologize, but I encountered an error: {e}"

# --------------- UI Components ---------------
def create_map(locations):
    if not locations: return None
    try:
        map_center = [np.mean([loc['lat'] for loc in locations]), np.mean([loc['lon'] for loc in locations])]
        m = folium.Map(location=map_center, zoom_start=16, tiles='CartoDB positron', attr='CampusGPT Map')
        for loc in locations:
            maps_url = f"https://www.google.com/maps/search/?api=1&query={loc['lat']},{loc['lon']}"
            popup_html = f"""<div style="width: 220px; font-family: 'Inter', sans-serif;">
                <h4 style="margin-bottom: 10px; color: #1e293b;">{loc['name'].title()}</h4>
                <p style="margin-bottom: 12px; color: #475569;">{loc.get('desc', '')[:100]}...</p>
                <a href="{maps_url}" target="_blank" style="background-color: #4f46e5; color: white; padding: 8px 12px; text-decoration: none; border-radius: 5px; display: inline-block; font-size: 14px; font-weight: 600;">Navigate</a>
            </div>"""
            folium.Marker([loc['lat'], loc['lon']], 
                         popup=folium.Popup(popup_html, max_width=270), 
                         tooltip=loc['name'].title(), 
                         icon=folium.Icon(color="darkblue", icon="location-arrow", prefix="fa")).add_to(m)
        return m
    except Exception as e:
        st.error(f"🗺️ Map creation failed: {e}"); return None

def display_welcome_message():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%); 
                border-radius: 12px; 
                padding: 2rem; 
                margin: 2rem 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        <h2 style="color: #4f46e5; margin-bottom: 1rem;">👋 Welcome to CampusGPT!</h2>
        <p style="color: #4b5563; font-size: 1.1rem; margin-bottom: 1.5rem;">
            Your intelligent campus assistant is ready to help, but first we need some information to get started.
        </p>
        <div style="background-color: white; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;">
            <h4 style="color: #4f46e5; margin-bottom: 1rem;">📌 Admin Instructions:</h4>
            <ol style="color: #4b5563; padding-left: 1.5rem;">
                <li style="margin-bottom: 0.5rem;">Select <b style="color: #4f46e5;">Admin</b> in the sidebar and enter the password.</li>
                <li style="margin-bottom: 0.5rem;">Upload PDF, TXT, or CSV files containing campus information.</li>
                <li>Click <b style="color: #4f46e5;">'Process & Build Index'</b> to make the data searchable.</li>
            </ol>
        </div>
        <p style="color: #6b7280; font-size: 0.9rem;">
            Once the documents are processed, you can ask questions about campus locations, distances, and general information.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --------------- Main Streamlit App ---------------
st.set_page_config(
    page_title="CampusGPT", 
    page_icon="🏫", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for elegant UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Base styles */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container adjustments */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 1.5rem 1rem;
    }
    
    [data-testid="stSidebar"] .stRadio > div {
        flex-direction: row;
        gap: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: white !important;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] .stButton button {
        background-color: white;
        color: #4f46e5;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s ease;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #f0f0f0;
        transform: translateY(-1px);
    }
    
    /* Chat message styling */
    .chat-message {
        display: flex;
        margin-bottom: 1.5rem;
        max-width: 85%;
    }
    
    .user-message {
        justify-content: flex-end;
        margin-left: auto;
    }
    
    .assistant-message {
        justify-content: flex-start;
    }
    
    .chat-bubble {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        line-height: 1.6;
        word-wrap: break-word;
        max-width: 90%;
    }
    
    .user-message .chat-bubble {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .assistant-message .chat-bubble {
        background-color: #f9fafb;
        color: #111827;
        border-bottom-left-radius: 4px;
        border: 1px solid #e5e7eb;
    }
    
    .assistant-message .chat-bubble a {
        color: #4f46e5;
        font-weight: 500;
        text-decoration: none;
    }
    
    .assistant-message .chat-bubble a:hover {
        text-decoration: underline;
    }
    
    .chat-icon {
        font-size: 1.5rem;
        margin-right: 0.75rem;
        margin-top: 0.25rem;
        align-self: flex-start;
    }
    
    /* Input box styling */
    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        padding: 1rem !important;
        border: 1px solid #e5e7eb !important;
    }
    
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        border: none !important;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #6b7280;
        font-size: 1.1rem;
    }
    
    /* Admin panel styling */
    .admin-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 4px solid #4f46e5;
    }
    
    /* Status indicators */
    .status-ready {
        color: #10b981;
        font-weight: 600;
    }
    
    .status-not-ready {
        color: #ef4444;
        font-weight: 600;
    }
    
    /* Danger zone styling */
    .danger-zone {
        border: 1px solid #fecaca;
        background-color: #fef2f2;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    
    /* Button styling */
    .stButton button {
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
    }
    
    /* Tab styling */
    [data-testid="stTabs"] {
        margin-top: 1rem;
    }
    
    [role="tablist"] button {
        padding: 0.5rem 1rem !important;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if 'confirm_delete' not in st.session_state:
    st.session_state.confirm_delete = False

# Load system data
index, corpus, location_map = load_system_data()
system_ready = (index is not None and corpus) or bool(location_map)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: white; font-size: 1.8rem; margin-bottom: 0.5rem;">🏫 CampusGPT</h1>
        <p style="color: #e0e7ff; font-size: 0.9rem;">Your Smart Campus Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Role selection
    role = st.radio(
        "Select Your Role",
        ["👤 User", "🔧 Admin"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", key="clear_chat"):
        st.session_state.chat_history = []
        st.toast("Chat history cleared!", icon="🔄")
        st.rerun()
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div class="footer">
        Made with ❤️ by Zubair Yamin Suhaib
    </div>
    """, unsafe_allow_html=True)

# Main content
st.markdown("""
<div class="main-header">
    <h1>CampusGPT</h1>
    <p>Your Intelligent Campus Assistant</p>
</div>
""", unsafe_allow_html=True)

if not client and role == "👤 User":
    st.error("""
    <div style="background-color: #fef2f2; color: #b91c1c; padding: 1rem; border-radius: 8px; border-left: 4px solid #dc2626;">
        <p style="margin: 0; font-weight: 500;">🔴 The AI Assistant is not configured. An administrator must set the GROQ_API_KEY in the Streamlit secrets.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if role == "🔧 Admin":
    if not st.session_state.authenticated:
        st.subheader("🔐 Admin Login")
        password = st.text_input("Enter Password", type="password", key="admin_pass")
        if st.button("🔑 Login", type="primary"):
            if password == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Incorrect password.")
    else:
        st.subheader("⚙️ Admin Control Panel")
        tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "ℹ️ System Info", "📋 CSV Guide"])
        
        with tab1:
            st.markdown("""
            <div class="admin-card">
                <h3 style="color: #4f46e5; margin-bottom: 1rem;">Upload Campus Documents</h3>
                <p style="color: #6b7280; margin-bottom: 1.5rem;">
                    Upload PDF or TXT files for general campus information, and CSV files for location data.
                    The system will extract text and location information to build a searchable knowledge base.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader(
                "Choose files",
                type=['pdf', 'txt', 'csv'],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            
            if st.button("🔄 Process & Build Index", type="primary"):
                if uploaded_files:
                    with st.spinner("Processing files... This may take a few moments depending on file size."):
                        file_data, csv_locs = process_uploaded_files(uploaded_files)
                        full_text = " ".join([d['text'] for d in file_data])
                        text_locs = extract_locations_from_text(full_text)
                        all_locations = {**csv_locs, **text_locs}
                        corpus_sentences = extract_sentences(file_data)
                        success, num_sentences, num_locations = build_and_save_data(corpus_sentences, all_locations)
                        
                        if success:
                            if num_sentences > 0 or num_locations > 0:
                                st.success(f"""
                                <div style="background-color: #ecfdf5; color: #065f46; padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981;">
                                    <p style="margin: 0; font-weight: 500;">✅ Processing complete!</p>
                                    <p style="margin: 0.5rem 0 0 0;">Saved {num_sentences} sentences and {num_locations} locations.</p>
                                </div>
                                """, unsafe_allow_html=True)
                                st.balloons()
                            else:
                                st.warning("""
                                <div style="background-color: #fffbeb; color: #92400e; padding: 1rem; border-radius: 8px; border-left: 4px solid #f59e0b;">
                                    <p style="margin: 0; font-weight: 500;">⚠️ No processable data found in the files.</p>
                                    <p style="margin: 0.5rem 0 0 0;">Please ensure your files contain text or properly formatted location data.</p>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.warning("""
                    <div style="background-color: #fffbeb; color: #92400e; padding: 1rem; border-radius: 8px; border-left: 4px solid #f59e0b;">
                        <p style="margin: 0; font-weight: 500;">Please upload at least one file.</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <div class="admin-card">
                <h3 style="color: #4f46e5; margin-bottom: 1rem;">System Status</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem;">
                    <div style="background-color: #f9fafb; border-radius: 8px; padding: 1rem; border: 1px solid #e5e7eb;">
                        <p style="color: #6b7280; margin: 0 0 0.5rem 0; font-size: 0.9rem;">System Status</p>
                        <p style="color: #4f46e5; margin: 0; font-size: 1.2rem; font-weight: 600;">{'<span class="status-ready">✅ Ready</span>' if system_ready else '<span class="status-not-ready">❌ Not Ready</span>'}</p>
                    </div>
                    <div style="background-color: #f9fafb; border-radius: 8px; padding: 1rem; border: 1px solid #e5e7eb;">
                        <p style="color: #6b7280; margin: 0 0 0.5rem 0; font-size: 0.9rem;">Indexed Sentences</p>
                        <p style="color: #4f46e5; margin: 0; font-size: 1.2rem; font-weight: 600;">{len(corpus) if corpus else 0}</p>
                    </div>
                    <div style="background-color: #f9fafb; border-radius: 8px; padding: 1rem; border: 1px solid #e5e7eb;">
                        <p style="color: #6b7280; margin: 0 0 0.5rem 0; font-size: 0.9rem;">Known Locations</p>
                        <p style="color: #4f46e5; margin: 0; font-size: 1.2rem; font-weight: 600;">{len(location_map) if location_map else 0}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📍 View Available Locations", expanded=False):
                if location_map:
                    st.markdown("""
                    <div style="background-color: #f9fafb; border-radius: 8px; padding: 1rem; border: 1px solid #e5e7eb; max-height: 300px; overflow-y: auto;">
                        <ul style="margin: 0; padding-left: 1.25rem;">
                    """, unsafe_allow_html=True)
                    for name in sorted(location_map.keys()):
                        st.markdown(f"<li style='margin-bottom: 0.25rem;'>{name.title()}</li>", unsafe_allow_html=True)
                    st.markdown("</ul></div>", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color: #f9fafb; border-radius: 8px; padding: 1rem; border: 1px solid #e5e7eb; text-align: center;">
                        <p style="color: #6b7280; margin: 0;">No locations loaded.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="danger-zone">
                <h3 style="color: #dc2626; margin-bottom: 1rem;">🚨 Danger Zone</h3>
                <p style="color: #6b7280; margin-bottom: 1rem;">
                    These actions are irreversible. Proceed with caution.
                </p>
                {button_html}
            </div>
            """.format(
                button_html="""
                <button onclick="document.getElementById('confirm-delete').style.display='block'" style="background-color: #ef4444; color: white; border: none; border-radius: 8px; padding: 0.5rem 1rem; font-weight: 600; cursor: pointer;">
                    🗑️ Clear All Data & Index
                </button>
                """ if not st.session_state.confirm_delete else """
                <div style="background-color: #fee2e2; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                    <p style="color: #b91c1c; font-weight: 600; margin-bottom: 1rem;">Are you sure? This will delete all processed data.</p>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                        <button onclick="document.getElementById('confirm-delete').style.display='none'" style="background-color: #ef4444; color: white; border: none; border-radius: 8px; padding: 0.5rem; font-weight: 600; cursor: pointer; width: 100%;">
                            Yes, delete everything
                        </button>
                        <button onclick="document.getElementById('confirm-delete').style.display='none'" style="background-color: #e5e7eb; color: #111827; border: none; border-radius: 8px; padding: 0.5rem; font-weight: 600; cursor: pointer; width: 100%;">
                            Cancel
                        </button>
                    </div>
                </div>
                """
            ), unsafe_allow_html=True)
            
            if st.session_state.confirm_delete:
                if st.button("Yes, I am sure, delete everything.", type="primary"):
                    try:
                        if os.path.exists(STORAGE_DIR):
                            shutil.rmtree(STORAGE_DIR)
                        if os.path.exists(DOCUMENTS_DIR):
                            shutil.rmtree(DOCUMENTS_DIR)
                        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
                        os.makedirs(STORAGE_DIR, exist_ok=True)
                        st.session_state.confirm_delete = False
                        st.success("""
                        <div style="background-color: #ecfdf5; color: #065f46; padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981;">
                            <p style="margin: 0; font-weight: 500;">All system data has been cleared.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.rerun()
                    except Exception as e:
                        st.error(f"""
                        <div style="background-color: #fef2f2; color: #b91c1c; padding: 1rem; border-radius: 8px; border-left: 4px solid #dc2626;">
                            <p style="margin: 0; font-weight: 500;">Failed to clear data: {e}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                if st.button("Cancel"):
                    st.session_state.confirm_delete = False
                    st.rerun()
        
        with tab3:
            st.markdown("""
            <div class="admin-card">
                <h3 style="color: #4f46e5; margin-bottom: 1rem;">CSV File Format Guide</h3>
                <p style="color: #6b7280; margin-bottom: 1.5rem;">
                    Your CSV file should contain columns for location names, latitude, and longitude. 
                    A description column is optional but recommended for better context.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            sample_df = pd.DataFrame({
                'name': ['Central Library', 'Student Center'], 
                'latitude': [34.0522, 34.0518], 
                'longitude': [-118.2437, -118.2434], 
                'description': ['Main campus library with study spaces.', 'Hub for student activities and dining.']
            })
            
            st.dataframe(
                sample_df.style
                    .set_properties(**{'background-color': '#f9fafb', 'color': '#111827', 'border': '1px solid #e5e7eb'})
                    .highlight_max(color='#dbeafe')
                    .highlight_min(color='#fee2e2'),
                use_container_width=True
            )
            
            st.download_button(
                "⬇️ Download CSV Template",
                sample_df.to_csv(index=False).encode('utf-8'),
                "locations_template.csv",
                "text/csv",
                help="Download a template CSV file to get started with location data"
            )

else:  # User View
    if not system_ready:
        display_welcome_message()
    else:
        # Display chat history
        for i, msg in enumerate(st.session_state.chat_history):
            is_user = msg["role"] == "user"
            
            st.markdown(f"""
            <div class="chat-message {'user-message' if is_user else 'assistant-message'}">
                <div class="chat-icon">{'👤' if is_user else '🏫'}</div>
                <div class="chat-bubble">
                    {msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display map if locations are found
            if not is_user and "locations" in msg and msg["locations"]:
                map_obj = create_map(msg["locations"])
                if map_obj:
                    st_folium(
                        map_obj, 
                        width=800, 
                        height=400, 
                        key=f"map_{i}",
                        returned_objects=[]
                    )

        # Chat input
        if prompt := st.chat_input("Ask about campus locations, distances, or general info..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.rerun()

        # Generate response to the last user message
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            with st.spinner("🤔 Thinking..."):
                last_prompt = st.session_state.chat_history[-1]["content"]
                chunks = retrieve_chunks(last_prompt, corpus, index)
                locs = match_locations(last_prompt, location_map)
                loc_info = "\n".join([f"{l['name'].title()}: (Lat: {l['lat']}, Lon: {l['lon']})" for l in locs])
                dist_info = compute_distance_info(locs)
                response = ask_chatbot(last_prompt, chunks, loc_info, dist_info)
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response, 
                    "locations": locs
                })
                st.rerun()
