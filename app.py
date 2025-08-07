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
from io import StringIO
import shutil
from langdetect import detect, LangDetectException

nltk.download('punkt_tab')

# --------------- Configuration ---------------
# Directories for storing data and uploaded documents
DOCUMENTS_DIR = "data/documents"
STORAGE_DIR = "storage"
FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "faiss_index.faiss")
CORPUS_PATH = os.path.join(STORAGE_DIR, "corpus.pkl")
LOCATION_DATA_PATH = os.path.join(STORAGE_DIR, "locations.pkl")

# Simple password for admin access
ADMIN_PASSWORD = "1234"

# Create necessary directories if they don't exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)


# --------------- Initialization & Caching ---------------
@st.cache_resource
def load_models_and_groq():
    """Load sentence transformer model and initialize Groq client."""
    try:
        # Use a multilingual model for broad language support
        embed_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

        # Load API key from Streamlit secrets
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
    """Download NLTK data for sentence tokenization."""
    try:
        nltk.download("punkt", quiet=True)
        return True
    except Exception:
        return False


nltk_loaded = load_nltk_data()


# --------------- Data Processing Functions ---------------
def process_uploaded_files(uploaded_files):
    """Process uploaded files (PDF, TXT, CSV) to extract text and location data."""
    file_data = []
    locations_from_csv = {}

    for uploaded_file in uploaded_files:
        try:
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
                            name = str(row[name_col]).strip().lower()
                            lat, lon = float(row[lat_col]), float(row[lon_col])
                            desc = str(row.get(desc_col, f"Location: {name}"))
                            if name and -90 <= lat <= 90 and -180 <= lon <= 180:
                                locations_from_csv[name] = {'name': name, 'lat': lat, 'lon': lon, 'desc': desc}
                        except (ValueError, TypeError):
                            continue # Ignore rows with invalid geo data
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")

    return file_data, locations_from_csv


def extract_sentences(text_data):
    """Extract clean sentences from a list of text data objects."""
    all_sentences = []
    for data in text_data:
        text = data['text']
        source = data['source']
        if not text: continue

        # Use NLTK if available, otherwise fallback to a simpler regex split
        if nltk_loaded:
            sentences = nltk.sent_tokenize(text)
        else:
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

        for s in sentences:
            s_clean = s.strip()
            # Filter out very short or non-sensical snippets
            if len(s_clean) > 25:
                all_sentences.append({'sentence': s_clean, 'source': source})
    return all_sentences


def extract_locations_from_text(text):
    """Find locations mentioned in text using regex patterns."""
    patterns = [
        r'([\w\s]{3,50}?)\s*-\s*Lat:\s*([-+]?\d{1,3}\.?\d+),?\s*Lon:\s*([-+]?\d{1,3}\.?\d+)',
        r'([\w\s]{3,50}?)\s+Latitude:\s*([-+]?\d{1,3}\.?\d+),?\s*Longitude:\s*([-+]?\d{1,3}\.?\d+)',
        r'([\w\s]{3,50}?)\s*\(\s*([-+]?\d{1,3}\.?\d+),\s*([-+]?\d{1,3}\.?\d+)\s*\)',
    ]
    locations = {}
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                name, lat, lon = match.groups()
                name = name.strip().lower()
                if name not in locations:
                    locations[name] = {'name': name, 'lat': float(lat), 'lon': float(lon),
                                       'desc': f"Found in document at coordinates {lat}, {lon}."}
            except (ValueError, IndexError):
                continue
    return locations


def build_and_save_index(corpus, locations):
    """Create FAISS index and save all processed data."""
    if not corpus or not embed_model:
        st.error("❌ Cannot build index: No sentences or embedding model available.")
        return False
    try:
        sentences_to_embed = [item['sentence'] for item in corpus]
        embeddings = embed_model.encode(sentences_to_embed, show_progress_bar=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings, dtype="float32"))
        faiss.write_index(index, FAISS_INDEX_PATH)

        with open(CORPUS_PATH, "wb") as f:
            pickle.dump(corpus, f)
        with open(LOCATION_DATA_PATH, "wb") as f:
            pickle.dump(locations, f)

        return True
    except Exception as e:
        st.error(f"❌ Error building index: {e}")
        return False


def load_system_data():
    """Load FAISS index, corpus, and location data from disk."""
    try:
        if os.path.exists(FAISS_INDEX_PATH):
            index = faiss.read_index(FAISS_INDEX_PATH)
            with open(CORPUS_PATH, "rb") as f: corpus = pickle.load(f)
            with open(LOCATION_DATA_PATH, "rb") as f: location_map = pickle.load(f)
            return index, corpus, location_map
    except Exception as e:
        st.error(f"⚠️ Error loading system data: {e}")
    return None, None, {}


# --------------- RAG & Chat Functions ---------------
def retrieve_chunks(query, corpus, index, top_k=5):
    """Retrieve relevant text chunks and their sources using FAISS."""
    if not all([query, corpus, index, embed_model]): return []
    try:
        query_embedding = embed_model.encode([query])
        _, I = index.search(np.array(query_embedding, dtype="float32"), top_k)
        return [corpus[i] for i in I[0] if i < len(corpus)]
    except Exception as e:
        st.warning(f"⚠️ Retrieval error: {e}")
        return []


def match_locations(query, location_map):
    """Find specific locations mentioned in the query."""
    if not location_map: return []
    query_lower = query.lower()
    found = []

    # Exact match first
    for name, loc in location_map.items():
        if name in query_lower:
            found.append(loc)

    # Fuzzy match if no exact matches found
    if not found:
        query_words = query_lower.split()
        location_names = list(location_map.keys())
        for word in query_words:
            matches = get_close_matches(word, location_names, n=1, cutoff=0.8)
            if matches:
                found.append(location_map[matches[0]])

    # Return unique locations
    return list({loc['name']: loc for loc in found}.values())


def compute_distance_info(locations):
    """Generate a human-readable distance string for 2 locations."""
    if len(locations) == 2:
        try:
            coord1, coord2 = (locations[0]["lat"], locations[0]["lon"]), (locations[1]["lat"], locations[1]["lon"])
            dist = geodesic(coord1, coord2)
            unit, val = ("km", f"{dist.kilometers:.2f}") if dist.kilometers >= 1 else ("meters", f"{dist.meters:.0f}")
            return f"The distance between {locations[0]['name'].title()} and {locations[1]['name'].title()} is approximately {val} {unit}."
        except:
            return ""
    return ""


def ask_chatbot(query, context_chunks, geo_context, distance_info):
    """Generate a response using the Groq API, with automatic language detection."""
    if not client:
        return "The AI assistant is currently offline. Please ensure the GROQ_API_KEY is configured by an administrator."

    # Auto-detect language of the query
    try:
        lang_code = detect(query)
        lang_map = {'en': 'English', 'ur': 'Urdu', 'hi': 'Hindi'}
        language = lang_map.get(lang_code, 'English') # Default to English if lang not in map
    except LangDetectException:
        language = "English" # Default to English for very short/unclear queries

    context = "\n".join([chunk['sentence'] for chunk in context_chunks])
    sources_text = "\n".join(list(set([f"- {chunk['source']}" for chunk in context_chunks])))

    system_prompt = "You are CampusGPT, a helpful campus assistant. Answer concisely and conversationally using the provided information. When you mention a specific location, include its coordinates in parentheses, like 'The Library (Lat: 34.05, Lon: -118.24)'. Do not list the sources separately at the end; just use the information naturally in your response."

    prompt = f"""{system_prompt}
    ---
    CONTEXT: {context if context else 'No general information found.'}
    ---
    SOURCES (for your reference only, do not list them):
    {sources_text if sources_text else 'No specific sources found.'}
    ---
    LOCATIONS: {geo_context if geo_context else 'No specific locations identified.'}
    ---
    DISTANCE: {distance_info if distance_info else 'Not applicable.'}
    ---
    QUESTION: {query}
    ---
    ANSWER IN {language.upper()}:"""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192", messages=[{"role": "user", "content": prompt}],
            temperature=0.6, max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"I apologize, but I encountered an error: {e}"


# --------------- UI Components ---------------
def create_map(locations):
    """Create a Folium map with markers for found locations."""
    if not locations: return None
    try:
        map_center = [np.mean([loc['lat'] for loc in locations]), np.mean([loc['lon'] for loc in locations])]
        m = folium.Map(location=map_center, zoom_start=16, tiles='CartoDB positron', attr='CampusGPT Map')
        for loc in locations:
            # Correct Google Maps URL
            Maps_url = f"https://www.google.com/maps?q={loc['lat']},{loc['lon']}"

            popup_html = f"""
            <div style="width: 220px; font-family: 'Inter', sans-serif;">
                <h4 style="margin-bottom: 10px; color: #1e293b;">{loc['name'].title()}</h4>
                <p style="margin-bottom: 12px; color: #475569;">{loc.get('desc', '')[:100]}...</p>
                <a href="{Maps_url}" target="_blank" style="background-color: #5850ec; color: white; padding: 8px 12px; text-decoration: none; border-radius: 5px; display: inline-block; font-size: 14px; font-weight: 600;">Navigate on Google Maps</a>
            </div>
            """

            folium.Marker(
                [loc['lat'], loc['lon']],
                popup=folium.Popup(popup_html, max_width=270),
                tooltip=loc['name'].title(),
                icon=folium.Icon(color="darkblue", icon="location-arrow", prefix="fa")
            ).add_to(m)
        return m
    except Exception as e:
        st.error(f"🗺️ Map creation failed: {e}")
        return None


def display_welcome_message():
    """Show a welcome or system-not-ready message."""
    st.markdown("""
        <div class="welcome-card">
            <h2>👋 Welcome to CampusGPT!</h2>
            <p>I'm your smart campus assistant. Before you can ask me questions, an administrator needs to upload some documents.</p>
            <h4>Admin Instructions:</h4>
            <ol>
                <li>Select the <strong>🔧 Admin</strong> role in the sidebar.</li>
                <li>Enter the password.</li>
                <li>Upload documents (PDF, TXT, or CSV), then click <strong>'Process & Build Index'</strong>.</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)


# --------------- Main Streamlit App ---------------
st.set_page_config(page_title="CampusGPT", page_icon="🏫", layout="wide")

# Custom CSS for a modern look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

    .block-container { max-width: 80rem; padding-left: 2rem; padding-right: 2rem; padding-top: 1rem; }
    .main-header { text-align: center; margin-bottom: 2rem; }
    .main-header h1 { font-size: 3.5rem; font-weight: 700; background: -webkit-linear-gradient(45deg, #5850ec, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .main-header p { color: #6b7280; font-size: 1.1rem; }

    .st-emotion-cache-1y4p8pa { padding-top: 4rem; }

    .chat-message { display: flex; align-items: flex-start; max-width: 85%; margin-bottom: 1.5rem; }
    .chat-bubble { padding: 1rem 1.25rem; border-radius: 1.25rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05); line-height: 1.6; }

    .user-message { justify-content: flex-end; margin-left: auto; }
    .user-message .chat-bubble { background-color: #5850ec; color: white; border-bottom-right-radius: 0.25rem; }
    .user-message .chat-icon { margin-left: 0.75rem; }

    .assistant-message { justify-content: flex-start; }
    .assistant-message .chat-bubble { background-color: #f1f5f9; color: #1e293b; border-bottom-left-radius: 0.25rem; }
    .assistant-message .chat-bubble a { color: #5850ec; font-weight: 600; }
    .assistant-message .chat-icon { margin-right: 0.75rem; }

    .chat-icon { font-size: 1.5rem; color: #94a3b8; align-self: flex-start; margin-top: 0.25rem;}

    .welcome-card { background-color: #f8fafc; border-left: 5px solid #5850ec; padding: 2rem; border-radius: 0.5rem; margin-top: 2rem; }

    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 44px; background-color: #f1f5f9; border-radius: 8px; }
    .stTabs [aria-selected="true"] { background-color: #5850ec; color: white; }

    .stButton>button { border-radius: 8px; }

    .sidebar-footer { text-align: center; position: absolute; bottom: 20px; width: 80%; color: #888; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if 'confirm_delete' not in st.session_state: st.session_state.confirm_delete = False


# Load data once
index, corpus, location_map = load_system_data()
system_ready = all([index is not None, corpus is not None])

# --- Sidebar ---
with st.sidebar:
    st.title("🏫 CampusGPT")
    st.markdown("---")

    role = st.radio("Select Your Role", ["👤 User", "🔧 Admin"], horizontal=True)

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.toast("Chat history cleared!", icon="🔄")
        st.rerun()

    st.markdown("---")
    st.markdown("<div class='sidebar-footer'>Made with ❤️ by Zubair Yamin Suhaib</div>", unsafe_allow_html=True)

# --- Main Page Content ---
st.markdown("<div class='main-header'><h1>CampusGPT</h1><p>Your Smart Campus Assistant</p></div>",
            unsafe_allow_html=True)

# Stop app if Groq client isn't initialized (e.g., missing API key)
if not client and role == "👤 User":
    st.error("🔴 The AI Assistant is not configured. An administrator must set the GROQ_API_KEY in the Streamlit secrets.")
    st.stop()


if role == "🔧 Admin":
    if not st.session_state.authenticated:
        st.subheader("🔐 Admin Login")
        password = st.text_input("Enter Password", type="password", key="admin_pass")
        if st.button("🔑 Login"):
            if password == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Incorrect password.")
    else:  # Authenticated Admin View
        st.subheader("⚙️ Admin Control Panel")
        tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "ℹ️ System Info & Management", "📋 CSV Guide"])

        with tab1:
            st.info("Upload documents (PDF, TXT) for general info and CSVs for location data.", icon="💡")
            uploaded_files = st.file_uploader("Upload Campus Documents", type=['pdf', 'txt', 'csv'],
                                              accept_multiple_files=True)

            if st.button("🔄 Process & Build Index", type="primary"):
                if uploaded_files:
                    with st.spinner("Processing files, this may take a moment..."):
                        file_data, csv_locs = process_uploaded_files(uploaded_files)

                        full_text_for_loc_extraction = " ".join([d['text'] for d in file_data])

                        text_locs = extract_locations_from_text(full_text_for_loc_extraction)
                        all_locations = {**csv_locs, **text_locs}

                        corpus_sentences = extract_sentences(file_data)

                        if corpus_sentences:
                            if build_and_save_index(corpus_sentences, all_locations):
                                st.success(
                                    f"✅ Index built successfully with {len(corpus_sentences)} sentences and {len(all_locations)} locations.")
                                st.balloons()
                            else:
                                st.error("❌ Failed to build index.")
                        else:
                            st.warning("⚠️ No text could be extracted to build an index.")
                else:
                    st.warning("Please upload at least one file.", icon="❗")

        with tab2:
            st.subheader("📊 System Status")
            st.metric("System Status", "✅ Ready" if system_ready else "❌ Not Ready")
            col1, col2 = st.columns(2)
            col1.metric("Indexed Sentences", len(corpus) if corpus else 0)
            col2.metric("Known Locations", len(location_map) if location_map else 0)

            with st.expander("📍 View Available Locations"):
                if location_map:
                    for name in sorted(location_map.keys()): st.write(f"• {name.title()}")
                else:
                    st.write("No locations loaded.")

            st.markdown("---")
            st.subheader("🚨 Danger Zone")
            if st.button("🗑️ Clear All Data & Index", type="secondary"):
                st.session_state.confirm_delete = True

            if st.session_state.confirm_delete:
                st.warning(
                    "**Are you sure?** This will delete all uploaded documents and the search index. This action cannot be undone.")
                col_del_1, col_del_2 = st.columns(2)
                if col_del_1.button("Yes, I am sure, delete everything.", type="primary"):
                    try:
                        if os.path.exists(STORAGE_DIR): shutil.rmtree(STORAGE_DIR)
                        if os.path.exists(DOCUMENTS_DIR): shutil.rmtree(DOCUMENTS_DIR)
                        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
                        os.makedirs(STORAGE_DIR, exist_ok=True)
                        st.session_state.confirm_delete = False
                        st.success("All system data has been cleared.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to clear data: {e}")
                if col_del_2.button("Cancel"):
                     st.session_state.confirm_delete = False
                     st.rerun()

        with tab3:
            st.markdown("""
            Your CSV file should contain columns for the location's name, latitude, and longitude. A description column is optional but recommended.
            Column names can be flexible (e.g., 'name' or 'location'; 'lat' or 'latitude').
            **Example:**
            """)
            sample_df = pd.DataFrame({
                'name': ['Central Library', 'Student Center'], 'latitude': [34.0522, 34.0518],
                'longitude': [-118.2437, -118.2434], 'description': ['Main campus library with study rooms.', 'A hub for student activities and dining.']
            })
            st.dataframe(sample_df, use_container_width=True)
            st.download_button("⬇️ Download CSV Template", sample_df.to_csv(index=False).encode('utf-8'),
                               "campus_locations_template.csv", "text/csv")

else:  # User View
    if not system_ready:
        display_welcome_message()
    else:
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message"><div class="chat-bubble">{message["content"]}</div><div class="chat-icon">👤</div></div>', unsafe_allow_html=True)
            else:  # Assistant
                st.markdown(f'<div class="chat-message assistant-message"><div class="chat-icon">🏫</div><div class="chat-bubble">{message["content"]}</div></div>', unsafe_allow_html=True)
                if "locations" in message and message["locations"]:
                    map_obj = create_map(message["locations"])
                    if map_obj:
                        st_folium(map_obj, width=700, height=400, key=f"map_{i}")

        # Chat input
        if prompt := st.chat_input("Ask about campus locations, distances, or general info..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.rerun()

        # Generate response if last message was from user
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            with st.spinner("🤔 Thinking..."):
                last_prompt = st.session_state.chat_history[-1]["content"]

                retrieved_chunks = retrieve_chunks(last_prompt, corpus, index)

                locations = match_locations(last_prompt, location_map)
                location_info = "\n".join([f"{loc['name'].title()}: (Lat: {loc['lat']}, Lon: {loc['lon']})" for loc in locations])
                distance_info = compute_distance_info(locations)

                response = ask_chatbot(last_prompt, retrieved_chunks, location_info, distance_info)

                st.session_state.chat_history.append({
                    "role": "assistant", "content": response, "locations": locations
                })
                st.rerun()
