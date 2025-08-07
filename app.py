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
import folium
from streamlit_folium import st_folium
from groq import Groq
import shutil
from langdetect import detect, LangDetectException
from thefuzz import process, fuzz

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
    """Download and verify NLTK 'punkt' data."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
            nltk.data.find('tokenizers/punkt')
        except Exception as e:
            st.error(f"❌ Failed to download NLTK 'punkt' data: {e}.")
            return False
    return True

nltk_loaded = load_nltk_data()

# --------------- Data Processing Functions ---------------
def process_uploaded_files(uploaded_files):
    """Process uploaded files, creating a richer location map from CSVs."""
    file_data = []
    # MODIFIED: location_map is now a list of dicts for richer data
    locations_list = []
    
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
                # IMPROVED: Process CSV to handle aliases and more details
                df = pd.read_csv(file_path)
                df.columns = [col.strip().lower() for col in df.columns]
                
                # Standardize potential column names
                name_col = next((c for c in ['name', 'department'] if c in df.columns), None)
                lat_col = next((c for c in ['latitude', 'lat'] if c in df.columns), None)
                lon_col = next((c for c in ['longitude', 'lon'] if c in df.columns), None)
                other_names_col = next((c for c in ['other names', 'aliases'] if c in df.columns), None)

                if name_col and lat_col and lon_col:
                    for _, row in df.iterrows():
                        try:
                            primary_name = str(row[name_col]).strip()
                            lat, lon = float(row[lat_col]), float(row[lon_col])
                            
                            if primary_name and -90 <= lat <= 90 and -180 <= lon <= 180:
                                # Create a list of all possible names for this location
                                all_names = {primary_name.lower()}
                                if other_names_col and pd.notna(row[other_names_col]):
                                    aliases = str(row[other_names_col]).split(',')
                                    all_names.update(alias.strip().lower() for alias in aliases if alias.strip())
                                
                                # Store all row data as a dictionary for context
                                location_details = row.to_dict()
                                location_details['primary_name'] = primary_name
                                location_details['all_names'] = list(all_names)
                                location_details['lat'] = lat
                                location_details['lon'] = lon
                                
                                locations_list.append(location_details)
                        except (ValueError, TypeError): continue
        except Exception as e: st.error(f"❌ Error processing {uploaded_file.name}: {e}")
        
    return file_data, locations_list

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
    index, corpus, location_map = None, [], []
    try:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CORPUS_PATH):
            index = faiss.read_index(FAISS_INDEX_PATH)
            with open(CORPUS_PATH, "rb") as f: corpus = pickle.load(f)
        if os.path.exists(LOCATION_DATA_PATH):
            with open(LOCATION_DATA_PATH, "rb") as f: location_map = pickle.load(f)
    except Exception as e:
        st.error(f"⚠️ Error loading system data: {e}"); return None, [], []
    return index, corpus, location_map

# --------------- RAG & Chat Functions ---------------
def retrieve_chunks(query, corpus, index, top_k=7):
    if not all([query, corpus, index, embed_model]): return []
    try:
        query_embedding = embed_model.encode([query])
        _, I = index.search(np.array(query_embedding, dtype="float32"), top_k)
        return [corpus[i] for i in I[0] if i < len(corpus)]
    except Exception as e:
        st.warning(f"⚠️ Retrieval error: {e}"); return []

def match_locations(query, locations_list, score_cutoff=85):
    """IMPROVED: Uses fuzzy matching to find the best location match from the query."""
    if not locations_list: return []
    
    # Create a flat list of all possible names to search against
    all_possible_names = []
    for i, loc in enumerate(locations_list):
        for name in loc['all_names']:
            all_possible_names.append((name, i)) # Store name and original index
            
    choices = [name for name, _ in all_possible_names]
    
    # Find the best match from the list of all names
    best_match = process.extractOne(query, choices, scorer=fuzz.partial_ratio, score_cutoff=score_cutoff)
    
    if best_match:
        # Find the original location dict using the index we stored
        matched_name, score = best_match
        original_index = next(idx for name, idx in all_possible_names if name == matched_name)
        return [locations_list[original_index]]
        
    return []

def compute_distance_info(locations):
    if len(locations) == 2:
        try:
            coord1 = (locations[0]["lat"], locations[0]["lon"])
            coord2 = (locations[1]["lat"], locations[1]["lon"])
            dist = geodesic(coord1, coord2)
            unit, val = ("km", f"{dist.kilometers:.2f}") if dist.kilometers >= 1 else ("meters", f"{dist.meters:.0f}")
            return f"The distance between {locations[0]['primary_name']} and {locations[1]['primary_name']} is approximately {val} {unit}."
        except: return ""
    return ""

def ask_chatbot(query, context_chunks, matched_locations, distance_info):
    """IMPROVED: Better prompt engineering for more focused answers."""
    if not client: return "The AI assistant is currently offline."
    try:
        lang_code = detect(query)
        language = {'en': 'English', 'ur': 'Urdu', 'hi': 'Hindi'}.get(lang_code, 'English')
    except LangDetectException: language = "English"

    context = "\n".join([chunk['sentence'] for chunk in context_chunks])
    
    # Prepare location context by formatting the dictionary
    geo_context = ""
    if matched_locations:
        loc_info = []
        for loc in matched_locations:
            details = f"Location: {loc['primary_name']}"
            if 'building' in loc and pd.notna(loc['building']): details += f", Building: {loc['building']}"
            if 'floor' in loc and pd.notna(loc['floor']): details += f", Floor: {loc['floor']}"
            details += f" (Lat: {loc['lat']:.5f}, Lon: {loc['lon']:.5f})"
            loc_info.append(details)
        geo_context = "\n".join(loc_info)

    system_prompt = "You are CampusGPT, an expert assistant for a college campus. Your primary goal is to provide accurate and concise information based ONLY on the context provided. Do not use any external knowledge. If the context doesn't contain the answer, clearly state that you don't have enough information. When asked for details about a location, synthesize information from all relevant fields provided."
    prompt = f"""{system_prompt}
    ---
    CONTEXT FROM DOCUMENTS:
    {context if context else 'No relevant information was found in the documents.'}
    ---
    IDENTIFIED LOCATIONS AND THEIR DETAILS:
    {geo_context if geo_context else 'No specific locations were mentioned or identified.'}
    ---
    CALCULATED DISTANCE:
    {distance_info if distance_info else 'Not applicable.'}
    ---
    USER'S QUESTION: {query}
    ---
    YOUR CONVERSATIONAL ANSWER (IN {language.upper()}):"""
    try:
        response = client.chat.completions.create(model="llama3-8b-8192", messages=[{"role": "user", "content": prompt}], temperature=0.5, max_tokens=1024)
        return response.choices[0].message.content
    except Exception as e: return f"I apologize, but I encountered an error: {e}"

# --------------- UI Components ---------------
def create_map(locations):
    if not locations: return None
    try:
        map_center = [np.mean([loc['lat'] for loc in locations]), np.mean([loc['lon'] for loc in locations])]
        m = folium.Map(location=map_center, zoom_start=17, tiles='CartoDB positron', attr='CampusGPT Map')
        for loc in locations:
            google_maps_url = f"https://www.google.com/maps/search/?api=1&query={loc['lat']},{loc['lon']}"
            
            desc = f"Building: {loc.get('building', 'N/A')}, Floor: {loc.get('floor', 'N/A')}"
            
            popup_html = f"""<div style="width: 220px; font-family: 'Inter', sans-serif;">
                <h4 style="margin-bottom: 10px; color: #1e293b;">{loc['primary_name']}</h4>
                <p style="margin-bottom: 12px; color: #475569;">{desc}</p>
                <a href="{google_maps_url}" target="_blank" style="background-color: #5850ec; color: white; padding: 8px 12px; text-decoration: none; border-radius: 5px; display: inline-block; font-size: 14px; font-weight: 600;">Navigate on Google Maps</a>
            </div>"""
            folium.Marker(
                [loc['lat'], loc['lon']],
                popup=folium.Popup(popup_html, max_width=270),
                tooltip="Click for details & navigation",
                icon=folium.Icon(color="darkblue", icon="location-arrow", prefix="fa")
            ).add_to(m)
        return m
    except Exception as e:
        st.error(f"🗺️ Map creation failed: {e}"); return None

def display_welcome_message():
    st.markdown("""<div class="welcome-card">
        <h2>👋 Welcome to CampusGPT!</h2>
        <p>An administrator needs to upload documents before you can ask questions.</p>
        <h4>Admin Instructions:</h4><ol>
        <li>Select <b>Admin</b> in the sidebar and enter the password.</li>
        <li>Upload PDF, TXT, or CSV files and click <b>'Process & Build Index'</b>.</li></ol>
    </div>""", unsafe_allow_html=True)

# --------------- Main Streamlit App ---------------
st.set_page_config(page_title="CampusGPT", page_icon="🏫", layout="wide")
st.markdown("""<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');html,body,[class*="st-"]{font-family:'Inter',sans-serif}.block-container{padding:1rem 2rem 2rem}.main-header h1{font-size:3.5rem;font-weight:700;background:-webkit-linear-gradient(45deg, #5850ec, #a855f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent}.chat-message{display:flex;align-items:flex-start;max-width:85%;margin-bottom:1.5rem}.chat-bubble{padding:1rem 1.25rem;border-radius:1.25rem;box-shadow:0 4px 6px rgba(0,0,0,.05);line-height:1.6;word-wrap:break-word}.user-message{justify-content:flex-end;margin-left:auto}.user-message .chat-bubble{background-color:#5850ec;color:#fff;border-bottom-right-radius:.25rem}.user-message .chat-icon{margin-left:.75rem}.assistant-message{justify-content:flex-start}.assistant-message .chat-bubble{background-color:#f1f5f9;color:#1e293b;border-bottom-left-radius:.25rem}.assistant-message .chat-bubble a{color:#5850ec;font-weight:600}.chat-icon{font-size:1.5rem;color:#94a3b8;align-self:flex-start;margin-top:.25rem}.welcome-card{background-color:#f8fafc;border-left:5px solid #5850ec;padding:2rem;border-radius:.5rem;margin-top:2rem}</style>""", unsafe_allow_html=True)

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if 'confirm_delete' not in st.session_state: st.session_state.confirm_delete = False

index, corpus, location_map = load_system_data()
system_ready = (index is not None and corpus) or bool(location_map)

with st.sidebar:
    st.title("🏫 CampusGPT")
    st.markdown("---")
    role = st.radio("Select Your Role", ["👤 User", "🔧 Admin"], horizontal=True)
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []; st.toast("Chat history cleared!", icon="🔄"); st.rerun()
    st.markdown("---")
    st.markdown("<div style='text-align:center;position:absolute;bottom:20px;width:80%;color:#888;'>Made with ❤️ by Zubair Yamin Suhaib</div>", unsafe_allow_html=True)

st.markdown("<div class='main-header' style='text-align:center'><h1>CampusGPT</h1><p>Your Smart Campus Assistant</p></div>", unsafe_allow_html=True)

if not client and role == "👤 User":
    st.error("🔴 The AI Assistant is not configured. An administrator must set the GROQ_API_KEY in the Streamlit secrets.")
    st.stop()

if role == "🔧 Admin":
    if not st.session_state.authenticated:
        st.subheader("🔐 Admin Login")
        password = st.text_input("Enter Password", type="password", key="admin_pass")
        if st.button("🔑 Login"):
            if password == ADMIN_PASSWORD: st.session_state.authenticated = True; st.rerun()
            else: st.error("❌ Incorrect password.")
    else:
        st.subheader("⚙️ Admin Control Panel")
        tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "ℹ️ System Info", "📋 CSV Guide"])
        with tab1:
            st.info("Upload documents (PDF, TXT) and location data (CSV).", icon="💡")
            uploaded_files = st.file_uploader("Upload Campus Documents", type=['pdf', 'txt', 'csv'], accept_multiple_files=True)
            if st.button("🔄 Process & Build Index", type="primary"):
                if uploaded_files:
                    with st.spinner("Processing files, this may take a moment..."):
                        file_data, locations_list = process_uploaded_files(uploaded_files)
                        corpus_sentences = extract_sentences(file_data)
                        success, num_s, num_l = build_and_save_data(corpus_sentences, locations_list)
                        if success:
                            if num_s > 0 or num_l > 0: st.success(f"✅ Processing complete! Saved {num_s} sentences and {num_l} locations."); st.balloons()
                            else: st.warning("⚠️ No processable data found in the files.")
                else: st.warning("Please upload at least one file.", icon="❗")
        with tab2:
            st.subheader("📊 System Status")
            st.metric("System Status", "✅ Ready" if system_ready else "❌ Not Ready")
            c1, c2 = st.columns(2); c1.metric("Indexed Sentences", len(corpus)); c2.metric("Known Locations", len(location_map))
            with st.expander("📍 View Available Locations"):
                if location_map:
                    for loc in location_map: st.write(f"• {loc['primary_name']}")
                else: st.write("No locations loaded.")
        with tab3:
            st.markdown("Your CSV file should have `name`, `latitude`, `longitude`, and optionally `other names` columns.")
            sample_df = pd.DataFrame({'name': ['Central Library', 'Botany Dept'], 'latitude': [34.07, 34.08], 'longitude': [74.81, 74.82], 'other names': ['Main Library', 'Bio-Sciences']})
            st.dataframe(sample_df, use_container_width=True)
else:  # User View
    if not system_ready: display_welcome_message()
    else:
        for i, msg in enumerate(st.session_state.chat_history):
            is_user = msg["role"] == "user"
            st.markdown(f"""<div class="chat-message {'user-message' if is_user else 'assistant-message'}">
                <div class="chat-icon">{'👤' if is_user else '🏫'}</div>
                <div class="chat-bubble">{msg["content"]}</div>
            </div>""", unsafe_allow_html=True)
            if not is_user and "map_data" in msg and msg["map_data"]:
                map_obj = create_map(msg["map_data"])
                if map_obj: 
                    st_folium(map_obj, width=700, height=450, key=f"map_{i}")
                    # FIXED: Add caption below the map
                    st.caption("Click any marker on the map for details and navigation.")
        if prompt := st.chat_input("Ask about campus locations, distances, or general info..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.rerun()
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            with st.spinner("🤔 Thinking..."):
                last_prompt = st.session_state.chat_history[-1]["content"]
                chunks = retrieve_chunks(last_prompt, corpus, index)
                locs = match_locations(last_prompt, location_map)
                dist_info = compute_distance_info(locs)
                response = ask_chatbot(last_prompt, chunks, locs, dist_info)
                st.session_state.chat_history.append({"role": "assistant", "content": response, "map_data": locs})
                st.rerun()
