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
import openrouteservice
from streamlit_js_eval import get_geolocation

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
def load_clients():
    """Load all models and API clients."""
    try:
        embed_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
        
        groq_api_key = st.secrets.get("GROQ_API_KEY")
        ors_api_key = st.secrets.get("ORS_API_KEY")

        groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        ors_client = openrouteservice.Client(key=ors_api_key) if ors_api_key else None
        
        if not groq_client: st.warning("⚠️ Groq API key not found. AI chat will be disabled.", icon="🔒")
        if not ors_client: st.warning("⚠️ OpenRouteService API key not found. Navigation will be disabled.", icon="🗺️")

        return embed_model, groq_client, ors_client
    except Exception as e:
        st.error(f"❌ Error loading models or clients: {e}")
        return None, None, None

embed_model, groq_client, ors_client = load_clients()

@st.cache_resource
def load_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download("punkt", quiet=True)
    return True

nltk_loaded = load_nltk_data()

# --------------- Data Processing & Categorization ---------------
def categorize_location(name):
    """Categorize a location based on keywords in its name for custom markers."""
    name_lower = name.lower()
    if any(keyword in name_lower for keyword in ['library']): return 'library'
    if any(keyword in name_lower for keyword in ['research', 'lab', 'center']): return 'research'
    if any(keyword in name_lower for keyword in ['dept', 'department', 'biochemistry', 'botany', 'math']): return 'academic'
    if any(keyword in name_lower for keyword in ['admin', 'office', 'block']): return 'admin'
    if any(keyword in name_lower for keyword in ['hostel', 'guesthouse']): return 'hostel'
    return 'default'

def process_uploaded_files(uploaded_files):
    file_data, locations_list = [], []
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
                df.columns = [col.strip().lower() for col in df.columns]
                name_col, lat_col, lon_col, other_names_col = 'name', 'latitude', 'longitude', 'other names'
                
                if all(c in df.columns for c in [name_col, lat_col, lon_col]):
                    for _, row in df.iterrows():
                        try:
                            primary_name = str(row[name_col]).strip()
                            lat, lon = float(row[lat_col]), float(row[lon_col])
                            if primary_name and -90 <= lat <= 90 and -180 <= lon <= 180:
                                all_names = {primary_name.lower()}
                                if other_names_col in df.columns and pd.notna(row[other_names_col]):
                                    aliases = str(row[other_names_col]).split(',')
                                    all_names.update(alias.strip().lower() for alias in aliases if alias.strip())
                                
                                location_details = row.to_dict()
                                location_details['primary_name'] = primary_name
                                location_details['all_names'] = list(all_names)
                                location_details['lat'], location_details['lon'] = lat, lon
                                location_details['category'] = categorize_location(primary_name)
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


# --------------- RAG, Navigation & Chat Functions ---------------
def retrieve_chunks(query, corpus, index, top_k=7):
    if not all([query, corpus, index, embed_model]): return []
    try:
        query_embedding = embed_model.encode([query])
        _, I = index.search(np.array(query_embedding, dtype="float32"), top_k)
        return [corpus[i] for i in I[0] if i < len(corpus)]
    except Exception: return []

def match_locations(query, locations_list, score_cutoff=85):
    if not locations_list: return []
    all_possible_names = []
    for i, loc in enumerate(locations_list):
        for name in loc['all_names']: all_possible_names.append((name, i))
    choices = [name for name, _ in all_possible_names]
    best_match = process.extractOne(query, choices, scorer=fuzz.partial_ratio, score_cutoff=score_cutoff)
    if best_match:
        matched_name, _ = best_match
        original_index = next(idx for name, idx in all_possible_names if name == matched_name)
        return [locations_list[original_index]]
    return []

def get_turn_by_turn_directions(start, end):
    if not ors_client:
        return "Navigation service is not configured. An administrator must add an ORS_API_KEY."
    try:
        route_request = {'coordinates': [start[::-1], end[::-1]], 'format': 'json', 'profile': 'foot-walking', 'instructions': True}
        route = ors_client.directions(**route_request)
        steps = route['routes'][0]['segments'][0]['steps']
        directions_text = "🚶‍♂️ **Here are your walking directions:**\n\n---\n"
        for i, step in enumerate(steps):
            directions_text += f"**{i+1}.** {step['instruction']} (for **{int(step['distance'])} meters**).\n"
        return directions_text
    except Exception as e:
        return f"Could not retrieve directions. The service may be unavailable or out of range. Error: {e}"

def ask_chatbot(query, context_chunks, matched_locations):
    if not groq_client: return "AI assistant is not configured."
    context = "\n".join([chunk['sentence'] for chunk in context_chunks])
    geo_context = ""
    if matched_locations:
        loc = matched_locations[0]
        details = f"Location: {loc['primary_name']}"
        if 'building' in loc and pd.notna(loc['building']): details += f", Building: {loc['building']}"
        if 'floor' in loc and pd.notna(loc['floor']): details += f", Floor: {loc['floor']}"
        geo_context = details

    system_prompt = "You are CampusGPT, an expert assistant for a college campus. Your goal is to provide accurate, concise information based ONLY on the context provided. Do not use any external knowledge. If the context doesn't contain the answer, clearly state that you don't have enough information. Synthesize details from the provided location data when asked."
    prompt = f"""{system_prompt}
    ---
    CONTEXT FROM DOCUMENTS: {context if context else 'No relevant information found.'}
    ---
    IDENTIFIED LOCATION: {geo_context if geo_context else 'No specific location was mentioned.'}
    ---
    USER'S QUESTION: {query}
    ---
    YOUR ANSWER:"""
    try:
        response = groq_client.chat.completions.create(model="llama3-8b-8192", messages=[{"role": "user", "content": prompt}], temperature=0.5, max_tokens=1024)
        return response.choices[0].message.content
    except Exception as e: return f"An error occurred with the AI model: {e}"

# --------------- UI Components ---------------
def create_map(locations):
    """Creates a map with themed markers AND permanent text labels."""
    if not locations: return None
    
    marker_themes = {
        'library': {'color': 'orange', 'icon': 'book'}, 'research': {'color': 'purple', 'icon': 'flask'},
        'academic': {'color': 'blue', 'icon': 'university'}, 'admin': {'color': 'green', 'icon': 'building'},
        'hostel': {'color': 'cadetblue', 'icon': 'home'}, 'default': {'color': 'darkblue', 'icon': 'info-sign'}
    }
    
    try:
        map_center = [np.mean([loc['lat'] for loc in locations]), np.mean([loc['lon'] for loc in locations])]
        m = folium.Map(location=map_center, zoom_start=17, tiles='CartoDB positron')
        for loc in locations:
            theme = marker_themes.get(loc.get('category', 'default'), marker_themes['default'])
            Maps_url = f"https://www.google.com/maps/search/?api=1&query={loc['lat']},{loc['lon']}"
            popup_html = f"<b>{loc['primary_name']}</b><br><a href='{Maps_url}' target='_blank'>Navigate on Google Maps</a>"
            
            # 1. Themed, clickable icon marker
            folium.Marker(
                [loc['lat'], loc['lon']],
                popup=folium.Popup(popup_html, max_width=270),
                tooltip=f"Click for details: {loc['primary_name']}",
                icon=folium.Icon(color=theme['color'], icon=theme['icon'], prefix='fa')
            ).add_to(m)

            # 2. Permanent text label on the map
            label_html = f'<div style="font-family: Arial, sans-serif; font-size: 11px; font-weight: bold; color: #2C3E50; background-color: rgba(255, 255, 255, 0.75); padding: 3px 6px; border-radius: 3px; border: 1px solid rgba(0,0,0,0.2); white-space: nowrap;">{loc["primary_name"]}</div>'
            folium.Marker(
                location=[loc['lat'], loc['lon']],
                icon=folium.features.DivIcon(
                    icon_size=(100, 36),
                    icon_anchor=(-10, 15),
                    html=label_html
                )
            ).add_to(m)
        return m
    except Exception as e:
        st.error(f"🗺️ Map creation failed: {e}"); return None

# --------------- Main Streamlit App ---------------
st.set_page_config(page_title="CampusGPT", page_icon="🏫", layout="wide")
st.markdown("""<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');html,body,[class*="st-"]{font-family:'Inter',sans-serif}.block-container{padding:1rem 2rem 2rem}.main-header h1{font-size:3.5rem;font-weight:700;background:-webkit-linear-gradient(45deg, #5850ec, #a855f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent}.chat-message{display:flex;align-items:flex-start;max-width:85%;margin-bottom:1.5rem}.chat-bubble{padding:1rem 1.25rem;border-radius:1.25rem;box-shadow:0 4px 6px rgba(0,0,0,.05);line-height:1.6;word-wrap:break-word}.user-message{justify-content:flex-end;margin-left:auto}.user-message .chat-bubble{background-color:#5850ec;color:#fff;border-bottom-right-radius:.25rem}.user-message .chat-icon{margin-left:.75rem}.assistant-message{justify-content:flex-start}.assistant-message .chat-bubble{background-color:#f1f5f9;color:#1e293b;border-bottom-left-radius:.25rem}.assistant-message .chat-bubble a{color:#5850ec;font-weight:600}.chat-icon{font-size:1.5rem;color:#94a3b8;align-self:flex-start;margin-top:.25rem}.welcome-card{background-color:#f8fafc;border-left:5px solid #5850ec;padding:2rem;border-radius:.5rem;margin-top:2rem}</style>""", unsafe_allow_html=True)

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "last_location" not in st.session_state: st.session_state.last_location = None

index, corpus, location_map = load_system_data()
system_ready = bool(location_map)

with st.sidebar:
    st.title("🏫 CampusGPT")
    role = st.radio("Select Your Role", ["User", "Admin"], horizontal=True, label_visibility="collapsed")
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history, st.session_state.last_location = [], None
        st.toast("Chat history cleared!", icon="🔄"); st.rerun()

st.markdown("<div class='main-header' style='text-align:center'><h1>CampusGPT</h1><p>Your Smart Campus Assistant</p></div>", unsafe_allow_html=True)

if role == "User":
    if not system_ready:
        st.warning("System not ready. An administrator needs to upload documents first.")
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar='👤' if msg["role"] == "user" else '🏫'):
                st.markdown(msg["content"], unsafe_allow_html=True)
                if "map_data" in msg and msg["map_data"]:
                    map_obj = create_map(msg["map_data"])
                    if map_obj: st_folium(map_obj, width=700, height=450)
        
        if st.session_state.last_location and ors_client:
            st.info(f"A location was found: **{st.session_state.last_location['primary_name']}**")
            if st.button("Get Directions to this location 🚶‍♂️"):
                user_geo = get_geolocation()
                if user_geo:
                    start_coords = (user_geo['coords']['latitude'], user_geo['coords']['longitude'])
                    end_coords = (st.session_state.last_location['lat'], st.session_state.last_location['lon'])
                    with st.spinner("Fetching walking directions..."):
                        directions = get_turn_by_turn_directions(start_coords, end_coords)
                    st.session_state.chat_history.append({"role": "assistant", "content": directions})
                else:
                    st.warning("Could not get your location. Please grant location permission in your browser.")
                st.session_state.last_location = None
                st.rerun()

        if prompt := st.chat_input("Ask about campus locations, or for details..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.last_location = None
            
            with st.spinner("Thinking..."):
                chunks = retrieve_chunks(prompt, corpus, index)
                locs = match_locations(prompt, location_map)
                response_content = ask_chatbot(prompt, chunks, locs)
                response_msg = {"role": "assistant", "content": response_content}
                
                if len(locs) == 1:
                    response_msg["map_data"] = locs
                    st.session_state.last_location = locs[0]

                st.session_state.chat_history.append(response_msg)
                st.rerun()
else: # Admin View
    if not st.session_state.get("authenticated", False):
        st.subheader("🔐 Admin Login")
        password = st.text_input("Enter Password", type="password", key="admin_pass")
        if st.button("🔑 Login"):
            if password == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Incorrect password.")
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
                            if num_s > 0 or num_l > 0:
                                st.success(f"✅ Processing complete! Saved {num_s} sentences and {num_l} locations.")
                                st.balloons()
                            else:
                                st.warning("⚠️ No processable data found in the files.")
                else:
                    st.warning("Please upload at least one file.", icon="❗")
        with tab2:
            st.subheader("📊 System Status")
            st.metric("System Status", "✅ Ready" if system_ready else "❌ Not Ready")
            c1, c2 = st.columns(2)
            c1.metric("Indexed Sentences", len(corpus))
            c2.metric("Known Locations", len(location_map))
            with st.expander("📍 View Available Locations"):
                if location_map:
                    for loc in location_map:
                        st.write(f"• {loc['primary_name']}")
                else:
                    st.write("No locations loaded.")
            st.markdown("---")
            st.subheader("🚨 Danger Zone")
            if st.button("🗑️ Clear All Data & Index", type="secondary"):
                st.session_state.confirm_delete = True
            if st.session_state.get("confirm_delete", False):
                st.warning("**Are you sure?** This will delete all processed data. This action cannot be undone.")
                col_del_1, col_del_2 = st.columns(2)
                if col_del_1.button("Yes, I am sure, delete everything.", type="primary"):
                    try:
                        if os.path.exists(STORAGE_DIR): shutil.rmtree(STORAGE_DIR)
                        if os.path.exists(DOCUMENTS_DIR): shutil.rmtree(DOCUMENTS_DIR)
                        os.makedirs(DOCUMENTS_DIR, exist_ok=True); os.makedirs(STORAGE_DIR, exist_ok=True)
                        st.session_state.confirm_delete = False
                        st.success("All system data has been cleared.")
                        st.rerun()
                    except Exception as e: st.error(f"Failed to clear data: {e}")
                if col_del_2.button("Cancel"):
                     st.session_state.confirm_delete = False
                     st.rerun()
        with tab3:
            st.markdown("Your CSV file should have `name`, `latitude`, `longitude`, and optionally `other names` columns.")
            sample_df = pd.DataFrame({
                'name': ['Central Library', 'Botany Dept'], 'latitude': [34.07, 34.08], 
                'longitude': [74.81, 74.82], 'other names': ['Main Library', 'Bio-Sciences Dept']
            })
            st.dataframe(sample_df, use_container_width=True)
            st.download_button("⬇️ Download CSV Template", sample_df.to_csv(index=False).encode('utf-8'), "campus_locations_template.csv", "text/csv")
