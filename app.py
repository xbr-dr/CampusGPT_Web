import streamlit as st
import pandas as pd
import PyPDF2
import numpy as np
import faiss
import folium
from streamlit_folium import st_folium
from sentence_transformers import SentenceTransformer
from langdetect import detect
import io
import json
import os
import re
from groq import Groq
import pickle
from typing import List, Dict, Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="Campus Information Assistant",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = []
if 'location_data' not in st.session_state:
    st.session_state.location_data = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'multilingual_model' not in st.session_state:
    st.session_state.multilingual_model = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load models (cached)
@st.cache_resource
def load_models():
    """Load sentence transformer models"""
    try:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        multilingual_model = SentenceTransformer('sentence-transformers/LaBSE')
        return model, multilingual_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_tables_from_pdf(pdf_file) -> List[pd.DataFrame]:
    """Extract tables from PDF (basic implementation)"""
    # Note: For production use, consider using tabula-py or pdfplumber
    # This is a simplified version
    try:
        text = extract_text_from_pdf(pdf_file)
        # Basic table detection - look for tabular patterns
        lines = text.split('\n')
        tables = []
        
        for line in lines:
            # Simple heuristic: if line has multiple numbers/coordinates
            if re.search(r'\d+\.\d+.*\d+\.\d+', line):
                # Try to parse as location data
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        # Attempt to extract location info
                        name = ' '.join(parts[:-3])
                        lat = float(parts[-3])
                        lon = float(parts[-2])
                        desc = parts[-1] if len(parts) > 3 else ""
                        
                        df = pd.DataFrame({
                            'name': [name],
                            'latitude': [lat],
                            'longitude': [lon],
                            'description': [desc]
                        })
                        tables.append(df)
                    except:
                        continue
        
        return tables
    except Exception as e:
        st.error(f"Error extracting tables from PDF: {str(e)}")
        return []

def process_text_to_sentences(text: str) -> List[str]:
    """Split text into sentences for RAG"""
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
    return sentences

def process_location_data(df: pd.DataFrame) -> List[Dict]:
    """Process DataFrame to extract location information"""
    locations = []
    
    # Common column name variations
    name_cols = ['name', 'location', 'place', 'building', 'facility']
    lat_cols = ['latitude', 'lat', 'y']
    lon_cols = ['longitude', 'lon', 'lng', 'x']
    desc_cols = ['description', 'desc', 'info', 'details']
    
    # Find the correct columns
    df_lower = df.columns.str.lower()
    
    name_col = next((col for col in df.columns if df_lower[df.columns.get_loc(col)] in name_cols), None)
    lat_col = next((col for col in df.columns if df_lower[df.columns.get_loc(col)] in lat_cols), None)
    lon_col = next((col for col in df.columns if df_lower[df.columns.get_loc(col)] in lon_cols), None)
    desc_col = next((col for col in df.columns if df_lower[df.columns.get_loc(col)] in desc_cols), None)
    
    if name_col and lat_col and lon_col:
        for _, row in df.iterrows():
            try:
                location = {
                    'name': str(row[name_col]),
                    'latitude': float(row[lat_col]),
                    'longitude': float(row[lon_col]),
                    'description': str(row[desc_col]) if desc_col else ""
                }
                locations.append(location)
            except (ValueError, TypeError):
                continue
    
    return locations

def create_faiss_index(texts: List[str], model) -> faiss.IndexFlatIP:
    """Create FAISS index for text embeddings"""
    if not texts or not model:
        return None
    
    try:
        embeddings = model.encode(texts, convert_to_tensor=False)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        return index
    except Exception as e:
        st.error(f"Error creating FAISS index: {str(e)}")
        return None

def search_knowledge_base(query: str, index, knowledge_base: List[str], model, k: int = 3) -> List[str]:
    """Search knowledge base using FAISS"""
    if not index or not knowledge_base or not model:
        return []
    
    try:
        query_embedding = model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = index.search(query_embedding, k)
        
        results = []
        for idx in indices[0]:
            if idx < len(knowledge_base):
                results.append(knowledge_base[idx])
        
        return results
    except Exception as e:
        st.error(f"Error searching knowledge base: {str(e)}")
        return []

def find_location_in_query(query: str, locations: List[Dict]) -> Optional[Dict]:
    """Find if query mentions any known location"""
    query_lower = query.lower()
    
    for location in locations:
        if location['name'].lower() in query_lower:
            return location
    
    return None

def generate_answer_with_groq(query: str, context: List[str], groq_api_key: str) -> str:
    """Generate answer using Groq API"""
    if not groq_api_key:
        return "Please configure Groq API key in the sidebar to get AI-generated responses."
    
    try:
        client = Groq(api_key=groq_api_key)
        
        context_text = "\n".join(context) if context else "No relevant context found."
        
        prompt = f"""Based on the following context, please answer the user's question. If the context doesn't contain relevant information, please say so.

Context:
{context_text}

Question: {query}

Answer:"""

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def admin_page():
    """Admin page for file upload and processing"""
    st.title("🔧 Admin Dashboard")
    st.markdown("Upload and process files to build the knowledge base")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload files (PDF, CSV, XLSX, XLS)",
        type=['pdf', 'csv', 'xlsx', 'xls'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.subheader("Processing Files")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_texts = []
        all_locations = []
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}...")
            
            try:
                if file.type == "application/pdf":
                    # Process PDF
                    text = extract_text_from_pdf(file)
                    if text:
                        sentences = process_text_to_sentences(text)
                        all_texts.extend(sentences)
                        st.success(f"Extracted {len(sentences)} sentences from {file.name}")
                    
                    # Try to extract tables from PDF
                    tables = extract_tables_from_pdf(file)
                    for table in tables:
                        locations = process_location_data(table)
                        all_locations.extend(locations)
                
                elif file.type in ["text/csv", "application/vnd.ms-excel", 
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                    # Process CSV/Excel
                    if file.name.endswith('.csv'):
                        df = pd.read_csv(file)
                    else:
                        df = pd.read_excel(file)
                    
                    st.dataframe(df.head())
                    
                    # Try to extract locations
                    locations = process_location_data(df)
                    all_locations.extend(locations)
                    
                    if locations:
                        st.success(f"Extracted {len(locations)} locations from {file.name}")
                    
                    # Also process as text for RAG
                    text_data = df.to_string()
                    sentences = process_text_to_sentences(text_data)
                    all_texts.extend(sentences)
            
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Update session state
        if all_texts:
            st.session_state.knowledge_base.extend(all_texts)
            st.success(f"Added {len(all_texts)} text segments to knowledge base")
        
        if all_locations:
            st.session_state.location_data.extend(all_locations)
            st.success(f"Added {len(all_locations)} locations to database")
        
        # Create/update FAISS index
        if st.session_state.knowledge_base and st.session_state.model:
            status_text.text("Creating search index...")
            st.session_state.faiss_index = create_faiss_index(
                st.session_state.knowledge_base, 
                st.session_state.model
            )
            if st.session_state.faiss_index:
                st.success("Search index created successfully!")
        
        status_text.text("Processing complete!")
    
    # Display current knowledge base stats
    st.subheader("📊 Knowledge Base Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Text Segments", len(st.session_state.knowledge_base))
        
    with col2:
        st.metric("Locations", len(st.session_state.location_data))
    
    # Display locations
    if st.session_state.location_data:
        st.subheader("📍 Stored Locations")
        locations_df = pd.DataFrame(st.session_state.location_data)
        st.dataframe(locations_df)
        
        # Clear data button
        if st.button("🗑️ Clear All Data"):
            st.session_state.knowledge_base = []
            st.session_state.location_data = []
            st.session_state.faiss_index = None
            st.rerun()

def user_page():
    """User page for chat interface"""
    st.title("💬 Campus Assistant")
    st.markdown("Ask questions about the campus or request navigation help!")
    
    # Chat interface
    st.subheader("Chat")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, (user_msg, bot_msg, location_info) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(user_msg)
            
            with st.chat_message("assistant"):
                st.write(bot_msg)
                
                if location_info:
                    st.subheader(f"📍 {location_info['name']}")
                    if location_info['description']:
                        st.write(location_info['description'])
                    
                    # Create map
                    m = folium.Map(
                        location=[location_info['latitude'], location_info['longitude']], 
                        zoom_start=16
                    )
                    folium.Marker(
                        [location_info['latitude'], location_info['longitude']],
                        popup=location_info['name'],
                        tooltip=location_info['name']
                    ).add_to(m)
                    
                    st_folium(m, height=300, width=700)
    
    # Chat input
    user_input = st.chat_input("Ask me anything about the campus...")
    
    if user_input:
        # Detect language
        try:
            detected_lang = detect(user_input)
            is_english = detected_lang == 'en'
        except:
            is_english = True
        
        # Choose appropriate model
        search_model = st.session_state.model if is_english else st.session_state.multilingual_model
        
        # Search knowledge base
        relevant_context = []
        if st.session_state.faiss_index and search_model:
            relevant_context = search_knowledge_base(
                user_input, 
                st.session_state.faiss_index, 
                st.session_state.knowledge_base, 
                search_model
            )
        
        # Check for location mentions
        location_info = find_location_in_query(user_input, st.session_state.location_data)
        
        # Generate response
        groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
        response = generate_answer_with_groq(user_input, relevant_context, groq_api_key)
        
        # Add to chat history
        st.session_state.chat_history.append((user_input, response, location_info))
        
        st.rerun()

def main():
    """Main application"""
    # Load models
    if st.session_state.model is None:
        with st.spinner("Loading AI models..."):
            st.session_state.model, st.session_state.multilingual_model = load_models()
    
    # Sidebar navigation
    st.sidebar.title("🏫 Campus Assistant")
    page = st.sidebar.selectbox("Select Page", ["User", "Admin"])
    
    # API Configuration
    st.sidebar.subheader("⚙️ Configuration")
    st.sidebar.markdown("Enter your Groq API key to enable AI responses")
    
    # Model status
    st.sidebar.subheader("🤖 Model Status")
    if st.session_state.model:
        st.sidebar.success("✅ English model loaded")
    else:
        st.sidebar.error("❌ English model not loaded")
    
    if st.session_state.multilingual_model:
        st.sidebar.success("✅ Multilingual model loaded")
    else:
        st.sidebar.error("❌ Multilingual model not loaded")
    
    # Data status
    st.sidebar.subheader("📊 Data Status")
    st.sidebar.info(f"Knowledge Base: {len(st.session_state.knowledge_base)} items")
    st.sidebar.info(f"Locations: {len(st.session_state.location_data)} items")
    
    # Page routing
    if page == "Admin":
        admin_page()
    else:
        user_page()

if __name__ == "__main__":
    main()
