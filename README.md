# CampusGPT 🏫

CampusGPT is a multilingual, map-aware chatbot designed for college campuses. It allows users to ask questions from uploaded institutional documents and locate buildings like the Admin Block or IT Department on a map.

## Features
- Upload academic PDFs (admin only)
- Ask questions in any language
- Find department infoand locations on campus maps
- Built with Groq LLM, FAISS, and Streamlit

## Roles
- **Admin**: Upload documents and rebuild the knowledge index
- **User**: Ask location- or institute-related questions

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
