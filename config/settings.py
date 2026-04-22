from pathlib import Path
from dotenv import load_dotenv
import os

#>> LLM Loading
import os
from dotenv import load_dotenv

# Only import streamlit safely
try:
    import streamlit as st
except:
    st = None

# Load local env
load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"

# Hybrid approach (works BOTH locally + cloud)
if os.getenv("GROQ_API_KEY"):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
elif st is not None:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = None

#>> Base Directory 

BASE_DIR = Path(__file__).parent.parent

#>> File Directory 

PDF_PATH = BASE_DIR/"data"/"IPC_CrPC_Evidence.pdf"

#>> Vector Database 

# str() -- FAISS.save_local() uses a string
VECTOR_DB_PATH = str(BASE_DIR/"vectorDatabase")

#>> Embedding Model 

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-l6-v2"

# converts text into 384-dimensional vectors. 
# Two sentences that mean the same thing will have vectors that are close together in that 384-dimensional space.

#>> Chunks 

CHUNK_SIZE = 1000 # each piece of text from pdf is ~1000 char,
CHUNK_OVERLAP = 150 # last 150 char of chunk1 are repeated at start of chunk 2, prevents loosing context.

#>> Retrieval Setting 

TOP_K_RESULTS = 5 # Returns 5 chunks
SEARCH_TYPE = "mmr" # Maximal marginal Relevance 
#  returns results that are both relevant AND diverse

#>> Act Sorces Label

ACT_SOURCES = {
    "IPC" : "Indian Penal Codel",
    "CrPC" : "Code of Criminal Procedure",
    "Evidence" : "Indian Evidence Act"
}