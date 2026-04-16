import os
import sys

#>> Adding Project Root to import from config/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings import(
    PDF_PATH,
    VECTOR_DB_PATH,
    EMBEDDING_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE
)

#>> RecursivechanracterSplitters
# It splits by paragraphs first, then line, then sentences, then words, then char, to keep related text together as much as possible

#>> Act Sources
def detect_act_source(text: str) -> str:
    
    #-- Detecting which Indian law act a chunk belongs to
    
    text_lower = text.lower()
    
    # IPC
    if any(keyword in text_lower for keyword in [
        "indian penal code",
        "whoever commits",
        "shall be punished with imprisonment",
        "ipc"
    ]):
        return "IPC"
    
    # Evidence
    elif any(keyword in text_lower for keyword in [
        "indian evidence act",
        "relevant fact",
        "admissibility",
        "burden of proof",
        "examination of witness"
    ]):
        return "Evidence"
    
    # CrPC
    else:
        return "CrPC"
    

#>> Loadig PDF and Splitting into Chunks
def load_and_split_pdf() -> list:
         
    print(f"Loading PDF from : {PDF_PATH}")
    
    #-- Pypdf reads the pdf page by page
    
    loader = PyPDFLoader(str(PDF_PATH))
    pages = loader.load()        
    print(f"Total Pages Loaded: {len(pages)}")
    
    #-- RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators=["\n\n","\n",". "," ",""]
    )
    
    chunks = splitter.split_documents(pages)
    print(f"Total chunks created : {len(chunks)}")
    
    return chunks

#>> Metadata
def add_metedata(chunks: list) -> list:
    
    #-- Tagging every chunk with which act is belongs to
    
    for i, chunk in enumerate(chunks):
     
        source = detect_act_source(chunk.page_content)
        
        #-- Adding Custom Metadata on Existing Page
        chunk.metadata["source"] = source
        chunk.metadata["chunk_id"] = i
        
    return chunks
       
#>> Creating Vector Database
def create_vector_store(chunks: list) -> None:
    
    #-- Converting chunks into vector and saving to FAISS
    
    print(f"Loading Embedding Model: ",{EMBEDDING_MODEL})
    
    #-- HuggingFace Loads the Model Locally
    embeddings = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        encode_kwargs = {"normalize_embeddings": True}
    ) 
        
    print("Creating FAISS index....")
    
    #-- FAISS.from_documents : 
    # 1. Calls embeddings.embed_documents() on every chunk 
    # 2. Builds the FAISS index from those vectors
    
    vector_store = FAISS.from_documents(
        documents = chunks,
        embedding=embeddings
    ) 
        
    #-- Saving vectors to database
    os.makedirs(VECTOR_DB_PATH,exist_ok=True)
    vector_store.save_local(VECTOR_DB_PATH)
    
    print(f"Vector store saved to: {VECTOR_DB_PATH}")
    print(f"Total vectors indexed: {len(chunks)}")
    
    
#>> Calling All the Functions
def run_ingestion() -> None:
    
    #-- Main Function
    
    print("=" * 50)   
    print("LegalMind - Ingestion Pipeling Processing")
    print("=" * 50)
    
    chunks = load_and_split_pdf() # Step 1 : Splitting Pdf
    chunks = add_metedata(chunks) # Step 2:  Tagging Manually Metadata
    create_vector_store(chunks) # Creating Embeddings & Saving
    
    print("=" * 50)
    print("Ingestion Completed !!!")
    print("=" * 50)
    

if __name__ == "__main__":
    run_ingestion()
         
        
        
    
    
    
   
