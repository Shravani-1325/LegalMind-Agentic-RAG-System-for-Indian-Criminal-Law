import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings import(
    VECTOR_DB_PATH,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
    SEARCH_TYPE
)

_vector_store = None
_embeddings = None

#>> Loading FAISS Index & Embeddings
def load_vector_store():
    
    #-- Using Module-level caching so it only loads once
    
    global _vector_store, _embeddings
    
    if _vector_store is not None:
        return _vector_store, _embeddings
    
    print("Loading embedding Model...")
    
    _embeddings = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        encode_kwargs = {"normalize_embeddings": True}
    )
    
    print("Loading FAISS vectos store..")
    
    #-- If Databse doesnt exist
    if not os.path.exists(VECTOR_DB_PATH):
        raise FileNotFoundError(
            f"Vector Store not Found at {VECTOR_DB_PATH}"
        )
        
    _vector_store = FAISS.load_local(
        VECTOR_DB_PATH,
        _embeddings,
        allow_dangerous_deserialization=True
    )
     
    print("Vector Store loaded Successfully")
    return _vector_store, _embeddings



#>> Retrival Function
def retrive_documents(query: str) -> tuple[list,list]:
    
    #-- Takes User Query, Returns Relevant Chunks ie, [docs] & [sources]
    
    vector_store, _ = load_vector_store()
    
    #-- diverse and relevant results
    if SEARCH_TYPE == "mmr":
        docs = vector_store.max_marginal_relevance_search(
            query,
            k = TOP_K_RESULTS, # how many final res to return
            fetch_k= TOP_K_RESULTS * 3 # candidates to consider before MMR filtering
            
        )
        
    else:
        #-- Plain similiarty search
        docs = vector_store.similarity_search(
            query,
            k = TOP_K_RESULTS
        )
        
    #-- Extracting Unique src
    sources = list(set(
        doc.metadata.get("source", "Unkown")
        for doc in docs
    ))
    
    return docs, sources


#>> Converting list of Document into single clean string
def format_docs_for_prompt(docs: list) -> str:
    
    #-- string can be inserted into prompt template
    
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        
        # Chuck clearly labeled with its source
        formatted.append(
            f"[Document {i} | Source: {source} | Page {page}]\n"
            f"{doc.page_content}"
        )
        
    # Joining all chunks with clear separator
    return "\n\n---\n\n".join(formatted)
        
        
#>> langchain Compatible retriver Object
def get_retriver():
    
    #-- used by nodes
    vector_store, _ = load_vector_store()
    
    return vector_store.as_retriever(
        search_type = SEARCH_TYPE,
        search_kwargs = {
            "k" : TOP_K_RESULTS,
            "fetch k": TOP_K_RESULTS *3
        }
    )