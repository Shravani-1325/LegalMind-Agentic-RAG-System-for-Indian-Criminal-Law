import os
import sys

#>> Getting setting.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from src.retrieval import retrive_documents, format_docs_for_prompt
from src.prompts import(
    PREPROCESSOR_PROMPT,
    EVALUATOR_PROMPT,
    RAG_PROMPT,
    GENERAL_PROMPT,
    VD_PROMPT
)  
from src.state import State
from config.settings import GROQ_MODEL, GROQ_API_KEY

#>> LLM API Initialization 
llm = ChatGroq(model = GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)

#>> n1. Cleans and Normalize the Raw User Question 
def query_preprocessor(state: State) -> dict:
    
    raw_question = state["question"]
    
    #-- building & invoking preprocessing Chain
    chain = PREPROCESSOR_PROMPT | llm
    result = chain.invoke({"question": raw_question})
    
    cleaned = result.content.strip()
    
    print(f"Original : {raw_question}")
    print(f"Cleaned: {cleaned}")
    
    return {"cleaned_question": cleaned}

#>> n2. Classifies the cleaned query for deciding route to [yes,no,vd] 
def evaluate_agent(state: State) -> dict:
    
    #-- "yes" : answer directly from FAISS
    #-- "vd" : answer from FAISS + GROQ combined
    #-- "no" : ansewr from GROQ general Knowlwdge
    
    question = state["cleaned_question"]     
    chain = EVALUATOR_PROMPT | llm    
    result = chain.invoke({"question": question})
    
    route = result.content.strip().lower()
    
    #-- FAllBACK
    if route not in["yes", "no", "vd"]:
        print(f"Unexpected route '{route}' - defaulting to vd")
        route = "vd"
    
    print("Route decided: ",route)
    return {"route" : route}

#>> n3 Fetching Relevant chunks from Vector Database FAISS
def rag_retriever(state: State) -> dict:
    
    #--> Node  only retrive the answer from FAISS it doesnt generate anything
    
    question = state["cleaned_question"]
    
    try:
        docs, sources = retrive_documents(question)
        
        print(f"Retrieved {len(docs)} documents")
        print(f"Sources Found: {sources}")
        
        return {
            "docs": docs,
            "sources" : sources,
            "error": ""
            
        }
        
    except Exception as e:
        # Storing Error in state if FAISS search fails
        
        print(f"Retrieval error: {e}")
        return {
            "docs": [],
            "sources": [],
            "error": f"Retrieval failed: {str(e)}"
        }
        
#>> n4 "Yes" Route -> Legal Answers from retrieved docs
def citation_formatter(state: State) -> dict:
    
    #-- "Yes" -> Documents(pdf) based answer
    
    #-- retrieving error
    if state.get("error"):
        return {
            "answer" : f"Error has been encountered retrieving documents. {state["error"]}",
            "section_refs": []
        } 
        
    #-- If pdf doesnt has answer
    if not state["docs"]:
        return{
            "answer": "I cound not find relevant information in the legal database for the question",
            "section_refs": []
        }
        
    question = state["cleaned_question"]
    
    #-- formating doc to clean string
    context = format_docs_for_prompt(state["docs"])
    
    chain = RAG_PROMPT | llm
    result = chain.invoke({
        "question" : question,
        "context" : context
    })
    
    answer = result.content
    
    #-- Extracting section references from the answer
    # utils.py will be handling the regex
    
    from src.utils import extract_section_refs
    section_refs = extract_section_refs(answer)
    
    print(f"Section references found: {section_refs}")
    
    return {
        "answer": answer,
        "section_refs": section_refs
    }
    
#>> n5 "No" route -> Answering the question frm GROQ GK
def groq_general(state: State) -> dict:
    
    #-- "no" -> Qqestion not in Database
    
    question = state["cleaned_question"]
    chain = GENERAL_PROMPT | llm
    result = chain.invoke({"question": question})
    
    return {
        "answer": result.content,
        "section_refs": [], # No section refs for GK
        "sources": ["General Knowledge"] # Overwriting sources
        
    }

#>> n6 "vd" route -> Combning FAISS retrieval with GROQ gk 
def vd_node(state: State) -> dict:
    
    #-- "vd" node -> needs both legal document(pdf) context and eleboration llm
    
    question = state["cleaned_question"]
    
    # Checking first if we have docs
    if state["docs"]:
        context = format_docs_for_prompt(state["docs"])
    
    else:
       # No docs retrieved - get ans from gk
       context = "No relevant documents found in the legal database"
       
    chain = VD_PROMPT | llm
    result = chain.invoke({
        "question": question,
        "context": context
    }) 
    
    answer = result.content
    
    from src.utils import extract_section_refs
    section_refs = extract_section_refs(answer)
    
    return {
        "answer" : answer,
        "section_refs" : section_refs
    }
    
    