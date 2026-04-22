import re # Regular Expression
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import ACT_SOURCES

# File is basically of HELPER FUNCTIONS that are used across multiple files

#>> Scans answer text and extract all the section numbers references in answer
def extract_section_refs(text: str) -> list:
    
    #-- o/p = ["Section 152", "Section 156"]
    
    if not text:
        return []
    
    #** Pattern 1 : "Section 154" or "Section 154A" or "Section 154(1)"
    pattern_section = r"Section\s+\d+[A-Za-z]?(?:\(\d+\))?"
    
    #** Pattern 2: "Sec. 154" shortcut 
    pattern_sec =r"Sec\.\s+\d+[A-Za-z]?"
    
    #** Pattern 3 : "u/s 154" -> under section 154 (common legal shortcut)
    pattern_us = r"[Uu]/[Ss]\s+\d+[A-Za-z]?"

    #-- re.findall returns list of all matches in the text
    sections = re.findall(pattern_section, text)
    secs = re.findall(pattern_sec, text)
    us = re.findall(pattern_us, text)
    
    #-- combine all matches
    all_refs = sections + secs + us
    
    # Removing Duplicates - dict.fromkeys() - preserves the unique keys
    unique_refs = list(dict.fromkeys(all_refs))
    
    return unique_refs

#>> Detecting which Indian law act piece of text belongs to 
def detect_act_from_text(text: str) -> str:
    
    #-- Used during ingestion for metadata tagging & during retrieval for source badge display
    #-- Returns: "IPC"/"CrPC"/"Evidence"/"Unknown"
    
    if not text:
        return "Unknown"
    
    
    text_lower = text.lower()
    text_lower = text.lower()
    
    # Scoring each act - more keyword matches = hight score
    
    scores = {"IPC": 0, "CrPC": 0, "Evidence": 0}
    
    #** IPC KEywords
    ipc_keywords = [
        "indian penal code",
        "whoever commits",
        "shall be punished",
        "imprisonment for life",
        "culpable homicide",
        "theft", "robbery", "murder",
        "assault", "cheating", "forgery"
        
    ]
    
    #** CrPC keyword
    crpc_keywords = [
        "code of criminal procedure",
        "magistrate",
        "first information report",
        "cognizable offence",
        "non-cognizable",
        "bail", "warrant", "summons",
        "police officer", "investigation",
        "trial", "charge sheet"
    ]
    
    #** Evidence Act Keywords
    evidenced_keywords = [
        "indian evidence act",
        "relevant fact",
        "admissibility",
        "burden of proof",
        "examination of witness",
        "cross examination",
        "hearsay", "confesssion",
        "documentary evidence"
    ]
    
    #** Counting matching for each act
    
    for keyword in ipc_keywords:
        if keyword in text_lower:
            scores["IPC"] +=1 
            
    for keyword in crpc_keywords:
        if keyword in text_lower:
            scores["CrPC"] += 1
            
    for keyword in evidenced_keywords:
        if keyword in text_lower:
            scores["Evidence"] += 1
                    
    # Returning act with highest scores
    best_match = max(scores, key = lambda k:scores[k])
    
    # IF no keywords matched at all -> Unknown
    if scores[best_match] == 0:
        return "CrPC" 
    
    return best_match    

#>>  Converting source list to human-readable strig for app
def format_sources_display(sources: list) -> str:
    
    #-- I/p:  ["CrPC", "IPC"]
    #-- O/p: "Code of Criminal Procedure (CrPC) • Indian Penal Code 
   
    if not sources:
        return "Sources: Legal Database"
    
    # ACT_SOURCES from setting.py
    display_names = []
    
    for source in sources:
        if source in ACT_SOURCES:
            full_name = f"{ACT_SOURCES[source]} ({source})"
            display_names.append(full_name)
            
        else:
            display_names.append(source)
        
    
    return " • ".join(display_names)

#>> Cleaning Common LLM output artifacts before displaying to user
def clean_llm_response(text: str) -> str:
    
    # LLMs sometimes add:
    # - Extra blank lines at start/end, some unwanted prefix, repeatition
    
    if not text:
        return "I was unable to generate an answer, Please try again"
    
    #-- Extra whitespace
    text = text.strip()
    
    #-- Removing Common LLM Preambles that slip through
    unwanted_prefixes = [
        "Answer: ", "Response:", "Here is my answer:",
        "Based on the context provided,",
        "Based on the documents provided,"
    ]
    
    for prefix in unwanted_prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            
    # Collaping multiple blank lines into single blank line   
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text

#>> Final safety check before returning answer to app
def validate_state_output(state: dict) -> dict:
    
    #-- Ensuring all required fields exist and have valid values
    
    #** If answer empty for any reason 
    if not state.get("answer"):
        state["answer"] = (
          "I was unable to find relevant information for your question"
          "Please try Rephrasing or Ask about a specific IPc/CrPC/Evidence section"
        )
        
    #** If sources somehow missing  
    if not state.get("sources"):
        state["sources"] = ["Legal Database"]
        
    #** if section_refs missing
    if not state.get("section_refs"):
        state["section_refs"] = []
        
    #** Clean Answer
    state["answer"] = clean_llm_response(state["answer"])
    
    return state
