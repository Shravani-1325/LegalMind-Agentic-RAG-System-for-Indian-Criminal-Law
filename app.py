import streamlit as st
import os
import sys

# Adding project root to python path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent import run_query  # LangGraph Pipeline
from src.utils import(
    format_sources_display,     #  "CrPC • IPC" display string
    validate_state_output,  # ensuring all the fileds exist
    extract_section_refs        # regex section extractor
)
from config.settings import ACT_SOURCES  # {"IPC": "Indian Penal Code"}

#>> PAGe CONFIG 
st.set_page_config(
    page_title= "LegalMind - Criminal Law AI",
    page_icon= "⚖️",
    layout="wide",
    initial_sidebar_state= "expanded" # sidebae open bydefault
)

#>> CSS LOADER 
# Reads the Custom CSS file and injects it intp the page
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    
    if os.path.exists(css_path):
        with open(css_path) as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html= True)

load_css()

#>> Session State Initializer
def initialize_session_state():
    # (Streamlit reruns entire script on every user interation)
    # The "if not in" check ensures we only initialiuze ONCE


    # Stores the full chat as list of dicts
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Tracks if we are currently waiting for LLM response
    if "is_loading" not in st.session_state:
        st.session_state.is_loading = False
        
    # Stores last full result dict - used by debug panel
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
        
        
initialize_session_state()

#>> Sidebar 
def render_sidebar():
    with st.sidebar:
        
        # Branding
        st.markdown("# ⚖️ LegalMind")
        st.markdown("*Agentic RAG for Criminal Law Intelligence*")
        st.divider()
        
        # Knowledge base info
        st.markdown("### 📚 Knowledge Base")
        st.markdown("""
                    This AI is trained on: 
                    - **Indian Penal Code (IPC)** 1860
                    - **Code of Criminal Procedure (CrPC)** 1973  
                    - **Indian Evidence Act** 1872
                    """)
        st.divider()
        
        # Quick Section Lookup
        st.markdown("### 🔍 Quick Section Lookup")
        
        section_num = st.text_input(
            "Enter Section Number",
            placeholder = "e.g. 154, 302, 420"
        )
        
        if st.button("Look Up", use_container_width=True):
            if section_num:
                # Storing in session_state so main area picks it up
                st.session_state.quick_lookup = f"What is Section {section_num} ?"
            
        st.divider()
        
        # Session Stats 
        st.markdown("### 📊 Session Stats")
        
        total = len(st.session_state.messages)
        user_msgs = len([m for m in st.session_state.messages
                         if m["role"] == "user"])
        
        # st.columns splits the sidebar into equal columns
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Messages", total)
        with col2:
            st.metric("Question Asked", user_msgs)
            
        # Routing Breakdown - only show if there are messages
        if st.session_state.messages:
            
            routes = [m.get("route", "") for m in st.session_state.messages
                      if m["role"] == "assistant"]
            
            yes_count = routes.count("yes")
            vd_count = routes.count("vd")
            no_count = routes.count("no")
            
            st.markdown(f"""
            **Routing Breakdown:**
            - 📄 From Database: {yes_count}
            - 🔀 Combined: {vd_count}  
            - 🧠 General Knowledge: {no_count}
             """)      
            
        st.divider()
        
        # Clear Chat
        if st.button("🗑️ Clear Chat History",
                     use_container_width=True,
                     type="secondary"):
            st.session_state.messages    = []
            st.session_state.last_result = None
            st.rerun()    # forcing immediate full rerun so chat clears instantly
            
render_sidebar() 

#>> Source Badge Renderer 
#>> Renders colored pills showing which ACT the answer came from
def render_source_badges(sources: list):
    
    if not sources:
        return
    
    badge_colors = {
        "IPC":               "#FF4B4B",   # red
        "CrPC":              "#1E88E5",   # blue
        "Evidence":          "#43A047",   # green
        "General Knowledge": "#9E9E9E"    # grey
    }
    
    spans = []
    for source in sources:
        color = badge_colors.get(source, "#9E9E9E") #default grey
        full_name = ACT_SOURCES.get(source, source) # Indian Penal Code
        
        span = (
            f'<span style="background-color:{color};'
                f'color:white;'
                f'padding:3px 10px;'
                f'border-radius:12px;'
                f'font-size:12px;'
                f'font-weight:600;'
                f'margin-right:6px;">'
                f'{full_name}</span>'
        )
        spans.append(span)

    # Wrap ALL spans in ONE div → ONE st.markdown call
    # Multiple st.markdown calls = multiple render passes = HTML breaks
    final_html = '<div style="margin:8px 0">' + "".join(spans) + '</div>'
    st.markdown(final_html, unsafe_allow_html=True)
    
#>> Section Tag Renderer 
def render_section_tags(section_refs: list):
    
    if not section_refs:
        return
    
    st.markdown("**Referenced Section**")
    spans = []
    for ref in section_refs:
        span= (
            f'<span style="'
            f'background-color:#F3E5F5;'    # light purple fill
            f'color:#6A1B9A;'               # dark purple text
            f'padding:3px 10px;'
            f'border-radius:6px;'
            f'font-size:12px;'
            f'font-weight:500;'
            f'margin-right:6px;'
            f'margin-bottom:4px;'
            f'display:inline-block;'
            f'border:1px solid #CE93D8;">'   # purple border
            f'{ref}</span>'
        )
        spans.append(span)
    final_html =  '<div style="margin:6px 0">' + "".join(spans) + '</div>'
    st.markdown(final_html, unsafe_allow_html=True)
    
    
#>> Chat History Rendere 
# Called every rerun to redisplay the full conversation
# Streamlit reruns the whole script on every interaction
# so we rebuild chat from session_state each time

def render_chat_history():
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            
            st.markdown(message["content"]) # display message text
            
            # Only assistant messages get badges + tags + route label
            if message["role"] == "assistant":
                if message.get("sources"):
                    render_source_badges(message["sources"])
                    
            if message.get("section_refs"):
                render_section_tags(message["section_refs"])
                
            # Route Label  small grey text at the bottom of message
            route = message.get("route", "")
            route_labels= {
                    "yes": "📄 Answered from Legal Database",
                    "vd":  "🔀 Combined Database + AI Knowledge",
                    "no":  "🧠 Answered from General Knowledge "
                    
            }
            
            if route in route_labels:
                st.caption(route_labels[route])
                
render_chat_history()

#>> Main Header 
st.markdown("## ⚖️ LegalMind — Criminal Law Intelligence")
st.markdown(
    "Ask questions about **IPC**, **CrPC**, or **Indian Evidence Act**. "
    "Get cited, structured legal answers."
)

# Show description ONLY when chat is empty — welcome screen
# Once user starts chatting it disappears cleanly
if not st.session_state.messages:
    st.markdown("""
    <div style="
        background: rgba(201, 168, 76, 0.06);
        border: 1px solid rgba(201, 168, 76, 0.20);
        border-left: 3px solid #C9A84C;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 16px 0 22px 0;
        backdrop-filter: blur(12px);
    ">
        <p style="
            font-family: 'EB Garamond', serif;
            color: #D4C5A9;
            font-size: 1.12rem;
            line-height: 1.9;
            margin: 0;
        ">
            <strong style="color:#E8C97A;">LegalMind</strong> is an 
            <strong style="color:#E8C97A;">Agentic RAG</strong> 
            (Retrieval-Augmented Generation) system built over 
            <strong style="color:#E8C97A;">6566 pages</strong> of Indian Criminal Law — 
            covering the <em>Indian Penal Code (IPC) 1860</em>, 
            <em>Code of Criminal Procedure (CrPC) 1973</em>, and 
            <em>Indian Evidence Act 1872</em>. Powered by a 
            <strong style="color:#E8C97A;">LangGraph</strong> agent that intelligently 
            routes your query — searching the legal database, combining sources, 
            or drawing from general knowledge — to deliver precise, cited answers 
            with Section references and Act attribution.
        </p>
        <p style="
            font-family: 'EB Garamond', serif;
            color: #8A8F9E;
            font-size: 0.95rem;
            margin: 12px 0 0 0;
            font-style: italic;
        ">
            💡 Try asking: "What is Section 154 CrPC?" or "What is punishment for murder under IPC?"
        </p>
    </div>
    """, unsafe_allow_html=True)




#>> Core Query Processor
# Main Function when user submit the questions
# Sends through full langraph pipeline and renders response

def process_query(user_question: str):
    
    # S1- Adding the user message to history immediately
    st.session_state.messages.append({
        "role" : "user",
        "content" : user_question
    })
    
    # S2- Render user bubble immediately
    with st.chat_message("user"):
        st.markdown(user_question)
        
    # S3- Shows assistant bubble with spinner while processing
    with st.chat_message("assistant"):
        with st.spinner("⚖️ LegalMind is researching..."):
            
            # Initialize result BEFORE try Block
            # If try fails - result still exists with empty defaults
            result = {
                "answer": "",
                "sources": [],
                "section_refs": [],
                "route": "",
                "error": ""
            }
            
            try: 
                #>> Running Full LangGraph pipeline
                #>> preprocessor -> evaluator -> retrievers -> answer node
                result = run_query(user_question)
                
                #>> Validating - ensuring all required fields exist and are clean
                result = validate_state_output(result)
                
            except Exception as e:
                result["answer"] = (
                    f"I encountered an error processing your question"
                    f"Please try again.\n\nError: {str(e)}"
                )
                
                result["route"] = "error"
                
        # S4- Extracting fields from result
        answer = result["answer"]
        sources = result.get("sources", [])
        section_refs = result.get("section_refs", [])
        route = result.get("route", "")
        
        # S5- Render answer, badges, tags, route label
        st.markdown(answer)
        
        if sources:
            render_source_badges(sources) # Colored act badges
            
        if section_refs:
            render_section_tags(section_refs) # purple section pills
            
        route_labels = {
            "yes": "📄 Answered from Legal Database",
            "vd":  "🔀 Combined Database + AI Knowledge",
            "no":  "🧠 Answered from General Knowledge"
        }
        
        if route in route_labels:
            st.caption(route_labels[route])
            
        
    # S6 — Saving assistant response to chat history
    # (so render_chat_history can show it on next rerun)
    st.session_state.messages.append({
        "role":         "assistant",
        "content":      answer,
        "sources":      sources,
        "section_refs": section_refs,
        "route":        route
    })
    
    # S7 - Saving Full result for debug panel
    st.session_state.last_result = result
            
            
        
#>> Quick Lookup Trigger 
# checking for it here in the main area and process it
# None check prevents re-triggering on every rerun

if hasattr(st.session_state, "quick_lookup") and st.session_state.quick_lookup:
    question = st.session_state.quick_lookup
    st.session_state.quick_lookup = None    # clear FIRST to prevent re-triggering
    process_query(question)


#>> Chat Input 
# Fixed text box at bottom of page
# st.chat_input returns typed text when user hits Enter
# Returns None if user hasn't typed anything yet
# Walrus operator := assigns AND checks in one line

if user_input := st.chat_input("Ask about IPC, CrPC, or Evidence Act..."):
    process_query(user_input)
    st.rerun()    # force full rerun so sidebar stats update immediately      
        
        
#>> Debug Panel 
# Collapsible section showing raw pipeline data
# Useful during development — remove or hide before final demo

# with st.expander("🔧 Debug Info", expanded=False):
    
#     if st.session_state.last_result:
#         result = st.session_state.last_result

#         # Three columns for clean layout
#         col1, col2, col3 = st.columns(3)

#         with col1:
#             st.markdown("**Original Question:**")
#             st.code(result.get("question", ""))         # raw user input

#             st.markdown("**Cleaned Question:**")
#             st.code(result.get("cleaned_question", "")) # after preprocessor

#         with col2:
#             st.markdown("**Route Taken:**")
#             st.code(result.get("route", ""))            # yes / vd / no

#             st.markdown("**Sources:**")
#             st.code(str(result.get("sources", [])))     # which acts found

#         with col3:
#             st.markdown("**Section Refs:**")
#             st.code(str(result.get("section_refs", [])))  # sections extracted

#             st.markdown("**Docs Retrieved:**")
#             st.code(f"{len(result.get('docs', []))} documents")  # FAISS count

#     else:
#         st.info("Ask a question to see debug information here.")