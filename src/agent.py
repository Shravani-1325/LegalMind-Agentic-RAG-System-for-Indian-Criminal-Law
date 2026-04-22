import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import START, END, StateGraph
from src.state import State
from src.nodes import (
    query_preprocessor,
    evaluate_agent,
    rag_retriever,
    citation_formatter,
    groq_general,
    vd_node
)

#-- agent.py is the ARCHITECT of entire system
#-- It just connects all the nodes builds in nodes.py into proper graph rules about who talk to whom
#-- nodes.py = Individual Workers
#-- agent.py = the org chart who shows how they connect each other
 
#>> Routing Fun that reads state and returns a string that maps to next node name 
def route_query(state: State) -> str:
    
    #-- Langraph calls this function after evaluate_agent
    
    route = state["route"]
    print(f"Routing to: {route}")
    return route # yes / no  /vd

#>> Constructs and Compiles the full LangGraph workflow 
def build_graph():
    
    #-- Returns compiled workflow ready for invocation..
    #-- CAlling this once at startup - stores the result
    
    # Initializing StateGraph with our State Schema
    graph = StateGraph(State)
    
    #** --- Adding Nodes ---
    # nodes = boxes in the flowchart (do some work)
    # ("node name(used in edges)", function to call for this node)
    
    graph.add_node("query_preprocessor", query_preprocessor)
    graph.add_node("evaluate_agent", evaluate_agent)
    graph.add_node("rag_retriever", rag_retriever)
    graph.add_node("citation_formatter", citation_formatter)
    graph.add_node("groq_general", groq_general)
    graph.add_node("vd_node", vd_node)
    

    #** --- Adding Normal Edges ---
    # Edges = arrows between boxes (define flow direction)
    # START is the special LangGraph constant that marks entry point of graph
        
    #- S1 normal edge- query preprocessing
    graph.add_edge(START, "query_preprocessor")
    
    #- S2 Evaluating after Preprocessing
    graph.add_edge("query_preprocessor", "evaluate_agent")
    
    #** --- Conditional Edges ---
    # Conditional edge = goes to different places based on a condition 
    
    #- S3 Conditional routing based on evaluate_agent output
    #- route_query reads state["route"] and returns "yes"/"no"/"vd"
    graph.add_conditional_edges(
        "evaluate_agent", 
        route_query,
        {
            "yes" : "rag_retriever", # legal question -> fetch from pdf
            "no": "groq_general", # general question -> GROQ only
            "vd" : "rag_retriever" # partial -> fetch from PDF first
            
            # both yes & vd got to rag_retrival but then diverge
        }
    )
    
    #- S4 After retrieval, route based on original decision
    '''yes path: rag_retriever → citation_formatter
          (answer ONLY from PDF, no extra knowledge)
          
        vd path:  rag_retriever → vd_node  
          (answer from PDF + Groq knowledge combined)
    '''
    
    graph.add_conditional_edges(
        "rag_retriever",
        route_query, # Same routing function - reads state["route"]
        {
            "yes": "citation_formatter",
            "vd": "vd_node"            
        }
    )
    
    #- S5 All three answer nodes lead to END
    graph.add_edge("citation_formatter", END)
    graph.add_edge("groq_general", END)
    graph.add_edge("vd_node", END)
    
    #** Compiliation 
    workflow = graph.compile()
    
    print("LegalMind Graph compiled successfully!!!")
    return workflow
    
#>> Module level graph instance
#-- All other files import this ready_to_use workflow

workflow = build_graph()

#>> Main Entry Point for running a query through LegalMind
def run_query(question: str)-> dict:
    
    #--called by app.py )with user's question.
    #-- returns the complete final state with answer and metadata
    
    # langGraph requires all state fields to be present at the Start
    
    initial_state = { # fills required fields, rest to be blank
        "question" : question,
        "cleaned_question" : "",
        "route" : "",
        "docs": [],
        "sources": [],
        "answer": "",
        "section_refs": [],
        "error": ""
    }
    
    print("\n" + "="*50)
    print(f"Query: {question}")
    print("="*50)

    #-- Invoke runs the full graph synchronously & returns final state after all nodes have run
    final_state= workflow.invoke(initial_state)
    
    print(f"Answer generated via route: {final_state['route']}")
    print("="*50 + "\n")
    
    return final_state
    