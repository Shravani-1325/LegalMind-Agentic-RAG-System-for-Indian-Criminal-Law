from typing import TypedDict, Literal

#>> State 
# -- State is like a shared notebook that get passed to every node in the graph
# -- Without State, nodes can't talk to each other

class State(TypedDict):
   
    question : str
    cleaned_question : str
    route : Literal["yes", "no", "vd" ]
    docs : list
    sources : list
    answer : str
    section_refs : list
    error : str
    

    
    