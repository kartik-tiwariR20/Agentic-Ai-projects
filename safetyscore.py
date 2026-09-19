from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict,Annotated
from langgraph.graph import StateGraph,START,END

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
)

model = ChatHuggingFace(llm=llm)

def mergescore(existing:dict , newupdate:dict)->dict:
    if existing is None:
        return newupdate
    return {**existing,**newupdate}

class Analyzerpipeline(TypedDict):
    raw_text : str
    safety_scores : Annotated[dict[str,int],mergescore]

def toxicity_node(state: Analyzerpipeline) -> dict:
    print("\n [Branch 1] Analyzing Toxicity and Hate Speech...")
    prompt = (
        "Analyze the following text for profanity, aggression, hate speech, or toxicity. "
        "Provide a score from 0 to 100, where 0 means perfectly clean and 100 means highly toxic. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = model.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    return {"safety_scores": {"toxicity_level": score}}

def copyright_node(state: Analyzerpipeline) -> dict:
    print("\n [Branch 2] Analyzing Copyright & Originality Risks...")
    prompt = (
        "Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal, "
        "or presents a corporate trademark risk. Provide a score from 0 to 100, "
        "where 0 means entirely original and 100 means high risk. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = model.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    return {"safety_scores": {"copyright_risk": score}}


def culture_node(state: Analyzerpipeline) -> dict:
    print("\n🌍 [Branch 3] Analyzing Regional & Cultural Sensitivity...")
    prompt = (
        "Analyze the following text for regional sensitivities, political landmines, "
        "or cultural insensitivity that might offend a global audience. Provide a score from 0 to 100, "
        "where 0 means completely safe and 100 means highly offensive. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = model.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    return {"safety_scores": {"cultural_insensitivity": score}}

graph = StateGraph(Analyzerpipeline)

graph.add_node("toxicity_node",toxicity_node)
graph.add_node("copyright_node",copyright_node)
graph.add_node("culture_node",culture_node)

graph.add_edge(START,"toxicity_node")
graph.add_edge(START,"copyright_node")
graph.add_edge(START,"culture_node")

graph.add_edge("toxicity_node",END)
graph.add_edge("copyright_node",END)
graph.add_edge("culture_node",END)

app = graph.compile()

raw_text = input("enter the text : ")

initial_state = {
    "raw_text" : raw_text,
    "safety_scores" :{}
}

final_state = app.invoke(initial_state)
print(final_state["safety_scores"])
