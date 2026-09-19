from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class pipelinestate(TypedDict):
    raw_input : str
    edited_text : str
    script_text : str
    final_output : str

def editor_node(state :pipelinestate) -> dict:

    prompt = (
        "You are an expert copyeditor. Clean up the following raw text. "
        "Fix any grammatical errors, spelling mistakes, and smooth out the transition flow "
        "while keeping the core message intact. Return only the edited text.\n\n"
        f"Text:\n{state['raw_input']}"
    )
    response = model.invoke(prompt)

    return {"edited_text" : response.content.strip()}

def scriptwriter_node(state: pipelinestate) -> dict:
    
    print("\n--- [Stage 2] Executing Scriptwriter Node ---")
    
    prompt = (
        "You are a charismatic YouTube content creator. Take this edited text and transform "
        "it into a highly engaging, punchy, conversational video script hook. Make it sound "
        "like a real person speaking passionately. Return only the script content.\n\n"
        f"Edited Text:\n{state['edited_text']}"
    )
    
    response = model.invoke(prompt)
    return {"script_text": response.content.strip()}

def translator_node(state: pipelinestate) -> dict:
   
    print("\n--- [Stage 3] Executing Hinglish Translator Node ---")
    
    prompt = (
        "You are an expert content localizer for the Indian market. Take the following script "
        "and convert it into natural, flowing 'Hinglish'. Do not simply translate it sentence-by-sentence "
        "or repeat information. Alternating comfortably between Hindi and English phrases just like "
        "an intellectual tech educator would speak naturally on a live stream. Keep the energy high! "
        "Return only the final Hinglish text.\n\n"
        f"Script:\n{state['script_text']}"
    )
    
    response = model.invoke(prompt)
    return {"final_output": response.content.strip()}

from langgraph.graph import StateGraph,START,END

graph = StateGraph(pipelinestate)

graph.add_node("editor",editor_node)
graph.add_node("scriptwriter",scriptwriter_node)
graph.add_node("translator",translator_node)

graph.add_edge(START,"editor")
graph.add_edge("editor","scriptwriter")
graph.add_edge("scriptwriter","translator")
graph.add_edge("translator",END)

app = graph.compile()

input = input("enter the content : ")

result = app.invoke({
    "raw_input":input
})

print(result['final_output'])
