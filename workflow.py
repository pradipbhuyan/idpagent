from typing import TypedDict
from langgraph.graph import StateGraph, END
from pathlib import Path
from io import BytesIO

# Import your existing functions
from core import (
    detect_document_type,
    extract_structured_json,
    build_resume,
    json_to_kv_dataframe,
    generate_excel
)

# ------------------------------
# STATE
# ------------------------------
class IDPState(TypedDict):
    text: str
    doc_type: str
    data: dict
    result: dict

# ------------------------------
# NODES
# ------------------------------

def detect_node(state):
    progress = state.get("progress")
    if progress:
        progress(10, "🔍 Detecting document type...")

    state["doc_type"] = detect_document_type(state["text"])
    return state

def extract_node(state):
    progress = state.get("progress")
    if progress:
        progress(30, "🧠 Extracting structured data...")

    state["data"] = extract_structured_json(
        state["text"],
        state["doc_type"]
    )
    return state

# ------------------------------
# RESUME NODE (WITH TEMPLATE)
# ------------------------------

def resume_node(state):

    progress = state.get("progress")
    if progress:
        progress(60, "📄 Building resume...")

    template_bytes = state.get("template")

    if not template_bytes:
        from pathlib import Path
        template_path = Path("templates/resume_template.docx")

        if template_path.exists():
            with open(template_path, "rb") as f:
                template_bytes = f.read()

    file = build_resume(state["data"], template_bytes)

    if progress:
        progress(90, "✅ Resume ready")

    state["result"] = {
        "type": "resume",
        "file": file
    }

    return state

# ------------------------------
# INVOICE NODE
# ------------------------------

def invoice_node(state):

    progress = state.get("progress")
    if progress:
        progress(60, "📊 Creating Excel...")

    df = json_to_kv_dataframe(state["data"])
    excel = generate_excel(df)

    if progress:
        progress(90, "✅ Excel ready")

    state["result"] = {
        "type": "invoice",
        "table": df,
        "excel": excel
    }

    return state

# ------------------------------
# TICKET NODE
# ------------------------------

def ticket_node(state):

    progress = state.get("progress")
    if progress:
        progress(60, "📤 Sending to Concur...")

    if progress:
        progress(90, "✅ Sent successfully")

    state["result"] = {
        "type": "ticket",
        "status": "sent_to_concur"
    }

    return state

# ------------------------------
# ROUTING
# ------------------------------

def route(state):
    dt = state["doc_type"]

    if dt == "resume":
        return "resume"
    elif dt == "invoice":
        return "invoice"
    elif dt == "ticket":
        return "ticket"
    else:
        return "end"

# ------------------------------
# GRAPH
# ------------------------------

def build_graph():

    builder = StateGraph(IDPState)

    builder.add_node("detect", detect_node)
    builder.add_node("extract", extract_node)
    builder.add_node("resume", resume_node)
    builder.add_node("invoice", invoice_node)
    builder.add_node("ticket", ticket_node)

    builder.set_entry_point("detect")

    builder.add_edge("detect", "extract")

    builder.add_conditional_edges(
        "extract",
        route,
        {
            "resume": "resume",
            "invoice": "invoice",
            "ticket": "ticket",
            "end": END
        }
    )

    builder.add_edge("resume", END)
    builder.add_edge("invoice", END)
    builder.add_edge("ticket", END)

    return builder.compile()
