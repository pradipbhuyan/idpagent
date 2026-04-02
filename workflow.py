from typing import TypedDict, Any, Optional
from pathlib import Path
import time

from langgraph.graph import StateGraph, END

from core import (
    detect_document_type,
    extract_structured_json,
    build_resume,
    json_to_kv_dataframe,
    generate_excel,
    send_to_concur,
    get_current_metrics_snapshot,
    diff_metrics_snapshot,
)

class IDPState(TypedDict, total=False):
    text: str
    template: Optional[bytes]
    filename: str
    progress: Any

    doc_type: str
    data: dict
    result: dict
    error: str
    step_metrics: list

def safe_progress(state, percent, message):
    progress = state.get("progress")
    if progress:
        progress(percent, message)

def add_step_metric(state, step, started_at, before_metrics, message=""):
    after_metrics = get_current_metrics_snapshot()
    delta = diff_metrics_snapshot(before_metrics, after_metrics)
    if "step_metrics" not in state or state["step_metrics"] is None:
        state["step_metrics"] = []
    state["step_metrics"].append({
        "step": step,
        "message": message,
        "duration": time.time() - started_at,
        "metrics": delta
    })

def get_resume_filename_from_data(data: dict) -> str:
    if not isinstance(data, dict):
        return "candidate.docx"

    name = (
        data.get("name")
        or data.get("Name")
        or data.get("candidate_name")
        or (
            data.get("personal_details", {}).get("name")
            if isinstance(data.get("personal_details"), dict)
            else None
        )
        or "candidate"
    )

    import re
    safe_name = re.sub(r'[\\/*?:"<>|]', "", str(name)).strip()
    safe_name = safe_name if safe_name else "candidate"
    return f"{safe_name}.docx"

def detect_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()
    safe_progress(state, 10, "Detecting document type")

    text = (state.get("text") or "").strip()
    if not text:
        state["error"] = "No extracted text available for processing"
        state["doc_type"] = "other"
        add_step_metric(state, "Detect document type", started_at, before, "No text found")
        return state

    try:
        state["doc_type"] = detect_document_type(text)
        add_step_metric(state, "Detect document type", started_at, before, f"Detected {state['doc_type']}")
    except Exception as e:
        state["error"] = f"Document type detection failed: {str(e)}"
        state["doc_type"] = "other"
        add_step_metric(state, "Detect document type", started_at, before, str(e))

    return state

def resume_extract_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()
    safe_progress(state, 35, "Extracting resume details")

    try:
        state["data"] = extract_structured_json(state["text"], "resume")
        add_step_metric(state, "Extract resume data", started_at, before, "Resume fields extracted")
    except Exception as e:
        state["error"] = f"Resume extraction failed: {str(e)}"
        state["data"] = {}
        add_step_metric(state, "Extract resume data", started_at, before, str(e))

    return state

def resume_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()
    safe_progress(state, 65, "Building resume")

    data = state.get("data") or {}
    template_bytes = state.get("template")

    if not template_bytes:
        possible_paths = [
            Path("templates/resume_template.docx"),
            Path("templates:resume_template.docx"),
            Path(__file__).parent / "templates" / "resume_template.docx",
            Path(__file__).parent / "templates:resume_template.docx",
        ]
        for template_path in possible_paths:
            if template_path.exists():
                with open(template_path, "rb") as f:
                    template_bytes = f.read()
                break

    if not template_bytes:
        state["error"] = "No resume template provided and default template not found"
        state["result"] = {
            "type": "resume",
            "file": None,
            "data": data,
            "file_name": "candidate.docx",
            "message": "Resume template missing"
        }
        add_step_metric(state, "Build resume", started_at, before, "Template missing")
        return state

    try:
        file_bytes = build_resume(data, template_bytes)
        file_name = get_resume_filename_from_data(data)

        safe_progress(state, 95, "Resume ready")
        state["result"] = {
            "type": "resume",
            "file": file_bytes,
            "data": data,
            "file_name": file_name,
            "message": "Resume generated successfully"
        }
        add_step_metric(state, "Build resume", started_at, before, "Resume file created")
    except Exception as e:
        state["error"] = f"Resume generation failed: {str(e)}"
        state["result"] = {
            "type": "resume",
            "file": None,
            "data": data,
            "file_name": "candidate.docx",
            "message": str(e)
        }
        add_step_metric(state, "Build resume", started_at, before, str(e))

    return state

def extract_json_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()
    safe_progress(state, 35, "Extracting structured JSON")

    doc_type = state.get("doc_type", "other")
    try:
        state["data"] = extract_structured_json(state["text"], doc_type)
        add_step_metric(state, f"Extract {doc_type} JSON", started_at, before, "Structured fields extracted")
    except Exception as e:
        state["error"] = f"Structured extraction failed: {str(e)}"
        state["data"] = {}
        add_step_metric(state, f"Extract {doc_type} JSON", started_at, before, str(e))

    return state

def invoice_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()
    safe_progress(state, 70, "Creating invoice output")

    data = state.get("data") or {}

    try:
        df = json_to_kv_dataframe(data)
        excel = generate_excel(df)

        safe_progress(state, 85, "Sending invoice to Concur")
        concur_result = send_to_concur("invoice", data, mode="mock")

        safe_progress(state, 95, "Invoice sent to Concur")

        state["result"] = {
            "type": "invoice",
            "table": df,
            "excel": excel,
            "data": data,
            "payload": concur_result.get("payload"),
            "concur_status": concur_result.get("status"),
            "concur_mode": concur_result.get("mode"),
            "concur_submission_id": concur_result.get("submission_id"),
            "concur_batch_id": concur_result.get("batch_id"),
            "concur_document_id": concur_result.get("document_id"),
            "concur_submitted_at": concur_result.get("submitted_at"),
            "concur_endpoint": concur_result.get("endpoint"),
            "concur_processing_state": concur_result.get("processing_state"),
            "concur_next_status": concur_result.get("next_status"),
            "message": concur_result.get("message", "Invoice processed successfully")
        }

        add_step_metric(state, "Create invoice output + send to Concur", started_at, before, "Invoice sent")
    except Exception as e:
        state["error"] = f"Invoice processing failed: {str(e)}"
        state["result"] = {
            "type": "invoice",
            "table": None,
            "excel": None,
            "data": data,
            "concur_status": "error",
            "concur_mode": "mock",
            "message": str(e)
        }
        add_step_metric(state, "Create invoice output + send to Concur", started_at, before, str(e))

    return state
    
def ticket_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()
    safe_progress(state, 70, "Preparing ticket payload")

    data = state.get("data") or {}

    try:
        safe_progress(state, 85, "Sending ticket to Concur")
        concur_result = send_to_concur("ticket", data, mode="mock")

        safe_progress(state, 95, "Ticket sent to Concur")

        state["result"] = {
            "type": "ticket",
            "status": "sent",
            "data": data,
            "payload": concur_result.get("payload"),
            "concur_status": concur_result.get("status"),
            "concur_mode": concur_result.get("mode"),
            "concur_submission_id": concur_result.get("submission_id"),
            "concur_batch_id": concur_result.get("batch_id"),
            "concur_document_id": concur_result.get("document_id"),
            "concur_submitted_at": concur_result.get("submitted_at"),
            "concur_endpoint": concur_result.get("endpoint"),
            "concur_processing_state": concur_result.get("processing_state"),
            "concur_next_status": concur_result.get("next_status"),
            "message": concur_result.get("message", "Ticket processed successfully")
        }

        add_step_metric(state, "Create ticket output + send to Concur", started_at, before, "Ticket sent")
    except Exception as e:
        state["error"] = f"Ticket processing failed: {str(e)}"
        state["result"] = {
            "type": "ticket",
            "status": "error",
            "data": data,
            "concur_status": "error",
            "concur_mode": "mock",
            "message": str(e)
        }
        add_step_metric(state, "Create ticket output + send to Concur", started_at, before, str(e))

    return state
    
def other_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()
    safe_progress(state, 80, "No structured extraction required")

    state["data"] = {}
    state["result"] = {
        "type": state.get("doc_type", "other"),
        "message": f"No structured output configured for document type: {state.get('doc_type', 'other')}"
    }
    add_step_metric(state, "Finalize generic output", started_at, before, "No structured processing needed")
    return state

def route_after_detect(state: IDPState) -> str:
    dt = state.get("doc_type", "other")
    if dt == "resume":
        return "resume_extract"
    elif dt in ["invoice", "ticket"]:
        return "extract_json"
    else:
        return "other"

def route_after_extract_json(state: IDPState) -> str:
    dt = state.get("doc_type", "other")
    if dt == "invoice":
        return "invoice"
    elif dt == "ticket":
        return "ticket"
    else:
        return "other"

def build_graph():
    builder = StateGraph(IDPState)

    builder.add_node("detect", detect_node)
    builder.add_node("resume_extract", resume_extract_node)
    builder.add_node("resume", resume_node)
    builder.add_node("extract_json", extract_json_node)
    builder.add_node("invoice", invoice_node)
    builder.add_node("ticket", ticket_node)
    builder.add_node("other", other_node)

    builder.set_entry_point("detect")

    builder.add_conditional_edges(
        "detect",
        route_after_detect,
        {
            "resume_extract": "resume_extract",
            "extract_json": "extract_json",
            "other": "other",
        }
    )

    builder.add_edge("resume_extract", "resume")

    builder.add_conditional_edges(
        "extract_json",
        route_after_extract_json,
        {
            "invoice": "invoice",
            "ticket": "ticket",
            "other": "other",
        }
    )

    builder.add_edge("resume", END)
    builder.add_edge("invoice", END)
    builder.add_edge("ticket", END)
    builder.add_edge("other", END)

    return builder.compile()
