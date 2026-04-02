# ==============================
# INTELLIGENT DOCUMENT PROCESSOR - AUTO MODE ONLY (LangGraph)
# IDP.py
# ==============================

import re
import json
import time
import base64
import tempfile
import hashlib
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

from openai import OpenAI
from docx import Document as DocxDocument
from pptx import Presentation
from streamlit_pdf_viewer import pdf_viewer

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader

from workflow import build_graph
from core import json_to_kv_dataframe, generate_excel

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config("IDP - Professional", layout="wide")

USERS = st.secrets.get("users", {})

# ------------------------------
# CACHED MODELS
# ------------------------------
@st.cache_resource
def get_llm(api_key, model):
    return ChatOpenAI(model=model, temperature=0, api_key=api_key)

@st.cache_resource
def get_embeddings(api_key):
    return OpenAIEmbeddings(api_key=api_key)

# ------------------------------
# AUTH
# ------------------------------
def validate_api_key(api_key):
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return True
    except Exception:
        return False

def login():
    logo_path = Path(__file__).parent / "IDP-Logo1.png"
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if logo_path.exists():
            st.image(logo_path, width=220)

        st.markdown("### 🔐 Sign In")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        api_key = st.text_input("OpenAI API Key", type="password")

        if st.button("Login", use_container_width=True):
            if username not in USERS or USERS[username]["password"] != password:
                st.error("Invalid username or password")
                return

            if not api_key:
                st.error("Please enter your OpenAI API key")
                return

            with st.spinner("Validating API key..."):
                if not validate_api_key(api_key):
                    st.error("Invalid OpenAI API key")
                    return

            st.session_state["logged_in"] = True
            st.session_state["user"] = username
            st.session_state["role"] = USERS[username].get("role", "user")
            st.session_state["api_key"] = api_key
            st.success(f"Welcome {username}")
            st.rerun()

# ------------------------------
# SESSION INIT
# ------------------------------
DEFAULT_METRICS = {
    "tokens": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "cost": 0.0,
    "response_times": [],
    "calls": 0
}

DEFAULT_KEYS = {
    "logged_in": False,
    "user": None,
    "role": None,
    "api_key": None,
    "model_choice": "gpt-4o-mini",
    "resume_template": None,
    "structured_data": None,
    "doc_type": None,
    "vectorstore": None,
    "full_text": None,
    "generated_resume": None,
    "chat_history": [],
    "suggested_questions": [],
    "metrics": DEFAULT_METRICS.copy(),
    "doc_costs": {},
    "doc_metrics": {},
    "auto_result": None,
    "file_hash": None,
    "current_file": None,
    "processing_error": None,
    "elapsed_time": 0.0,
    "run_started_at": None,
    "agent_events": [],
    "agent_logs": [],
    "agent_status": "Idle",
    "current_step": "Waiting for upload",
    "progress_value": 0,
}

for k, v in DEFAULT_KEYS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------------------
# LOGIN GATE
# ------------------------------
if not st.session_state["logged_in"]:
    login()
    st.stop()

if not st.session_state.get("api_key"):
    st.error("API key missing. Please login again.")
    st.stop()

# ------------------------------
# UTILS
# ------------------------------
def reset_document_state():
    st.session_state["structured_data"] = None
    st.session_state["doc_type"] = None
    st.session_state["vectorstore"] = None
    st.session_state["full_text"] = None
    st.session_state["generated_resume"] = None
    st.session_state["chat_history"] = []
    st.session_state["suggested_questions"] = []
    st.session_state["auto_result"] = None
    st.session_state["processing_error"] = None
    st.session_state["elapsed_time"] = 0.0
    st.session_state["run_started_at"] = None
    st.session_state["agent_events"] = []
    st.session_state["agent_logs"] = []
    st.session_state["agent_status"] = "Idle"
    st.session_state["current_step"] = "Waiting for upload"
    st.session_state["progress_value"] = 0

def get_metrics_snapshot():
    m = st.session_state.get("metrics", {})
    return {
        "tokens": m.get("tokens", 0),
        "input_tokens": m.get("input_tokens", 0),
        "output_tokens": m.get("output_tokens", 0),
        "cost": m.get("cost", 0.0),
        "calls": m.get("calls", 0),
    }

def diff_metrics(before, after):
    return {
        "tokens": after.get("tokens", 0) - before.get("tokens", 0),
        "input_tokens": after.get("input_tokens", 0) - before.get("input_tokens", 0),
        "output_tokens": after.get("output_tokens", 0) - before.get("output_tokens", 0),
        "cost": after.get("cost", 0.0) - before.get("cost", 0.0),
        "calls": after.get("calls", 0) - before.get("calls", 0),
    }

def tracked_llm_call(prompt):
    llm = get_llm(
        st.session_state["api_key"],
        st.session_state.get("model_choice", "gpt-4o-mini")
    )

    start = time.time()
    response = llm.invoke(prompt)
    duration = time.time() - start

    try:
        usage = getattr(response, "response_metadata", {}).get("token_usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
    except Exception:
        input_tokens = len(str(prompt)) // 4
        output_tokens = len(str(getattr(response, "content", ""))) // 4

    total_tokens = input_tokens + output_tokens
    input_cost = input_tokens * 0.00015 / 1000
    output_cost = output_tokens * 0.0006 / 1000
    total_cost = input_cost + output_cost

    m = st.session_state.metrics
    m["tokens"] += total_tokens
    m["input_tokens"] += input_tokens
    m["output_tokens"] += output_tokens
    m["cost"] += total_cost
    m["calls"] += 1
    m["response_times"].append(duration)

    doc = st.session_state.get("current_file") or "unknown"
    if doc not in st.session_state.doc_costs:
        st.session_state.doc_costs[doc] = {"cost": 0, "tokens": 0}

    st.session_state.doc_costs[doc]["cost"] += total_cost
    st.session_state.doc_costs[doc]["tokens"] += total_tokens
    return response

def push_agent_log(message):
    st.session_state.agent_logs.append(message)

def record_agent_event(step, status, message="", duration=None, metrics=None):
    st.session_state.agent_events.append({
        "step": step,
        "status": status,
        "message": message,
        "duration": duration,
        "metrics": metrics or {}
    })

def update_progress(percent, message):
    st.session_state["progress_value"] = percent
    st.session_state["current_step"] = message
    st.session_state["agent_status"] = "Running"

def save_temp_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

def extract_docx_text(file_path):
    doc = DocxDocument(file_path)
    parts = []

    for p in doc.paragraphs:
        if p.text and p.text.strip():
            parts.append(p.text.strip())

    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    for section in doc.sections:
        for p in section.header.paragraphs:
            if p.text and p.text.strip():
                parts.append(p.text.strip())
        for p in section.footer.paragraphs:
            if p.text and p.text.strip():
                parts.append(p.text.strip())

    return "\n".join(parts).strip()

def process_file(uploaded_file):
    documents = []
    if not uploaded_file:
        return documents

    uploaded_file.seek(0)
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix in [".png", ".jpg", ".jpeg"]:
        encoded = base64.b64encode(uploaded_file.getvalue()).decode()
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """You are an OCR assistant.

Extract ALL visible text from the image.

Rules:
- Do NOT skip anything
- Preserve numbers, amounts, dates
- Preserve line structure
- If it's a receipt, extract:
  - vendor name
  - date
  - items
  - total
- Output plain text only
"""
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                }
            ]
        )

        try:
            llm = get_llm(
                st.session_state["api_key"],
                st.session_state.get("model_choice", "gpt-4o-mini")
            )
            response = llm.invoke([message])
            content = getattr(response, "content", "") or ""
            if not str(content).strip():
                content = "No readable text found in image"
            documents.append(Document(page_content=str(content)))
        except Exception as e:
            st.error(f"OCR failed: {str(e)}")
            return []

        return documents

    file_path = save_temp_file(uploaded_file)

    try:
        if suffix == ".txt":
            try:
                documents.extend(TextLoader(file_path, encoding="utf-8").load())
            except Exception:
                documents.extend(TextLoader(file_path, encoding="cp1252").load())

        elif suffix == ".pdf":
            docs = PyPDFLoader(file_path).load()
            if docs:
                documents.extend(docs)
            else:
                st.warning("PDF contains no extractable text")

        elif suffix == ".docx":
            text = extract_docx_text(file_path)
            if text.strip():
                documents.append(Document(page_content=text))
            else:
                st.warning("DOCX contains no extractable text")

        elif suffix == ".pptx":
            prs = Presentation(file_path)
            text_parts = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text and shape.text.strip():
                        text_parts.append(shape.text.strip())
            text = "\n".join(text_parts).strip()
            if text:
                documents.append(Document(page_content=text))
            else:
                st.warning("PPTX contains no extractable text")

        elif suffix == ".xlsx":
            excel_file = pd.ExcelFile(file_path)
            sheet_texts = []
            for sheet in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet)
                sheet_texts.append(f"Sheet: {sheet}")
                sheet_texts.append(df.to_string(index=False))
            text = "\n\n".join(sheet_texts).strip()
            if text:
                documents.append(Document(page_content=text))
            else:
                st.warning("Excel contains no extractable text")

        else:
            st.error(f"Unsupported file type: {suffix}")
            return []

    except Exception as e:
        st.error(f"File processing failed: {str(e)}")
        return []

    return documents

def create_vectorstore(docs):
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    if not chunks:
        return None

    for chunk in chunks:
        chunk.metadata = {"source": st.session_state.get("current_file", "unknown")}

    try:
        emb = get_embeddings(st.session_state["api_key"])
        return Chroma.from_documents(chunks, embedding=emb)
    except Exception:
        st.error("🚨 Vectorstore creation failed")
        st.code(traceback.format_exc())
        return None

def get_invoice_filename_from_data(data):
    if not isinstance(data, dict):
        return "invoice_data.xlsx"

    invoice_name = (
        data.get("invoice_number")
        or data.get("invoice_no")
        or data.get("invoice_id")
        or data.get("bill_number")
        or data.get("vendor")
        or data.get("supplier")
        or data.get("name")
        or "invoice_data"
    )

    safe_name = re.sub(r'[\\/*?:"<>|]', "", str(invoice_name)).strip()
    safe_name = safe_name if safe_name else "invoice_data"
    return f"{safe_name}.xlsx"

def get_suggested_questions(doc_type):
    if doc_type == "invoice":
        return [
            "What is the total amount?",
            "Who is the vendor?",
            "What is the invoice date?",
            "List all line items"
        ]
    elif doc_type == "resume":
        return [
            "Summarize this candidate",
            "What skills does the candidate have?",
            "What is the experience?",
            "What is the education background?"
        ]
    elif doc_type == "report":
        return [
            "What is this report about?",
            "Identify the document flow",
            "What are the key points?",
            "What are the main sections?"
        ]
    elif doc_type == "ticket":
        return [
            "What is the ticket number?",
            "What is the travel date?",
            "What is the total amount?",
            "What are the key details?"
        ]
    else:
        return [
            "What is this document?",
            "What are the key points?",
            "Extract important information"
        ]

def normalize_graph_result(result):
    if not isinstance(result, dict):
        return {
            "doc_type": None,
            "structured_data": None,
            "result": {},
            "error": "Graph returned non-dict output"
        }

    doc_type = result.get("doc_type") or result.get("type")
    structured_data = result.get("data") if doc_type in ["invoice", "ticket"] else None
    inner = result.get("result", {})
    if not isinstance(inner, dict):
        inner = {}

    if not inner:
        inner = {
            "type": result.get("result_type") or doc_type,
            "file": result.get("file"),
            "excel": result.get("excel"),
            "table": result.get("table"),
            "payload": result.get("payload"),
            "message": result.get("message"),
            "file_name": result.get("file_name"),
            "data": result.get("data"),
        }

    return {
        "doc_type": doc_type,
        "structured_data": structured_data,
        "result": inner,
        "error": result.get("error"),
        "step_metrics": result.get("step_metrics", [])
    }

def process_uploaded_document(uploaded_file):
    current_file = uploaded_file.name
    st.session_state.current_file = current_file
    st.session_state.run_started_at = time.time()
    st.session_state.agent_status = "Running"
    push_agent_log(f"Upload received: {current_file}")
    record_agent_event("Upload received", "done", f"File: {current_file}")

    docs = process_file(uploaded_file)
    if not docs:
        raise ValueError("Failed to process document")

    record_agent_event("Text extraction", "done", "Uploaded file converted into searchable text")

    full_text = "\n".join(
        [str(d.page_content) for d in docs if d is not None and getattr(d, "page_content", None)]
    ).strip()

    if not full_text:
        raise ValueError("No text extracted from document")

    vectorstore = create_vectorstore(docs)

    graph = build_graph()
    graph_input = {
        "text": full_text,
        "template": st.session_state.get("resume_template"),
        "filename": uploaded_file.name,
        "progress": update_progress,
    }

    before_metrics = get_metrics_snapshot()

    try:
        raw_result = graph.invoke(graph_input)
    except Exception as e:
        raise RuntimeError(f"Auto processing failed: {str(e)}") from e

    after_metrics = get_metrics_snapshot()
    auto_metrics = diff_metrics(before_metrics, after_metrics)

    normalized = normalize_graph_result(raw_result)
    doc_type = normalized.get("doc_type")
    structured_data = normalized.get("structured_data")
    result = normalized.get("result", {})
    error = normalized.get("error")
    step_metrics = normalized.get("step_metrics", [])

    if error:
        raise RuntimeError(str(error))

    if not doc_type:
        raise ValueError("Document type could not be determined by workflow")

    if doc_type not in ["invoice", "ticket"]:
        structured_data = None

    st.session_state.full_text = full_text
    st.session_state.vectorstore = vectorstore
    st.session_state.doc_type = doc_type
    st.session_state.structured_data = structured_data
    st.session_state.auto_result = {
        "doc_type": doc_type,
        "structured_data": structured_data,
        "result": result,
        "metrics": auto_metrics,
        "step_metrics": step_metrics
    }
    st.session_state.suggested_questions = get_suggested_questions(doc_type)

    if doc_type == "resume":
        st.session_state.generated_resume = result.get("file")

    st.session_state.elapsed_time = time.time() - st.session_state.run_started_at
    st.session_state.agent_status = "Done"
    st.session_state.current_step = "Completed"
    st.session_state.progress_value = 100
    push_agent_log(f"Processing complete: {doc_type}")
    record_agent_event("Output ready", "done", f"Detected type: {doc_type}")

# ------------------------------
# CHAT / DETAILS RENDERERS
# ------------------------------
def render_chat_section():
    if st.session_state.get("vectorstore") is None:
        st.caption("Upload and process a document first")
        return

    suggested = st.session_state.get("suggested_questions", [])
    if suggested:
        cols = st.columns(len(suggested))
        for i, q in enumerate(suggested):
            if cols[i].button(q, key=f"detail_suggest_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": q})
                docs = st.session_state.vectorstore.similarity_search(
                    q, k=2, filter={"source": st.session_state.current_file}
                )
                context = "\n\n".join([d.page_content[:800] for d in docs])
                response = tracked_llm_call(
                    f"""
You are a strict document QA assistant.

RULES:
- Answer ONLY from the provided context
- DO NOT add external knowledge
- DO NOT assume anything
- If partially found, return the best matching value
- Keep answer concise and factual
- NO markdown, NO formatting

CONTEXT:
{context}

QUESTION:
{q}
"""
                ).content
                st.session_state.chat_history.append({"role": "assistant", "content": response})

    for msg in st.session_state.get("chat_history", []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask about this document")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        docs = st.session_state.vectorstore.similarity_search(
            query, k=2, filter={"source": st.session_state.current_file}
        )
        context = "\n\n".join([d.page_content for d in docs])
        response = tracked_llm_call(
            f"Answer strictly from context.\nContext:\n{context}\nQ:{query}"
        ).content
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

def render_metrics_section():
    m = st.session_state.get("metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cost", f"${m.get('cost', 0.0):.6f}")
    c2.metric("Total Tokens", m.get("tokens", 0))
    c3.metric("Input Tokens", m.get("input_tokens", 0))
    c4.metric("Output Tokens", m.get("output_tokens", 0))

    if m.get("response_times"):
        df = pd.DataFrame({
            "Call": list(range(len(m["response_times"]))),
            "Response Time": m["response_times"]
        })
        st.line_chart(df.set_index("Call"))

def render_empty_state():
    st.markdown("### Ready to Process")
    st.caption("Upload a file to start agentic processing.")
    st.info("You will see live agent activity on the left and final output here.")

def render_resume_result(result):
    st.success(result.get("message", "Resume generated successfully"))

    file_name = result.get("file_name", "generated_resume.docx")
    file_bytes = result.get("file")
    data = result.get("data", {}) or {}

    top1, top2 = st.columns([2, 1])

    with top1:
        st.caption(f"Output File: {file_name}")

    with top2:
        if file_bytes:
            st.download_button(
                "Download Resume",
                data=file_bytes,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### Candidate")
        st.write(data.get("name", ""))
        st.caption(data.get("email", ""))
        st.caption(data.get("phone", ""))
        st.caption(data.get("location", ""))

        st.markdown("#### Skills")
        skills = data.get("skills", [])
        st.caption(", ".join(skills[:15]) if skills else "No skills found")

    with c2:
        st.markdown("#### Summary")
        st.text_area(
            "Summary",
            value=data.get("summary", ""),
            height=110,
            label_visibility="collapsed"
        )

    st.markdown("#### Experience")
    experience = data.get("experience", [])
    if experience:
        for exp in experience[:2]:
            st.markdown(
                f"**{exp.get('role', '')}** - {exp.get('company', '')}  \n"
                f"{exp.get('start_date', '')} - {exp.get('end_date', '')}"
            )
            for item in exp.get("description", [])[:1]:
                st.caption(f"- {item}")
            st.markdown("---")
    else:
        st.caption("No experience found")
        
def render_invoice_result(result):
    st.success(result.get("message", "Invoice processed successfully"))

    data = st.session_state.get("structured_data", {}) or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vendor", str(data.get("vendor") or data.get("supplier") or "-"))
    c2.metric("Invoice No", str(data.get("invoice_number") or data.get("invoice_no") or "-"))
    c3.metric("Date", str(data.get("invoice_date") or "-"))
    c4.metric("Total", str(data.get("total") or "-"))

    concur_status = result.get("concur_status", "-")
    concur_mode = result.get("concur_mode", "-")
    st.caption(f"Concur Status: {concur_status} | Mode: {concur_mode}")

    excel = result.get("excel")
    if excel:
        st.download_button(
            "Download Excel",
            excel,
            get_invoice_filename_from_data(data),
            use_container_width=False
        )

    table = result.get("table")
    if table is not None:
        st.markdown("#### Extracted Fields")
        st.dataframe(table, use_container_width=True, height=220, hide_index=True)

    if result.get("payload"):
        with st.expander("Concur Payload", expanded=False):
            st.json(result["payload"])
        
def render_ticket_result(result):
    st.success(result.get("message", "Ticket processed successfully"))

    data = st.session_state.get("structured_data", {}) or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Traveler", str(data.get("traveler_name") or "-"))
    c2.metric("Airline", str(data.get("airline") or "-"))
    c3.metric("Route", f"{data.get('from', '-')}" + " → " + f"{data.get('to', '-')}")
    c4.metric("Amount", str(data.get("amount") or "-"))

    concur_status = result.get("concur_status", "-")
    concur_mode = result.get("concur_mode", "-")
    st.caption(f"Concur Status: {concur_status} | Mode: {concur_mode}")

    payload = result.get("payload")
    if payload:
        with st.expander("Concur Payload", expanded=True):
            st.json(payload)
            
def render_generic_result(result):
    st.success(result.get("message", "Processing completed"))
    text = st.session_state.get("full_text", "")
    if text:
        st.markdown("#### Extracted Preview")
        st.text_area(
            "Preview",
            value=text[:2500],
            height=180,
            label_visibility="collapsed"
        )

def render_result_workspace():
    st.markdown("### Result Workspace")

    if st.session_state.get("processing_error"):
        st.error(st.session_state["processing_error"])
        return

    if not st.session_state.get("auto_result"):
        render_empty_state()
        return

    doc_type = st.session_state.get("doc_type")
    result = st.session_state.get("auto_result", {}).get("result", {})

    if doc_type == "resume":
        render_resume_result(result)
    elif doc_type == "invoice":
        render_invoice_result(result)
    elif doc_type == "ticket":
        render_ticket_result(result)
    else:
        render_generic_result(result)

def render_agent_activity_panel():
    st.markdown("### Agent Activity")
    st.info(f"Current Step: {st.session_state.get('current_step', 'Waiting for upload')}")
    st.progress(st.session_state.get("progress_value", 0))

    events = st.session_state.get("agent_events", [])
    if not events:
        st.caption("No activity yet")
        return

    for event in events:
        status = event.get("status", "pending")
        step = event.get("step", "")
        message = event.get("message", "")
        duration = event.get("duration")
        metrics = event.get("metrics", {}) or {}

        if status == "done":
            icon = "✅"
        elif status == "running":
            icon = "🔄"
        elif status == "error":
            icon = "❌"
        else:
            icon = "⏳"

        st.markdown(f"{icon} **{step}**")
        if message:
            st.caption(message)

        meta_parts = []
        if duration is not None:
            meta_parts.append(f"{duration:.2f}s")
        if metrics.get("calls"):
            meta_parts.append(f"calls: {metrics.get('calls')}")
        if metrics.get("cost"):
            meta_parts.append(f"cost: ${metrics.get('cost', 0.0):.6f}")
        if meta_parts:
            st.caption(" | ".join(meta_parts))

    with st.expander("Live Logs", expanded=False):
        logs = st.session_state.get("agent_logs", [])
        if logs:
            for log in logs[-25:]:
                st.write(f"- {log}")
        else:
            st.caption("No logs available")

    with st.expander("Usage", expanded=False):
        auto_metrics = (st.session_state.get("auto_result") or {}).get("metrics", {}) or {}
        c1, c2 = st.columns(2)
        c1.metric("Calls", auto_metrics.get("calls", 0))
        c2.metric("Cost", f"${auto_metrics.get('cost', 0.0):.6f}")

        step_metrics = (st.session_state.get("auto_result") or {}).get("step_metrics", []) or []
        if step_metrics:
            rows = []
            for item in step_metrics:
                rows.append({
                    "Step": item.get("step"),
                    "Cost ($)": round(item.get("metrics", {}).get("cost", 0.0), 6),
                    "Duration (s)": round(item.get("duration", 0.0), 2),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=180, hide_index=True)
            
def render_details_section():
    st.markdown("---")
    st.markdown("### Details")

    with st.expander("Extracted Text", expanded=False):
        full_text = st.session_state.get("full_text")
        if full_text:
            st.text_area("Extracted Text", full_text, height=300, label_visibility="collapsed")
        else:
            st.caption("No extracted text available")

    with st.expander("Structured JSON", expanded=False):
        doc_type = st.session_state.get("doc_type")
        data = st.session_state.get("structured_data")
        if doc_type in ["invoice", "ticket"] and data:
            st.json(data)
        else:
            st.caption("Structured JSON is available only for invoice and ticket")

    with st.expander("Concur Delivery Status", expanded=False):
        auto_result = st.session_state.get("auto_result", {})
        result = auto_result.get("result", {}) if isinstance(auto_result, dict) else {}

        if st.session_state.get("doc_type") in ["invoice", "ticket"]:
            st.write(f"Status: {result.get('concur_status', 'not sent')}")
            st.write(f"Mode: {result.get('concur_mode', '-')}")
            if result.get("payload"):
                st.json(result["payload"])
        else:
            st.caption("Concur delivery applies only to invoice and ticket")

    with st.expander("Document Chat", expanded=False):
        render_chat_section()

    with st.expander("Metrics", expanded=False):
        render_metrics_section()
# ------------------------------
# HEADER / CONTROLS
# ------------------------------
def render_header():
    logo_path = Path(__file__).parent / "IDP-Logo1.png"
    c0, c1, c2, c3 = st.columns([1.2, 4, 1.2, 1.2])

    with c0:
        if logo_path.exists():
            st.image(logo_path, width=140)

    with c1:
        st.markdown("## Intelligent Document Processor")
        st.caption("AI-powered document understanding & automation")
        st.write(f"File: {st.session_state.get('current_file') or 'No file uploaded'}")

    with c2:
        st.metric("Status", st.session_state.get("agent_status", "Idle"))

    with c3:
        st.metric("Elapsed", f"{st.session_state.get('elapsed_time', 0.0):.1f}s")

def render_upload_controls():
    with st.sidebar:
        st.markdown("### 👤 User Info")
        st.write(f"**User:** {st.session_state['user']}")
        st.write(f"**Role:** {st.session_state['role']}")
        st.success("🔑 API key loaded securely")

        st.markdown("### 🤖 Model")
        model_choice = st.selectbox(
            "Choose Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-5"],
            index=["gpt-4o-mini", "gpt-4o", "gpt-5"].index(
                st.session_state.get("model_choice", "gpt-4o-mini")
            )
        )
        st.session_state["model_choice"] = model_choice

        if st.button("🚪 Logout"):
            for key in ["logged_in", "user", "role", "api_key"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Logged out")
            st.rerun()

        st.markdown("### ⚙️ Processing Mode")
        st.success("Auto (LangGraph)")

        template_file = st.file_uploader(
            "Resume Template (optional)",
            type=["docx"],
            key="sidebar_resume_template"
        )
        if template_file:
            st.session_state["resume_template"] = template_file.getvalue()

        st.markdown("---")
        cost = st.session_state.get("metrics", {}).get("cost", 0)
        st.write(f"Session Cost 💰 ${round(cost, 6)}")

    c1, c2 = st.columns([5, 1])

    with c1:
        uploaded_file = st.file_uploader(
            "Upload document",
            type=["txt", "pdf", "docx", "pptx", "xlsx", "png", "jpg", "jpeg"],
            key="main_file_uploader"
        )

    with c2:
        st.write("")
        st.write("")
        if st.button("Reset", use_container_width=True):
            reset_document_state()
            st.rerun()

    return uploaded_file
    
# ------------------------------
# MAIN
# ------------------------------
render_header()
uploaded_file = render_upload_controls()

if uploaded_file:
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

    if st.session_state.get("file_hash") != file_hash:
        reset_document_state()
        st.session_state.file_hash = file_hash

        try:
            process_uploaded_document(uploaded_file)
            st.success(f"✅ Processed Successfully | Type: {st.session_state.doc_type.upper()}")
        except Exception as e:
            st.session_state.processing_error = str(e)
            st.session_state.agent_status = "Failed"
            st.session_state.current_step = "Failed"
            st.session_state.elapsed_time = (
                time.time() - st.session_state.run_started_at
                if st.session_state.run_started_at else 0.0
            )
            push_agent_log(f"Error: {str(e)}")
            record_agent_event("Processing failed", "error", str(e))
            st.error(f"❌ {str(e)}")
            st.code(traceback.format_exc())
    else:
        if st.session_state.get("doc_type"):
            st.success(f"✅ Processed Successfully | Type: {st.session_state.doc_type.upper()}")

left_col, right_col = st.columns([1, 1.6], gap="large")

with left_col:
    render_agent_activity_panel()

with right_col:
    render_result_workspace()

render_details_section()
