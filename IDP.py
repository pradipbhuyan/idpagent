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
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)

from workflow import build_graph
from core import (
    json_to_kv_dataframe,
    generate_excel,
)

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config("IDP - Professional", layout="wide")

# ------------------------------
# USERS
# ------------------------------
USERS = st.secrets.get("users", {})

# ------------------------------
# CACHED MODELS
# ------------------------------
@st.cache_resource
def get_llm(api_key, model):
    return ChatOpenAI(
        model=model,
        temperature=0,
        api_key=api_key
    )

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
    "active_tab": "Preview",
    "suggested_questions": [],
    "metrics": DEFAULT_METRICS.copy(),
    "doc_costs": {},
    "doc_metrics": {},
    "auto_result": None,
    "file_hash": None,
    "current_file": None,
    "processing_error": None,
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


def update_progress(percent, message):
    if "progress_bar" not in st.session_state:
        st.session_state.progress_bar = st.progress(0)

    st.session_state.progress_bar.progress(percent, text=message)


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
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded}"
                    }
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)
    if not chunks:
        return None

    for chunk in chunks:
        chunk.metadata = {
            "source": st.session_state.get("current_file", "unknown")
        }

    try:
        emb = get_embeddings(st.session_state["api_key"])
        vectorstore = Chroma.from_documents(chunks, embedding=emb)
        return vectorstore
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
    }


def process_uploaded_document(uploaded_file):
    current_file = uploaded_file.name
    st.session_state.current_file = current_file

    if current_file not in st.session_state.doc_metrics:
        st.session_state.doc_metrics[current_file] = {
            "tokens": 0,
            "response_times": [],
            "calls": 0
        }

    docs = process_file(uploaded_file)
    if not docs:
        raise ValueError("Failed to process document")

    full_text = "\n".join(
        [
            str(d.page_content)
            for d in docs
            if d is not None and getattr(d, "page_content", None)
        ]
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

    try:
        raw_result = graph.invoke(graph_input)
    except Exception as e:
        raise RuntimeError(f"Auto processing failed: {str(e)}") from e

    normalized = normalize_graph_result(raw_result)

    doc_type = normalized.get("doc_type")
    structured_data = normalized.get("structured_data")
    result = normalized.get("result", {})
    error = normalized.get("error")

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
        "result": result
    }
    st.session_state.suggested_questions = get_suggested_questions(doc_type)

    if doc_type == "resume":
        st.session_state.generated_resume = result.get("file")

    return st.session_state.auto_result

# ------------------------------
# SIDEBAR
# ------------------------------
with st.sidebar:
    st.markdown("### 👤 User Info")
    st.write(f"**User:** {st.session_state['user']}")
    st.write(f"**Role:** {st.session_state['role']}")
    st.success("🔑 API key loaded securely")

    st.markdown("### 🤖 Model")
    model_choice = st.selectbox(
        "Choose Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-5"],
        index=["gpt-4o-mini", "gpt-4o", "gpt-5"].index(st.session_state.get("model_choice", "gpt-4o-mini"))
    )
    st.session_state["model_choice"] = model_choice
    st.caption(f"Using: {st.session_state['model_choice']}")

    if st.button("🚪 Logout"):
        for key in ["logged_in", "user", "role", "api_key"]:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Logged out")
        st.rerun()

    st.markdown("### ⚙️ Processing Mode")
    st.success("Auto (LangGraph)")

    template_file = st.file_uploader(
        "Upload Resume Template (Optional)",
        type=["docx"],
        key="sidebar_resume_template"
    )
    if template_file:
        st.session_state["resume_template"] = template_file.getvalue()

    st.markdown("---")
    cost = st.session_state.get("metrics", {}).get("cost", 0)
    st.write(f"Session Cost 💰 ${round(cost, 6)}")

# ------------------------------
# HEADER
# ------------------------------
logo_path = Path(__file__).parent / "IDP-Logo1.png"
col1, col2 = st.columns([1, 7], gap="small")

with col1:
    if logo_path.exists():
        st.image(logo_path, width=280)

with col2:
    st.markdown("## Intelligent Document Processor")
    st.caption("AI-powered document understanding & automation")

# ------------------------------
# FILE UPLOAD
# ------------------------------
uploaded_file = st.file_uploader(
    "Drag and drop file here",
    type=["txt", "pdf", "docx", "pptx", "xlsx", "png", "jpg", "jpeg"]
)

if uploaded_file:
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

    if st.session_state.get("file_hash") != file_hash:
        reset_document_state()
        st.session_state.file_hash = file_hash

        progress = st.progress(0, text="Processing started...")

        try:
            progress.progress(15, text="Reading uploaded document...")
            process_uploaded_document(uploaded_file)
            progress.progress(100, text="Processing completed")
            st.success(f"✅ Processed Successfully | Type: {st.session_state.doc_type.upper()}")

        except Exception as e:
            st.session_state.processing_error = str(e)
            progress.empty()
            st.error(f"❌ {str(e)}")
            st.code(traceback.format_exc())
    else:
        if st.session_state.get("doc_type"):
            st.success(f"✅ Processed Successfully | Type: {st.session_state.doc_type.upper()}")

# ------------------------------
# TABS
# ------------------------------
tabs = ["Preview", "JSON", "Download", "Concur", "Chat", "Auto", "Metrics"]

selected_tab = st.radio(
    "",
    tabs,
    horizontal=True,
    key="active_tab"
)

# ------------------------------
# PREVIEW
# ------------------------------
if selected_tab == "Preview":
    if uploaded_file:
        if "pdf" in uploaded_file.type:
            pdf_viewer(uploaded_file.getvalue(), height=200)

        elif "image" in uploaded_file.type:
            st.image(uploaded_file, width=300)

        elif "text" in uploaded_file.type:
            try:
                preview_text = uploaded_file.getvalue().decode("utf-8")
            except Exception:
                preview_text = uploaded_file.getvalue().decode("cp1252", errors="ignore")
            st.text_area("Preview", preview_text, height=200)

        elif "word" in uploaded_file.type or uploaded_file.name.endswith(".docx"):
            path = save_temp_file(uploaded_file)
            text = extract_docx_text(path)
            st.text_area("DOCX Preview", text, height=250)

        elif uploaded_file.name.endswith(".pptx"):
            path = save_temp_file(uploaded_file)
            prs = Presentation(path)
            slides_text = []

            for i, slide in enumerate(prs.slides):
                slide_text = [f"Slide {i + 1}"]
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text and shape.text.strip():
                        slide_text.append(shape.text.strip())
                slides_text.append("\n".join(slide_text))

            st.text_area("PPTX Preview", "\n\n".join(slides_text), height=300)

        elif uploaded_file.name.endswith(".xlsx"):
            path = save_temp_file(uploaded_file)
            df = pd.read_excel(path)
            st.dataframe(df)
            st.text_area("Excel Preview (Text)", df.to_string(index=False), height=200)

# ------------------------------
# JSON
# ------------------------------
if selected_tab == "JSON":
    doc_type = st.session_state.get("doc_type")
    data = st.session_state.get("structured_data")

    if not uploaded_file:
        st.info("Upload and process a document first")
    elif doc_type in ["invoice", "ticket"] and data:
        st.json(data)
    elif doc_type in ["invoice", "ticket"]:
        st.warning("No structured JSON available")
    else:
        st.info("JSON view is only available for invoice and ticket documents")

# ------------------------------
# CHAT
# ------------------------------
if selected_tab == "Chat":
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process a document first")
    else:
        if st.session_state.suggested_questions:
            st.markdown("### 💡 Suggested Questions")
            cols = st.columns(len(st.session_state.suggested_questions))

            for i, q in enumerate(st.session_state.suggested_questions):
                if cols[i].button(q, key=f"suggest_{i}"):
                    st.session_state.chat_history.append(
                        {"role": "user", "content": q}
                    )

                    docs = st.session_state.vectorstore.similarity_search(
                        q,
                        k=2,
                        filter={"source": st.session_state.current_file}
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

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                    st.text(response)

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        query = st.chat_input("Ask a question")

        if query:
            st.session_state.chat_history.append(
                {"role": "user", "content": query}
            )

            docs = st.session_state.vectorstore.similarity_search(
                query,
                k=2,
                filter={"source": st.session_state.current_file}
            )
            context = "\n\n".join([d.page_content for d in docs])

            response = tracked_llm_call(
                f"Answer strictly from context.\nContext:\n{context}\nQ:{query}"
            ).content

            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )
            st.write(response)

# ------------------------------
# DOWNLOAD
# ------------------------------
if selected_tab == "Download":
    if not uploaded_file:
        st.info("Upload and process a document first")
    elif st.session_state.get("processing_error"):
        st.error(st.session_state.processing_error)
    else:
        doc_type = st.session_state.get("doc_type")
        auto_result = st.session_state.get("auto_result") or {}
        result = auto_result.get("result", {}) if isinstance(auto_result, dict) else {}
        data = st.session_state.get("structured_data")

        if doc_type == "resume":
            resume_file = result.get("file") or st.session_state.get("generated_resume")

            if resume_file:
                file_name = result.get("file_name") or "generated_resume.docx"
                st.caption(f"📄 {file_name}")
                st.download_button(
                    "Download Resume",
                    data=resume_file,
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            else:
                st.error("Resume file missing from auto workflow output")

        elif doc_type == "invoice":
            if data:
                st.download_button(
                    "Download JSON",
                    json.dumps(data, indent=2),
                    "data.json",
                    mime="application/json"
                )

                df = None
                try:
                    df = result.get("table")
                    if df is None:
                        df = json_to_kv_dataframe(data)
                except Exception:
                    df = json_to_kv_dataframe(data)

                if isinstance(df, pd.DataFrame):
                    st.dataframe(df, use_container_width=True)
                    excel = result.get("excel") or generate_excel(df)

                    file_name = get_invoice_filename_from_data(data)
                    st.caption(f"📄 {file_name}")

                    st.download_button(
                        "Download Excel",
                        excel,
                        file_name
                    )
                else:
                    st.warning("Invoice table is not available")
            else:
                st.warning("No invoice JSON available")

        elif doc_type == "ticket":
            if data:
                st.download_button(
                    "Download JSON",
                    json.dumps(data, indent=2),
                    "data.json",
                    mime="application/json"
                )
                st.success("Ticket JSON ready for download")
            else:
                st.warning("No ticket JSON available")

        else:
            st.info("No downloadable structured output available for this document type")

# ------------------------------
# CONCUR
# ------------------------------
if selected_tab == "Concur":
    st.subheader("Send to Concur Integration")

    supported_types = ["invoice", "ticket"]
    mode = st.radio("Mode", ["Mock", "Real (Simulated OAuth)"], horizontal=True)

    if st.session_state.doc_type in supported_types:
        st.info(f"Document Type Supported: {st.session_state.doc_type.upper()}")

        if mode == "Real (Simulated OAuth)":
            if st.button("Authenticate with Concur"):
                st.session_state.concur_token = "mock_token"
                st.success("Authenticated")

        if st.button("Send to Concur"):
            progress = st.progress(0, text="Preparing payload...")

            payload = None
            auto_result = st.session_state.get("auto_result") or {}
            result = auto_result.get("result", {}) if isinstance(auto_result, dict) else {}

            if result.get("payload"):
                payload = result.get("payload")
            else:
                payload = {
                    "type": st.session_state.doc_type,
                    "data": st.session_state.structured_data,
                    "line_items": (
                        json_to_kv_dataframe(st.session_state.structured_data).to_dict(orient="records")
                        if st.session_state.structured_data
                        else []
                    )
                }

            progress.progress(40, text="Connecting...")
            time.sleep(1)

            progress.progress(70, text="Sending...")
            time.sleep(1)

            if mode == "Mock":
                st.success("✅ Sent (Mock)")
            else:
                if "concur_token" not in st.session_state:
                    st.error("Authenticate first")
                    progress.empty()
                    st.stop()
                st.success("✅ Sent to API")

            progress.progress(100, text="Completed")
            progress.empty()
            st.json(payload)
    else:
        st.warning("Only Invoice or Ticket supported")

# ------------------------------
# METRICS
# ------------------------------
if selected_tab == "Metrics":
    st.subheader("📊 Cost & Usage Analytics")

    m = st.session_state.metrics

    if m["calls"] == 0:
        st.warning("No usage yet")
        st.stop()

    avg_time = sum(m["response_times"]) / m["calls"] if m["calls"] else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Cost ($)", round(m["cost"], 6))
    col2.metric("🧠 Total Tokens", m["tokens"])
    col3.metric("⚡ LLM Calls", m["calls"])
    col4.metric("⏱ Avg Time (s)", round(avg_time, 2))

    st.markdown("---")
    st.subheader("🔎 Token Breakdown")
    st.write(f"Input Tokens: {m['input_tokens']}")
    st.write(f"Output Tokens: {m['output_tokens']}")

    if m["response_times"]:
        df = pd.DataFrame({
            "Call": list(range(len(m["response_times"]))),
            "Response Time": m["response_times"]
        })
        st.line_chart(df.set_index("Call"))

    st.subheader("📄 Cost per Document")
    doc_data = [
        {"Document": k, "Cost": v["cost"], "Tokens": v["tokens"]}
        for k, v in st.session_state.doc_costs.items()
    ]

    doc_df = pd.DataFrame(doc_data)
    if not doc_df.empty:
        doc_df.insert(0, "SL No", range(1, len(doc_df) + 1))
        st.dataframe(doc_df, use_container_width=True, hide_index=True)
        st.bar_chart(doc_df.set_index("Document")[["Cost"]])
    else:
        st.info("No document-level cost recorded yet")

    if m["cost"] > 0.05:
        st.warning("⚠️ High usage detected (>$0.05)")

# ------------------------------
# AUTO OUTPUT TAB
# ------------------------------
if selected_tab == "Auto":
    if not uploaded_file:
        st.info("Upload a document to use Auto Mode")
    elif st.session_state.get("processing_error"):
        st.error(st.session_state.processing_error)
    elif not st.session_state.get("auto_result"):
        st.warning("No auto result available")
    else:
        st.subheader("🤖 Auto Processing Output")

        auto_result = st.session_state.auto_result
        result = auto_result.get("result", {})
        doc_type = auto_result.get("doc_type", "")

        st.write(f"📄 Detected Type: **{doc_type.upper()}**")

        if doc_type == "resume":
            file = result.get("file") or st.session_state.get("generated_resume")

            if file:
                file_name = result.get("file_name") or "generated_resume.docx"
                st.success("✅ Resume generated successfully")
                st.caption(f"📄 {file_name}")

                st.download_button(
                    label="⬇️ Download Resume",
                    data=file,
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            else:
                st.error("Resume file missing from workflow output")

        elif doc_type == "invoice":
            st.success("✅ Invoice processed")

            table = result.get("table")
            if isinstance(table, pd.DataFrame):
                st.dataframe(table, use_container_width=True)
            elif st.session_state.get("structured_data"):
                try:
                    fallback_df = json_to_kv_dataframe(st.session_state.structured_data)
                    st.dataframe(fallback_df, use_container_width=True)
                except Exception:
                    st.info("Invoice table not available")

            excel = result.get("excel")
            if excel:
                st.download_button(
                    "⬇️ Download Excel",
                    excel,
                    get_invoice_filename_from_data(st.session_state.get("structured_data") or {})
                )

        elif doc_type == "ticket":
            st.success("✅ Ticket processed")
            if st.session_state.get("structured_data"):
                st.json(st.session_state.structured_data)
            elif result.get("message"):
                st.info(result.get("message"))

        else:
            msg = result.get("message") or "Processing completed"
            st.success(f"✅ {msg}")
