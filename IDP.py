
# ==============================
# INTELLIGENT DOCUMENT PROCESSOR - with Agentic Mode
# ==============================

import os
import base64
import tempfile
import json
import re
from pathlib import Path
from io import BytesIO

import streamlit as st
import pandas as pd

import os
import streamlit as st

#os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
)

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from docx import Document as DocxDocument
from streamlit_pdf_viewer import pdf_viewer

from workflow import build_graph

from core import (
    detect_document_type,
    extract_structured_json,
    build_resume,
    json_to_kv_dataframe,
    generate_excel
)
# ------------------------------
# LLM & EMBEDDINGS
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
# INIT
# ------------------------------

st.set_page_config("IDP - Professional", layout="wide")

# ------------------------------
# LOGIN + API KEY VALIDATION
# ------------------------------

import streamlit as st
from pathlib import Path
from openai import OpenAI

USERS = st.secrets.get("users", {})

# ------------------------------
# VALIDATE API KEY
# ------------------------------
def validate_api_key(api_key):
    try:
        client = OpenAI(api_key=api_key)

        # Lightweight test call
        client.models.list()

        return True
    except Exception:
        return False


# ------------------------------
# LOGIN FUNCTION
# ------------------------------
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

            # Validate user
            if username not in USERS or USERS[username]["password"] != password:
                st.error("Invalid username or password")
                return

            if not api_key:
                st.error("Please enter your OpenAI API key")
                return

            # 🔑 Validate API key
            with st.spinner("Validating API key..."):
                if not validate_api_key(api_key):
                    st.error("Invalid OpenAI API key")
                    return

            # Save session
            st.session_state["logged_in"] = True
            st.session_state["user"] = username
            st.session_state["role"] = USERS[username].get("role", "user")
            st.session_state["api_key"] = api_key

            st.success(f"Welcome {username}")

            st.rerun()

# ------------------------------
# SESSION INIT
# ------------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "user" not in st.session_state:
    st.session_state["user"] = None

if "role" not in st.session_state:
    st.session_state["role"] = None

if "api_key" not in st.session_state:
    st.session_state["api_key"] = None


# ------------------------------
# LOGIN GATE
# ------------------------------
if not st.session_state["logged_in"]:
    login()
    st.stop()
    
# ------------------------------
# API KEY SAFETY 
# ------------------------------
if not st.session_state.get("api_key"):
    st.error("API key missing. Please login again.")
    st.stop()

# ------------------------------
# SIDEBAR (USER INFO + LOGOUT)
# ------------------------------
with st.sidebar:
    st.markdown("### 👤 User Info")
    st.write(f"**User:** {st.session_state['user']}")
    st.write(f"**Role:** {st.session_state['role']}")

    st.success("🔑 API key loaded securely")

    # 🤖 Model Selection
    st.markdown("### 🤖 Model")
    
    model_choice = st.selectbox(
        "Choose Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-5"],
        index=0
    )
    
    # Store in session
    st.session_state["model_choice"] = model_choice

    st.caption(f"Using: {st.session_state['model_choice']}")
    
    if st.button("🚪 Logout"):
        for key in ["logged_in", "user", "role", "api_key"]:
            if key in st.session_state:
                del st.session_state[key]

        st.success("Logged out")
        st.rerun()

    st.markdown("### ⚙️ Mode")

    mode = st.radio(
        "Processing Mode",
        ["Manual", "Auto (LangGraph)"],
        horizontal=True
    )

    st.session_state["mode"] = mode

    template_file = st.file_uploader(
        "Upload Resume Template (Optional)",
        type=["docx"]
    )
    
    if template_file:
        st.session_state["resume_template"] = template_file.getvalue()
    
    st.markdown("---")

    # 💰 Cost
    cost = st.session_state.get("metrics", {}).get("cost", 0)
    st.write(f"Session Cost 💰 ${round(cost, 6)}")
    

logo_path = Path(__file__).parent / "IDP-Logo1.png"

col1, col2 = st.columns([1, 7], gap="small")

with col1:
    st.image(logo_path, width=280)

with col2:
    st.markdown("## Intelligent Document Processor")
    st.caption("AI-powered document understanding & automation")

# Session state
for key in ["structured_data", "doc_type", "vectorstore", "full_text"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "generated_resume" not in st.session_state:
    st.session_state.generated_resume = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": 0.0,
        "response_times": [],
        "calls": 0
    }

if "doc_costs" not in st.session_state:
    st.session_state.doc_costs = {}

if "doc_metrics" not in st.session_state:
    st.session_state.doc_metrics = {}

# ------------------------------
# FILE UPLOAD
# ------------------------------

uploaded_file = st.file_uploader(
    "Drag and drop file here",
    type=["txt", "pdf", "docx", "pptx", "xlsx", "png", "jpg", "jpeg"]
)

# ------------------------------
# HELPERS
# ------------------------------

def save_temp_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def load_docx_safe(file_path):
    doc = DocxDocument(file_path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return [Document(page_content=text)]


def process_file(uploaded_file):
    documents = []

    if not uploaded_file:
        return documents

    # ✅ Reset pointer (important)
    uploaded_file.seek(0)

    suffix = Path(uploaded_file.name).suffix.lower()

    # ------------------------------
    # 🖼️ IMAGE OCR
    # ------------------------------
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

            if not content.strip():
                content = "No readable text found in image"

        except Exception as e:
            st.error(f"OCR failed: {str(e)}")
            content = "OCR failed"

        documents.append(Document(page_content=str(content)))

    # ------------------------------
    # 📄 OTHER FILE TYPES
    # ------------------------------
    else:
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
                documents.extend(load_docx_safe(file_path))

            elif suffix == ".pptx":
                from pptx import Presentation
                prs = Presentation(file_path)

                text = []
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text") and shape.text.strip():
                            text.append(shape.text)

                documents.append(Document(page_content="\n".join(text)))

            elif suffix == ".xlsx":
                df = pd.read_excel(file_path)
                text = df.to_string(index=False)
                documents.append(Document(page_content=text))

        except Exception as e:
            st.error(f"File processing failed: {str(e)}")

    return documents


def safe_json_parse(response):
    try:
        return json.loads(response)
    except:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return {"error": "Invalid JSON output", "raw_response": response[:500]}


def generate_resume_summary(data):
    prompt = f"""
Create a professional resume summary.
Write candidate name at the top.
Write education, certification and expereince in concise bullet points.
STRICT RULES:
- No markdown
- No ** or *
- Plain text only
{json.dumps(data)}
"""
    return tracked_llm_call(prompt).content

    
def create_vectorstore(docs):

    if not docs:
        st.error("❌ No documents to index")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        st.error("❌ No chunks created")
        return None

    for chunk in chunks:
        chunk.metadata = {
            "source": st.session_state.get("current_file", "unknown")
        }

    try:
        emb = get_embeddings(st.session_state["api_key"])

        vectorstore = Chroma.from_documents(
            chunks,
            embedding=emb
        )

        return vectorstore

    except Exception as e:
        import traceback
        st.error("🚨 Vectorstore creation failed")
        st.code(traceback.format_exc())
        return None

import time

def tracked_llm_call(prompt):
    import time

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
    except:
        input_tokens = len(str(prompt)) // 4
        output_tokens = len(str(response.content)) // 4

    total_tokens = input_tokens + output_tokens

    input_cost = input_tokens * 0.00015 / 1000
    output_cost = output_tokens * 0.0006 / 1000
    total_cost = input_cost + output_cost

    # GLOBAL
    m = st.session_state.metrics
    m["tokens"] += total_tokens
    m["input_tokens"] += input_tokens
    m["output_tokens"] += output_tokens
    m["cost"] += total_cost
    m["calls"] += 1
    m["response_times"].append(duration)

    # DOCUMENT
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
    
# ------------------------------
# PROCESSING WITH PROGRESS
# ------------------------------

if uploaded_file:

    import hashlib
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

    if st.session_state.get("file_hash") != file_hash:

        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.session_state.full_text = None
        st.session_state.suggested_questions = []

        current_file = uploaded_file.name
        st.session_state.current_file = current_file

        if current_file not in st.session_state.doc_metrics:
            st.session_state.doc_metrics[current_file] = {
                "tokens": 0,
                "response_times": [],
                "calls": 0
            }

        progress = st.progress(0, text="Processing Started...")

        docs = process_file(uploaded_file)

        if not docs:
            st.error("❌ Failed to process document")
            st.stop()

        progress.progress(20, text="File processed")

        st.session_state.full_text = "\n".join(
            [
                str(d.page_content)
                for d in docs
                if d is not None and getattr(d, "page_content", None)
            ]
        )

        # ------------------------------
        # AUTO MODE (LangGraph)
        # ------------------------------
        if st.session_state.get("mode") == "Auto (LangGraph)":
        
            graph = build_graph()
        
            with st.spinner("🤖 Running Auto Processing..."):
        
                result = graph.invoke({
                    "text": st.session_state.full_text,
                    "template": st.session_state.get("resume_template"),
                    "progress": update_progress if "update_progress" in globals() else None
                })
        
            st.session_state.auto_result = result

            st.session_state.doc_type = result.get("doc_type")
        
            st.success(f"Auto Processed → {result['doc_type'].upper()}")
        
        # ------------------------------
        # CONTINUE MANUAL FLOW
        # ------------------------------
        else:
        
            if not st.session_state.full_text.strip():
                st.error("❌ No text extracted (possibly scanned or empty)")
                st.stop()
        
            progress.progress(40, text="Text extracted")
        
            st.session_state.doc_type = detect_document_type(st.session_state.full_text)
            progress.progress(60, text="Document type detected")
        
            st.session_state.structured_data = extract_structured_json(
                st.session_state.full_text,
                st.session_state.doc_type
            )
            progress.progress(80, text="Structured data extracted")
        
            st.session_state.vectorstore = create_vectorstore(docs)
        
            progress.progress(100, text="Vector index created")
            
        # Suggested questions
        doc_type = st.session_state.doc_type

        if doc_type == "invoice":
            st.session_state.suggested_questions = [
                "What is the total amount?",
                "Who is the vendor?",
                "What is the invoice date?",
                "List all line items"
            ]

        elif doc_type == "resume":
            st.session_state.suggested_questions = [
                "Summarize this candidate",
                "What skills does the candidate have?",
                "What is the experience?",
                "What is the education background?"
            ]

        elif doc_type == "report":
            st.session_state.suggested_questions = [
                "what is this report about",
                "identify the document flow",
                "What are the key points?",
                "What are the main sections?"
            ]

        else:
            st.session_state.suggested_questions = [
                "What is this document",
                "What are the key points?",
                "Extract important information"
            ]

        st.session_state.processed_file = uploaded_file.name
        st.session_state.file_hash = file_hash

    doc_type = st.session_state.get("doc_type")

    if doc_type:
        st.success(f"✅ Processed Successfully | Type: {doc_type.upper()}")
        
# ------------------------------
# TABS
# ------------------------------

#tabs = ["Preview", "JSON", "Chat", "Download", "Concur"]
#tabs = ["Preview", "JSON", "Download", "Concur", "Chat", "Metrics"]
tabs = ["Preview", "JSON", "Download", "Concur", "Chat", "Auto", "Metrics"]

selected_tab = st.radio(
    "",
    tabs,
    horizontal=True,
    key="active_tab"
)

# PREVIEW
if selected_tab == "Preview":
    if uploaded_file:
        if "pdf" in uploaded_file.type:
            pdf_viewer(uploaded_file.getvalue(), height=200)
        elif "image" in uploaded_file.type:
            st.image(uploaded_file, width=300)
        elif "text" in uploaded_file.type:
            try:
                preview_text = uploaded_file.getvalue().decode("utf-8")
            except:
                preview_text = uploaded_file.getvalue().decode("cp1252", errors="ignore")
            st.text_area("Preview", preview_text, height=200)
        elif "word" in uploaded_file.type or uploaded_file.name.endswith(".docx"):
            path = save_temp_file(uploaded_file)
            doc = DocxDocument(path)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            st.text_area("DOCX Preview", text, height=200)
        elif uploaded_file.name.endswith(".pptx"):
            from pptx import Presentation
            path = save_temp_file(uploaded_file)
            prs = Presentation(path)
            slides_text = []
            for i, slide in enumerate(prs.slides):
                slide_text = [f"Slide {i+1}"]
        
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_text.append(shape.text.strip())
        
                slides_text.append("\n".join(slide_text))
            st.text_area("PPTX Preview", "\n\n".join(slides_text), height=300)
        elif uploaded_file.name.endswith(".xlsx"):
            import pandas as pd
            path = save_temp_file(uploaded_file)
            df = pd.read_excel(path)
            st.dataframe(df)
            # Optional text preview (for consistency with RAG)
            st.text_area("Excel Preview (Text)", df.to_string(index=False), height=200)

# JSON
if selected_tab == "JSON":
    if st.session_state.structured_data:
        st.json(st.session_state.structured_data)

# ------------------------------
# CHAT
# ------------------------------
if selected_tab == "Chat":

    # ❗ If no document processed
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process a document first")

    else:
        # ------------------------------
        # 🎯 Suggested Questions FIRST
        # ------------------------------
        if st.session_state.suggested_questions:
            st.markdown("### 💡 Suggested Questions")

            cols = st.columns(len(st.session_state.suggested_questions))

            for i, q in enumerate(st.session_state.suggested_questions):
                if cols[i].button(q, key=f"suggest_{i}"):

                    # Add user question
                    st.session_state.chat_history.append(
                        {"role": "user", "content": q}
                    )

                    # Retrieve context
                    docs = st.session_state.vectorstore.similarity_search(
                        q,
                        k=2,
                        filter={"source": st.session_state.current_file}
                    )
                    context = "\n\n".join([d.page_content[:800] for d in docs])

                    # Generate response
                    response = tracked_llm_call(
                        f"""
                    You are a strict document QA assistant.
                    
                    RULES:
                    - Answer ONLY from the provided context
                    - DO NOT add external knowledge
                    - DO NOT assume anything
                    - If partially found, return best matching value
                    - Keep answer concise and factual
                    - NO markdown, NO formatting
                    
                    CONTEXT:
                    {context}
                    
                    QUESTION:
                    {q}
                    """
                    ).content

                    # Store response
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )

                    # Show response immediately
                    st.text(response)

        # ------------------------------
        # 💬 Chat History
        # ------------------------------
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # ------------------------------
        # ✏️ User Input
        # ------------------------------
        query = st.chat_input("Ask a question")

        if query:
            # Add user message
            st.session_state.chat_history.append(
                {"role": "user", "content": query}
            )

            # Retrieve context
            docs = st.session_state.vectorstore.similarity_search(
                query,   
                k=2,
                filter={"source": st.session_state.current_file}
            )
            context = "\n\n".join([d.page_content for d in docs])

            # Generate response
            response = tracked_llm_call(
                f"Answer strictly from context.\nContext:\n{context}\nQ:{query}"
            ).content

            # Store response
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )

            # Display response
            st.write(response)
            
# DOWNLOAD
if selected_tab == "Download":
    if st.session_state.structured_data:
        st.download_button("Download JSON", json.dumps(st.session_state.structured_data, indent=2), "data.json")

        if st.session_state.doc_type == "invoice":

            df = json_to_kv_dataframe(st.session_state.structured_data)
            st.dataframe(df)

            excel = generate_excel(df)

            # 🎯 Extract meaningful invoice name
            data = st.session_state.structured_data

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

            # Clean filename
            safe_name = re.sub(r'[\\/*?:"<>|]', "", str(invoice_name))

            file_name = f"{safe_name}.xlsx"

            # ✅ Show filename
            st.caption(f"📄 {file_name}")

            st.download_button(
                "Download Excel",
                excel,
                file_name
            )

        if st.session_state.doc_type == "resume":

            template_file = st.file_uploader("Upload Resume Template", type=["docx"])

            if template_file:
                st.session_state.generated_resume = build_resume(
                    st.session_state.structured_data,
                    template_file
                )

            if st.session_state.generated_resume:

                data = st.session_state.structured_data

                name = (
                    data.get("name")
                    or data.get("Name")
                    or data.get("candidate_name")
                    or (data.get("personal_details", {}).get("name") if isinstance(data.get("personal_details"), dict) else None)
                    or "candidate"
                )

                safe_name = re.sub(r'[\\/*?:"<>|]', "", name)

            # ✅ ADD THIS to display created file name
                file_name = f"{safe_name}.docx"
                st.caption(f"📄 {file_name}")

                st.download_button(
                    "Download Resume",
                    st.session_state.generated_resume,
                    f"{safe_name}.docx"
                )

# CONCUR
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

            payload = {
                "type": st.session_state.doc_type,
                "data": st.session_state.structured_data,
                "line_items": json_to_kv_dataframe(st.session_state.structured_data).to_dict(orient="records")
            }

            import time
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

# METRICS

if selected_tab == "Metrics":
    st.subheader("📊 Cost & Usage Analytics")

    m = st.session_state.metrics

    if m["calls"] == 0:
        st.warning("No usage yet")
        st.stop()

    avg_time = sum(m["response_times"]) / m["calls"]

    # ---- TOP KPIs ----
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("💰 Total Cost ($)", round(m["cost"], 6))
    col2.metric("🧠 Total Tokens", m["tokens"])
    col3.metric("⚡ LLM Calls", m["calls"])
    col4.metric("⏱ Avg Time (s)", round(avg_time, 2))

    st.markdown("---")

    # ---- TOKEN BREAKDOWN ----
    st.subheader("🔎 Token Breakdown")

    st.write(f"Input Tokens: {m['input_tokens']}")
    st.write(f"Output Tokens: {m['output_tokens']}")

    # ---- COST OVER TIME ----
    if m["response_times"]:
        df = pd.DataFrame({
            "Call": list(range(len(m["response_times"]))),
            "Response Time": m["response_times"]
        })
        st.line_chart(df.set_index("Call"))


    # ---- DOCUMENT COST ----
    st.subheader("📄 Cost per Document")

    doc_data = [
        {"Document": k, "Cost": v["cost"], "Tokens": v["tokens"]}
        for k, v in st.session_state.doc_costs.items()
    ]

    doc_df = pd.DataFrame(doc_data)

    if not doc_df.empty:
    
        # ✅ Add Serial Number starting from 1
        doc_df.insert(0, "SL No", range(1, len(doc_df) + 1))
    
        st.dataframe(doc_df, use_container_width=True, hide_index=True)
        
        # Chart stays same
        st.bar_chart(doc_df.set_index("Document")[["Cost"]])
    
    else:
        st.info("No document-level cost recorded yet")
        
    # ---- COST ALERT ----
    if m["cost"] > 0.05:
        st.warning("⚠️ High usage detected (>$0.05)")

# ------------------------------
# AUTO OUTPUT TAB
# ------------------------------
if selected_tab == "Auto":

    if st.session_state.get("mode") != "Auto (LangGraph)":
        st.info("Switch to Auto Mode from sidebar to use this feature")

    elif not st.session_state.get("auto_result"):
        st.warning("Upload and process a document in Auto Mode")

    else:
        st.subheader("🤖 Auto Processing Output")

        result = st.session_state.auto_result
        res = result.get("result", {})
        doc_type = result.get("doc_type", "")

        st.write(f"📄 Detected Type: **{doc_type.upper()}**")

        # ------------------------------
        # RESUME DOWNLOAD
        # ------------------------------
        if res.get("type") == "resume":

            st.success("✅ Resume generated successfully")

            file = res.get("file")

            if file:
                st.download_button(
                    label="⬇️ Download Resume",
                    data=file,
                    file_name="generated_resume.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            else:
                st.error("Resume file missing")

        # ------------------------------
        # INVOICE OUTPUT
        # ------------------------------
        elif res.get("type") == "invoice":

            st.success("✅ Invoice processed")

            st.dataframe(res.get("table"), use_container_width=True)

            st.download_button(
                "⬇️ Download Excel",
                res.get("excel"),
                "invoice.xlsx"
            )

        # ------------------------------
        # TICKET OUTPUT
        # ------------------------------
        elif res.get("type") == "ticket":

            st.success("✅ Sent to Concur (simulated)")
            st.success("Sent to Concur (simulated)")
