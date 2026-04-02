import json
import re
from io import BytesIO
from docx import Document as DocxDocument
from langchain_openai import ChatOpenAI
from streamlit import session_state as st_state

import tempfile
from pathlib import Path


def safe_json_parse(text):
    """
    Safely parse LLM JSON output.
    Handles:
    - trailing commas
    - partial JSON
    - text before/after JSON
    """

    if not text:
        return {}

    # Remove markdown wrappers if any
    text = text.strip().replace("```json", "").replace("```", "").strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except:
        pass

    # Try to extract JSON block
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass

    # Fix common trailing comma issue
    try:
        text_fixed = re.sub(r",\s*}", "}", text)
        text_fixed = re.sub(r",\s*]", "]", text_fixed)
        return json.loads(text_fixed)
    except:
        pass

    # Final fallback
    return {"raw_output": text}

def extract_structured_json(text, doc_type):
    """
    Extract structured JSON from document text.
    For resumes, this version explicitly preserves all date fields.
    """

    clean_text = re.sub(r"[^\x00-\x7F]+", " ", text or "")
    clean_text = clean_text.replace("{", "").replace("}", "").strip()

    if "api_key" not in st_state:
        return {"error": "Missing API key"}

    llm = ChatOpenAI(
        model=st_state.get("model_choice", "gpt-4o-mini"),
        temperature=0,
        api_key=st_state["api_key"]
    )

    if doc_type == "resume":
        prompt = f"""
You are a strict JSON generator.

Return ONLY valid JSON.
Do not add markdown.
Do not wrap in triple backticks.
Do not add explanation.

STRICT SCHEMA:
{{
  "name": "",
  "email": "",
  "phone": "",
  "location": "",
  "linkedin": "",
  "skills": [],
  "summary": "",
  "education": [
    {{
      "institution": "",
      "degree": "",
      "field_of_study": "",
      "start_date": "",
      "end_date": "",
      "graduation_date": "",
      "location": "",
      "details": []
    }}
  ],
  "experience": [
    {{
      "company": "",
      "role": "",
      "location": "",
      "start_date": "",
      "end_date": "",
      "is_current": false,
      "description": []
    }}
  ],
  "certifications": [
    {{
      "name": "",
      "issuer": "",
      "date": "",
      "expiry_date": ""
    }}
  ],
  "projects": [
    {{
      "name": "",
      "role": "",
      "start_date": "",
      "end_date": "",
      "description": []
    }}
  ]
}}

STRICT RULES:
- Extract ALL experience entries.
- Extract ALL education entries.
- Extract ALL certifications if present.
- Preserve ALL dates exactly as written in the CV.
- Never omit employment date ranges.
- Never omit education dates, graduation dates, or certification dates if present.
- Keep month/year or full date exactly as seen.
- If an entry says Present or Current, store that value in "end_date" exactly as written and set "is_current" to true.
- Do not merge multiple jobs into one job.
- Do not summarize away date fields.
- Use empty strings for missing scalar values.
- Use empty arrays for missing list values.
- description and details must always be arrays of strings.

CV TEXT:
{clean_text[:12000]}
"""
    else:
        prompt = f"""
Extract ALL possible key-value pairs from the document.

Return ONLY valid JSON.

RULES:
- Capture every identifiable field
- Preserve original field names where possible
- Include nested structures if present
- Do NOT summarize
- Do NOT skip fields

DOCUMENT TEXT:
{clean_text[:12000]}
"""

    try:
        response = llm.invoke(prompt).content.strip()
        response = response.replace("```json", "").replace("```", "").strip()

        parsed = safe_json_parse(response)

        if isinstance(parsed, list):
            merged = {}
            for item in parsed:
                if isinstance(item, dict):
                    merged.update(item)
            parsed = merged if merged else {"data": parsed}

        if not isinstance(parsed, dict):
            parsed = {"data": parsed}

        if doc_type == "resume":
            # fallback name
            if not parsed.get("name"):
                try:
                    name_prompt = f"""
Extract only the candidate's full name from this resume text.
Return only the name.
No explanation.

{clean_text[:3000]}
"""
                    fallback_name = llm.invoke(name_prompt).content.strip()
                    parsed["name"] = fallback_name
                except Exception:
                    parsed["name"] = "Candidate"

            # normalize top-level fields
            for field in [
                "name", "email", "phone", "location", "linkedin", "summary"
            ]:
                if field not in parsed or parsed[field] is None:
                    parsed[field] = ""

            for field in ["skills", "education", "experience", "certifications", "projects"]:
                if field not in parsed or parsed[field] is None:
                    parsed[field] = []

            # normalize education entries
            normalized_education = []
            for edu in parsed.get("education", []):
                if isinstance(edu, dict):
                    normalized_education.append({
                        "institution": str(edu.get("institution", "") or ""),
                        "degree": str(edu.get("degree", "") or ""),
                        "field_of_study": str(edu.get("field_of_study", "") or ""),
                        "start_date": str(edu.get("start_date", "") or ""),
                        "end_date": str(edu.get("end_date", "") or ""),
                        "graduation_date": str(edu.get("graduation_date", "") or ""),
                        "location": str(edu.get("location", "") or ""),
                        "details": edu.get("details", []) if isinstance(edu.get("details", []), list) else []
                    })
            parsed["education"] = normalized_education

            # normalize experience entries
            normalized_experience = []
            for exp in parsed.get("experience", []):
                if isinstance(exp, dict):
                    normalized_experience.append({
                        "company": str(exp.get("company", "") or ""),
                        "role": str(exp.get("role", "") or ""),
                        "location": str(exp.get("location", "") or ""),
                        "start_date": str(exp.get("start_date", "") or ""),
                        "end_date": str(exp.get("end_date", "") or ""),
                        "is_current": bool(exp.get("is_current", False)),
                        "description": exp.get("description", []) if isinstance(exp.get("description", []), list) else []
                    })
            parsed["experience"] = normalized_experience

            # normalize certifications
            normalized_certifications = []
            for cert in parsed.get("certifications", []):
                if isinstance(cert, dict):
                    normalized_certifications.append({
                        "name": str(cert.get("name", "") or ""),
                        "issuer": str(cert.get("issuer", "") or ""),
                        "date": str(cert.get("date", "") or ""),
                        "expiry_date": str(cert.get("expiry_date", "") or "")
                    })
            parsed["certifications"] = normalized_certifications

            # normalize projects
            normalized_projects = []
            for proj in parsed.get("projects", []):
                if isinstance(proj, dict):
                    normalized_projects.append({
                        "name": str(proj.get("name", "") or ""),
                        "role": str(proj.get("role", "") or ""),
                        "start_date": str(proj.get("start_date", "") or ""),
                        "end_date": str(proj.get("end_date", "") or ""),
                        "description": proj.get("description", []) if isinstance(proj.get("description", []), list) else []
                    })
            parsed["projects"] = normalized_projects

            if parsed.get("name"):
                parsed["name"] = parsed["name"].strip().title()

        return parsed

    except Exception as e:
        return {
            "error": "LLM request failed",
            "details": str(e)[:300]
        }


def generate_resume_summary(data):
    """
    Generate a concise professional summary, while preserving career timeline references.
    This does not replace detailed experience/education sections; it only builds a top summary.
    """

    if "api_key" not in st_state:
        return "Summary not available"

    llm = ChatOpenAI(
        model=st_state.get("model_choice", "gpt-4o-mini"),
        temperature=0,
        api_key=st_state["api_key"]
    )

    prompt = f"""
Create a professional resume summary in plain text.

STRICT RULES:
- No markdown
- No * or **
- Plain text only
- 4 to 8 concise bullet-style lines using simple hyphen prefixes
- Mention total profile strengths, major domains, and seniority
- Preserve important date context where relevant
- Do not invent dates
- Do not remove date references if they are important to career continuity
- Do not rewrite exact extracted timelines in a conflicting way

CANDIDATE DATA:
{json.dumps(data, ensure_ascii=False)}
"""

    try:
        return llm.invoke(prompt).content.strip()
    except Exception:
        return "Summary not available"


def build_resume(data, template_file):
    """
    Build resume DOCX with structured sections that preserve all dates end-to-end.
    Expected placeholders in template can include:
    {{name}}, {{email}}, {{phone}}, {{location}}, {{linkedin}}, {{summary}},
    {{skills}}, {{experience}}, {{education}}, {{certifications}}, {{projects}}
    """

    def safe_str(value):
        return "" if value is None else str(value)

    def format_date_range(start_date, end_date):
        start_date = safe_str(start_date).strip()
        end_date = safe_str(end_date).strip()

        if start_date and end_date:
            return f"{start_date} - {end_date}"
        if start_date:
            return start_date
        if end_date:
            return end_date
        return ""

    def format_skills(skills):
        if not isinstance(skills, list) or not skills:
            return ""
        return ", ".join(str(s).strip() for s in skills if str(s).strip())

    def format_experience(experience):
        if not isinstance(experience, list) or not experience:
            return ""

        lines = []
        for exp in experience:
            if not isinstance(exp, dict):
                continue

            role = safe_str(exp.get("role")).strip()
            company = safe_str(exp.get("company")).strip()
            location = safe_str(exp.get("location")).strip()
            start_date = safe_str(exp.get("start_date")).strip()
            end_date = safe_str(exp.get("end_date")).strip()
            desc = exp.get("description", [])

            header_parts = []
            title_part = " - ".join([p for p in [role, company] if p])
            if title_part:
                header_parts.append(title_part)

            date_range = format_date_range(start_date, end_date)
            if date_range:
                header_parts.append(f"({date_range})")

            if location:
                header_parts.append(location)

            if header_parts:
                lines.append(" ".join(header_parts))

            if isinstance(desc, list):
                for item in desc:
                    item = safe_str(item).strip()
                    if item:
                        lines.append(f"- {item}")

            lines.append("")

        return "\n".join(lines).strip()

    def format_education(education):
        if not isinstance(education, list) or not education:
            return ""

        lines = []
        for edu in education:
            if not isinstance(edu, dict):
                continue

            institution = safe_str(edu.get("institution")).strip()
            degree = safe_str(edu.get("degree")).strip()
            field = safe_str(edu.get("field_of_study")).strip()
            location = safe_str(edu.get("location")).strip()
            start_date = safe_str(edu.get("start_date")).strip()
            end_date = safe_str(edu.get("end_date")).strip()
            graduation_date = safe_str(edu.get("graduation_date")).strip()
            details = edu.get("details", [])

            first_line_parts = []
            degree_part = ", ".join([p for p in [degree, field] if p])
            if degree_part:
                first_line_parts.append(degree_part)
            if institution:
                first_line_parts.append(institution)

            if first_line_parts:
                lines.append(" - ".join(first_line_parts))

            date_text = ""
            if graduation_date:
                date_text = graduation_date
            else:
                date_text = format_date_range(start_date, end_date)

            second_line_parts = []
            if date_text:
                second_line_parts.append(date_text)
            if location:
                second_line_parts.append(location)

            if second_line_parts:
                lines.append(", ".join(second_line_parts))

            if isinstance(details, list):
                for item in details:
                    item = safe_str(item).strip()
                    if item:
                        lines.append(f"- {item}")

            lines.append("")

        return "\n".join(lines).strip()

    def format_certifications(certifications):
        if not isinstance(certifications, list) or not certifications:
            return ""

        lines = []
        for cert in certifications:
            if not isinstance(cert, dict):
                continue

            name = safe_str(cert.get("name")).strip()
            issuer = safe_str(cert.get("issuer")).strip()
            date = safe_str(cert.get("date")).strip()
            expiry_date = safe_str(cert.get("expiry_date")).strip()

            line_parts = []
            if name:
                line_parts.append(name)
            if issuer:
                line_parts.append(issuer)

            line = " - ".join(line_parts)

            date_bits = []
            if date:
                date_bits.append(f"Issued: {date}")
            if expiry_date:
                date_bits.append(f"Expires: {expiry_date}")

            if date_bits:
                line = f"{line} ({', '.join(date_bits)})" if line else ", ".join(date_bits)

            if line:
                lines.append(line)

        return "\n".join(lines).strip()

    def format_projects(projects):
        if not isinstance(projects, list) or not projects:
            return ""

        lines = []
        for proj in projects:
            if not isinstance(proj, dict):
                continue

            name = safe_str(proj.get("name")).strip()
            role = safe_str(proj.get("role")).strip()
            start_date = safe_str(proj.get("start_date")).strip()
            end_date = safe_str(proj.get("end_date")).strip()
            description = proj.get("description", [])

            header_parts = []
            title = " - ".join([p for p in [name, role] if p])
            if title:
                header_parts.append(title)

            date_range = format_date_range(start_date, end_date)
            if date_range:
                header_parts.append(f"({date_range})")

            if header_parts:
                lines.append(" ".join(header_parts))

            if isinstance(description, list):
                for item in description:
                    item = safe_str(item).strip()
                    if item:
                        lines.append(f"- {item}")

            lines.append("")

        return "\n".join(lines).strip()

    def replace_placeholders_in_paragraph(paragraph, placeholders):
        for key, value in placeholders.items():
            if key in paragraph.text:
                paragraph.text = paragraph.text.replace(key, value)

    def replace_placeholders(doc, placeholders):
        for para in doc.paragraphs:
            replace_placeholders_in_paragraph(para, placeholders)

        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        replace_placeholders_in_paragraph(para, placeholders)

        # headers/footers
        for section in doc.sections:
            for para in section.header.paragraphs:
                replace_placeholders_in_paragraph(para, placeholders)
            for para in section.footer.paragraphs:
                replace_placeholders_in_paragraph(para, placeholders)

    # generate summary
    summary = generate_resume_summary(data)

    # load template
    if template_file:
        try:
            if isinstance(template_file, bytes):
                doc = DocxDocument(BytesIO(template_file))
            elif hasattr(template_file, "read"):
                content = template_file.read()
                template_file.seek(0)
                doc = DocxDocument(BytesIO(content))
            else:
                doc = DocxDocument()
        except Exception as e:
            print("Template load error:", e)
            doc = DocxDocument()
    else:
        doc = DocxDocument()

    # prepare structured placeholders with preserved dates
    placeholders = {
        "{{name}}": safe_str(data.get("name", "")),
        "{{email}}": safe_str(data.get("email", "")),
        "{{phone}}": safe_str(data.get("phone", "")),
        "{{location}}": safe_str(data.get("location", "")),
        "{{linkedin}}": safe_str(data.get("linkedin", "")),
        "{{summary}}": safe_str(summary),
        "{{skills}}": format_skills(data.get("skills", [])),
        "{{experience}}": format_experience(data.get("experience", [])),
        "{{education}}": format_education(data.get("education", [])),
        "{{certifications}}": format_certifications(data.get("certifications", [])),
        "{{projects}}": format_projects(data.get("projects", [])),
    }

    replace_placeholders(doc, placeholders)

    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()

def save_temp_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

def detect_document_type(text):

    # ------------------------------
    # SAFETY CHECK
    # ------------------------------
    if "api_key" not in st_state:
        return "other"

    llm = ChatOpenAI(
        model=st_state.get("model_choice", "gpt-4o-mini"),
        temperature=0,
        api_key=st_state["api_key"]
    )

    prompt = f"""
Classify document into ONE label:

resume
invoice
receipt
report
ticket
other

STRICT RULES:
- Return ONLY one word
- No explanation
- No sentence

{text[:2000]}
"""

    try:
        raw = llm.invoke(prompt).content.lower().strip()
    except Exception:
        return "other"

    # ------------------------------
    # ROBUST MATCHING
    # ------------------------------
    labels = ["resume", "invoice", "receipt", "report", "ticket"]

    for label in labels:
        if label in raw:
            return label

    return "other"

# Resume helpers
def replace_placeholders(doc, placeholders):

    for para in doc.paragraphs:
        for key, value in placeholders.items():
            if key in para.text:
                para.text = para.text.replace(key, str(value))



def json_to_kv_dataframe(data):
    rows = []

    def flatten(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                flatten(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                flatten(f"{prefix}[{i}]", item)
        else:
            rows.append({"Field": prefix, "Value": json.dumps(obj) if isinstance(obj, (dict, list)) else str(obj)})

    flatten("", data)
    return pd.DataFrame(rows)



def generate_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='data')
    return output.getvalue()
