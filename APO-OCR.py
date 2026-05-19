import streamlit as st
import openai
import pdfplumber
import json
import os
import re
import zipfile
import io
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(page_title="🧾 Azure AI Invoice Batch Extractor", layout="wide", page_icon="🧾")
st.title("🧾 Azure AI Invoice Batch Extractor")
st.markdown("Upload **multiple PDF invoices** → Azure AI extracts structured fields → Export results as CSV or ZIP")

# --- Sidebar: Azure Configuration ---
with st.sidebar:
    st.header("⚙️ Azure OpenAI Settings")

    default_endpoint = st.secrets.get("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    default_key = st.secrets.get("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY", ""))

    azure_endpoint = st.text_input(
        "Azure Endpoint",
        value=default_endpoint,
        type="password",
        help="Format: https://YOUR-RESOURCE.openai.azure.com"
    )
    azure_api_key = st.text_input("API Key", value=default_key, type="password")

    deployment_name = st.text_input(
        "Deployment Name",
        value="gpt-5.4-mini",
        help="Must match exactly what's deployed in Azure Portal"
    )

    # ✅ Pre-select your API version
    api_version = st.selectbox(
        "API Version",
        options=["2023-05-15", "2024-02-15-preview", "2024-06-01", "2025-04-01-preview"],
        index=3,  # Default to your version
        help="2025-04-01-preview uses Responses API with text.format"
    )

    st.markdown("---")
    st.info("🔐 Store credentials in `.streamlit/secrets.toml` for production.")

# --- Your Exact Extraction Prompt (Locked) ---
EXTRACTION_PROMPT = """Extract the following fields from this PDF invoice:
1. VendorName - The company name issuing the invoice
2. InvoiceNumber - Invoice number/ID (Strip all special symbols, hyphens, slashes, or spaces)
3. InvoiceDate - Invoice date in DD-MM-YYYY format
4. INVOICE_AMOUNT - Total amount due (numeric value only)
5. Currency - Currency code (USD, EUR, GBP, etc.)
6. TaxAmount - Tax/VAT amount if visible

RULES:
- Return ONLY raw valid JSON matching the structure. Do not include markdown wraps or conversational text explanations.
- If a field is missing, set it to null.
- For numeric values, return raw numbers, not strings.

JSON STRUCTURE:
{
  "VendorName": null,
  "InvoiceNumber": null,
  "InvoiceDate": null,
  "INVOICE_AMOUNT": null,
  "Currency": null,
  "TaxAmount": null
}"""


# --- 🔥 ROBUST JSON PARSER (Works with any API) ---
def parse_json_from_response(text: str) -> dict:
    import json, re
    # Remove markdown code fences
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    # Extract JSON object between outermost braces
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        text = match.group(0)
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt common fixes
        fixed = text.replace("'", '"')
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        fixed = re.sub(r'(\w+):', r'"\1":', fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON. Preview: {text[:300]}...") from e


# --- 🔥 EXTRACTION FUNCTION: API-Version Aware ---
def extract_invoice_fields(text: str, client, deployment: str, api_version: str) -> dict:
    """
    Send text to Azure OpenAI and parse JSON response.
    Automatically uses Chat Completions or Responses API based on api_version.
    """
    system_message = (
        "You are a precise invoice data extraction engine. "
        "You MUST output ONLY valid raw JSON. No markdown. No explanations. No extra text."
    )

    # Truncate text to stay within context window
    MAX_CHARS = 100000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n...[TEXT TRUNCATED FOR CONTEXT LIMIT]..."

    user_content = EXTRACTION_PROMPT + "\n\nINVOICE TEXT:\n" + text

    # 🔥 Route based on API version
    if api_version.startswith("2025-"):
        # === Responses API (2025-04-01-preview) ===
        response = client.responses.create(
            model=deployment,
            input=[  # ← "input" instead of "messages"
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            text={  # ← response_format moved to text.format
                "format": {
                    "type": "json_object"
                }
            }
        )
        raw_content = response.output_text.strip()
    else:
        # === Chat Completions API (older versions) ===
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0
            # No response_format to avoid routing issues
        )
        raw_content = response.choices[0].message.content.strip()

    return parse_json_from_response(raw_content)


def extract_text_from_pdf(file) -> str:
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages).strip()


def clean_numeric_value(val):
    if val is None or isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        cleaned = re.sub(r"[^\d.\-]", "", val)
        if not cleaned or cleaned in ["-", "."]:
            return None
        try:
            return float(cleaned) if '.' in cleaned else int(cleaned)
        except ValueError:
            return None
    return val


def create_zip_from_jsons(results: list) -> io.BytesIO:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for item in results:
            if item.get("success") and item.get("data"):
                safe_name = re.sub(r'[^\w\-_\.]', '_', item['filename'])
                filename = f"{safe_name.rsplit('.', 1)[0]}_extracted.json"
                zip_file.writestr(filename, json.dumps(item["data"], indent=2))
    zip_buffer.seek(0)
    return zip_buffer


# --- Main UI ---
st.subheader("📁 Upload Invoice PDFs")
uploaded_files = st.file_uploader(
    "Choose PDF files (hold Ctrl/Cmd to select multiple)",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

if uploaded_files:
    st.info(f"📎 **{len(uploaded_files)} file(s) selected**")
    col1, col2 = st.columns([3, 1])
    with col1:
        process_btn = st.button("🚀 Process All Invoices", type="primary", use_container_width=True)
    with col2:
        if st.session_state.batch_results and st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.batch_results = []
            st.rerun()

    if process_btn:
        if not azure_endpoint or not azure_api_key:
            st.error("❌ Please configure Azure Endpoint and API Key in the sidebar.")
            st.stop()
        if not deployment_name:
            st.error("❌ Please provide a valid Deployment Name.")
            st.stop()

        clean_endpoint = azure_endpoint.rstrip('/').removesuffix('/openai/v1').removesuffix('/openai')

        try:
            client = openai.AzureOpenAI(
                azure_endpoint=clean_endpoint,
                api_key=azure_api_key,
                api_version=api_version
            )
        except Exception as e:
            st.error(f"❌ Failed to initialize Azure client: {e}")
            st.stop()

        progress_bar = st.progress(0)
        status_text = st.empty()
        all_results = []
        errors = []

        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"🔄 Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
            progress_bar.progress(idx / len(uploaded_files))

            result_item = {
                "filename": uploaded_file.name,
                "size_kb": round(uploaded_file.size / 1024, 1),
                "success": False, "data": None, "error": None,
                "timestamp": datetime.now().isoformat()
            }

            try:
                full_text = extract_text_from_pdf(uploaded_file)
                if not full_text:
                    raise ValueError("No text extracted - PDF may be image-only/scanned")

                # 🔥 Pass api_version to route correctly
                extracted = extract_invoice_fields(full_text, client, deployment_name, api_version)

                for field in ["INVOICE_AMOUNT", "TaxAmount"]:
                    extracted[field] = clean_numeric_value(extracted.get(field))

                result_item["success"] = True
                result_item["data"] = extracted

            except openai.APIError as e:
                result_item["error"] = f"Azure API Error: {str(e)}"
                errors.append(f"{uploaded_file.name}: {str(e)}")
            except (json.JSONDecodeError, ValueError) as e:
                result_item["error"] = f"JSON Parse Error: {str(e)}"
                errors.append(f"{uploaded_file.name}: Invalid JSON response")
            except Exception as e:
                result_item["error"] = f"{type(e).__name__}: {str(e)}"
                errors.append(f"{uploaded_file.name}: {str(e)}")

            all_results.append(result_item)
            progress_bar.progress((idx + 1) / len(uploaded_files))

        st.session_state.batch_results = all_results
        status_text.text("✅ Batch processing completed!")
        progress_bar.empty()

        successful = [r for r in all_results if r["success"]]
        failed = [r for r in all_results if not r["success"]]

        st.success(f"🎉 **Completed**: {len(successful)} succeeded, {len(failed)} failed")

        if errors:
            with st.expander(f"⚠️ View {len(errors)} Error(s)", expanded=False):
                for err in errors:
                    st.warning(f"• {err}")

        if successful:
            st.subheader("📊 Extracted Data Summary")
            table_data = []
            for item in successful:
                row = {"Filename": item["filename"], "Size (KB)": item["size_kb"]}
                row.update(item["data"])
                table_data.append(row)

            df = pd.DataFrame(table_data)
            preferred_order = ["Filename", "Size (KB)", "VendorName", "InvoiceNumber", "InvoiceDate", "INVOICE_AMOUNT",
                               "Currency", "TaxAmount"]
            existing_cols = [c for c in preferred_order if c in df.columns]
            remaining_cols = [c for c in df.columns if c not in preferred_order]
            df = df[existing_cols + remaining_cols]

            st.dataframe(df, use_container_width=True, hide_index=True)

            col_csv, col_zip = st.columns(2)
            with col_csv:
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Combined CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"invoices_batch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_zip:
                zip_buffer = create_zip_from_jsons(all_results)
                st.download_button(
                    label="📦 Download JSON Files (ZIP)",
                    data=zip_buffer,
                    file_name=f"invoices_batch_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )

            with st.expander("🔍 View Individual Results", expanded=False):
                for item in successful:
                    st.markdown(f"##### ✅ {item['filename']}")
                    st.json(item["data"])
                    st.divider()

        if failed:
            with st.expander(f"❌ {len(failed)} Failed File(s) Details", expanded=False):
                for item in failed:
                    st.markdown(f"##### ❌ {item['filename']} ({item['size_kb']} KB)")
                    st.code(item["error"], language="text")
                    st.divider()
else:
    st.info("👆 Upload one or more PDF invoices to begin batch extraction")

st.markdown("---")
st.caption(f"Built with Streamlit + Azure OpenAI | Deployment: `{deployment_name}` | API: `{api_version}`")
