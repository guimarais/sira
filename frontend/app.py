"""SIRA — Stock Investment Research Assistant — Streamlit frontend."""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="SIRA", page_icon="📈", layout="wide")
st.title("📈 SIRA — Stock Investment Research Assistant")

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict] role/content for Claude history
if "display" not in st.session_state:
    st.session_state.display = []   # list[dict] role/content/meta for rendering


# ── Sidebar: file uploads ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Data")

    with st.expander("PDF document", expanded=False):
        pdf_file = st.file_uploader("Choose a PDF", type=["pdf"], key="pdf_upload")
        if st.button("Ingest PDF", disabled=pdf_file is None):
            with st.spinner("Ingesting…"):
                try:
                    r = requests.post(
                        f"{API_URL}/upload-pdf",
                        files={"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")},
                        timeout=120,
                    )
                    r.raise_for_status()
                    d = r.json()["detail"]
                    st.success(f"{d['filename']} — {d['chunks_added']} chunks added")
                except requests.HTTPError as exc:
                    st.error(exc.response.json().get("detail", str(exc)))
                except Exception as exc:
                    st.error(str(exc))

    with st.expander("CSV / XLSX stock data", expanded=False):
        data_file = st.file_uploader(
            "Choose a CSV or XLSX", type=["csv", "xlsx"], key="data_upload"
        )
        sheet_input = st.text_input(
            "Sheet (XLSX only — name or index)", value="0", key="sheet_input"
        )
        if st.button("Ingest file", disabled=data_file is None):
            with st.spinner("Ingesting…"):
                try:
                    if data_file.name.lower().endswith(".csv"):
                        r = requests.post(
                            f"{API_URL}/upload-csv",
                            files={"file": (data_file.name, data_file.getvalue(), "text/csv")},
                            timeout=60,
                        )
                    else:
                        r = requests.post(
                            f"{API_URL}/upload-xlsx",
                            params={"sheet": sheet_input},
                            files={
                                "file": (
                                    data_file.name,
                                    data_file.getvalue(),
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                )
                            },
                            timeout=60,
                        )
                    r.raise_for_status()
                    d = r.json()["detail"]
                    st.success(f"{d['rows_inserted']} rows ingested — columns: {', '.join(d['columns'])}")
                except requests.HTTPError as exc:
                    st.error(exc.response.json().get("detail", str(exc)))
                except Exception as exc:
                    st.error(str(exc))

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.display = []
        st.rerun()

    st.divider()
    with st.expander("Backend health", expanded=False):
        if st.button("Check"):
            try:
                r = requests.get(f"{API_URL}/health", timeout=5)
                body = r.json()
                colour = "green" if body["status"] == "ok" else "orange"
                st.markdown(f"**Status:** :{colour}[{body['status']}]")
                for k, v in body["checks"].items():
                    icon = "✅" if v == "ok" else "⚠️"
                    st.write(f"{icon} {k}: {v}")
            except Exception as exc:
                st.error(str(exc))


# ── Chat history display ───────────────────────────────────────────────────────
for entry in st.session_state.display:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry["role"] == "assistant" and (entry.get("sources") or entry.get("intent")):
            with st.expander("Sources & intent"):
                st.write(f"**Intent:** {entry.get('intent', '—')}")
                for src in entry.get("sources", []):
                    st.write(src)


# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask an investment research question…"):
    # Show user message immediately
    st.session_state.display.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                r = requests.post(
                    f"{API_URL}/query",
                    json={"question": prompt, "history": st.session_state.messages},
                    timeout=120,
                )
                r.raise_for_status()
                body = r.json()
                answer = body["answer"]
                sources = body.get("sources", [])
                intent = body.get("intent", "")

                st.markdown(answer)
                if sources or intent:
                    with st.expander("Sources & intent"):
                        st.write(f"**Intent:** {intent}")
                        for src in sources:
                            st.write(src)

                # Append to Claude history (plain role/content only)
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Append to display history (includes metadata)
                st.session_state.display.append(
                    {"role": "assistant", "content": answer, "sources": sources, "intent": intent}
                )

            except requests.HTTPError as exc:
                err = exc.response.json().get("detail", str(exc))
                st.error(err)
                st.session_state.display.append({"role": "assistant", "content": f"Error: {err}"})
            except Exception as exc:
                st.error(str(exc))
                st.session_state.display.append({"role": "assistant", "content": f"Error: {exc}"})
