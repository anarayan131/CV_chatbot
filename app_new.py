import streamlit as st
import os
import tempfile

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough



# --- Streamlit UI ---
st.set_page_config(page_title="CV Chatbot", page_icon="🤖")
st.title("🤖 CV Chatbot (Groq + Llama + HuggingFace)")
st.write("Upload a CV PDF, add web links, and ask questions!")


# --- RAG Pipeline ---
@st.cache_resource(show_spinner="Processing your data...")
def build_rag_chain(file, urls):

    # get API key here (your old code incorrectly placed this outside)
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY missing. Add it in your environment variables.")
        st.stop()

    all_docs = []

    # Load PDF
    if file:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getbuffer())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            all_docs.extend(loader.load_and_split())
            os.remove(tmp_path)

        except Exception as e:
            st.error(f"PDF loading error: {e}")
            return None

    # Load URLs
    if urls:
        try:
            url_list = [u.strip() for u in urls.split("\n") if u.strip()]
            if url_list:
                loader = WebBaseLoader(url_list)
                all_docs.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading URLs: {e}")

    if not all_docs:
        st.warning("No data found. Upload a CV and/or enter URLs.")
        return None

    # Split docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(all_docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vectorstore
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever()

    # Groq LLM
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.2
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are a professional AI assistant built to help recruiters understand the candidate's CV and background.

    Your tone must be:
    - concise
    - confident
    - professional
    - recruiter-friendly
    - strength-focused
    - strictly grounded in the provided context

    RULES:
    1. ONLY use information explicitly found in the context. No speculation.
    2.  If the context lacks information, say: “I do not have access to this infomration.” and ignore following instructions.
    3. Start every answer with a 1–2 sentence high-level summary.
    4. Then provide a short, clear, structured breakdown:
       - Key Skills
       - Relevant Experience
       - Achievements / Impact
    5. Highlight leadership, ownership, technical strengths, and results when present.

    CONTEXT:
    {context}

    QUESTION:
    {question}
    """)

    # LCEL pipeline
    qa_chain = (
        RunnableParallel({
            "context": retriever,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
    )

    return qa_chain


# --- UI ELEMENTS ---
uploaded_file = st.file_uploader("Upload your CV (PDF)", type="pdf")
urls_input = st.text_area(
    "Add web links (one per line)",
    placeholder="https://www.offis.de/offis/person/anand-narayan.html\nhttps://www.linkedin.com/in/anand-narayan-3a7069110/?originalSubdomain=de"
)
process_button = st.button("Process Inputs")


# --- BUILD CHAIN ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if process_button:
    if uploaded_file or urls_input:
        st.session_state.qa_chain = build_rag_chain(uploaded_file, urls_input)
        if st.session_state.qa_chain:
            st.session_state.messages = []
            st.success("Done! You can now ask questions.")
    else:
        st.warning("Upload a PDF or paste URLs first.")


# --- CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --- CHAT INPUT ---
if user_input := st.chat_input("Ask a question about the CV"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if "qa_chain" in st.session_state:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain.invoke(user_input)

            # Groq wrapper returns a ChatMessage object
            answer = result.content

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
    else:
        st.info("Process your PDF/URLs first.")
