import streamlit as st
import faiss
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


st.set_page_config(page_title="Ask My CV", page_icon="🤖", layout="centered")
st.title("🤖 Chat With My CV")

# ---- Load precomputed vectorstore ----
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever()

retriever = load_vectorstore()

# ---- LLM ----
groq_api_key = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
    temperature=0.2
)

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
    2. If the context lacks information, say: “I do not have access to this infomration.” and ignore following instructions.
    3. Highlight Key Skills, achievements, leadership, ownership, technical strengths, and results when present.

Context:
{context}

Question:
{question}
""")

qa_chain = (
    RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
)

# ---- Chat interface ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

user_input = st.chat_input("Hello! I'm ANa, Anand's AI assistant. Ask me anything about his background, works and skills.")
if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})
    st.chat_message("user").markdown(user_input)

    with st.spinner("Thinking..."):
        response = qa_chain.invoke(user_input)

    answer = response.content
    st.session_state.messages.append({"role":"assistant","content":answer})
    st.chat_message("assistant").markdown(answer)
