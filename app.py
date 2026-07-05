import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# =====================================================
# Chroma Vector Database
# =====================================================

DB_PATH = "chroma_db"

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embedding
)

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

# =====================================================
# Load SLM
# =====================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    return tokenizer, model


tokenizer, model = load_model()

# =====================================================
# Streamlit UI
# =====================================================

st.set_page_config(
    page_title="Research Paper RAG using SLM",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Paper RAG using SLM")

st.write(
    "Ask questions about the uploaded research papers."
)

question = st.text_input(
    "Enter your question:"
)

if question:

    with st.spinner("Searching research papers..."):

        docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

    prompt = f"""
You are an AI Research Assistant.

Answer ONLY using the information provided in the context.

If the answer is not present in the context, reply exactly:

"The answer is not available in the uploaded research papers."

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    with st.spinner("Generating answer..."):

        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    answer = generated_text.split("Answer:")[-1].strip()

    st.subheader("Answer")

    st.write(answer)

    st.subheader("Retrieved Context")

    for i, doc in enumerate(docs, start=1):
        with st.expander(f"Document Chunk {i}"):
            st.write(doc.page_content)