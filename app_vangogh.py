import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from gigachat import GigaChat
import os

st.set_page_config(
    page_title="The Letters of Vincent van Gogh",
    page_icon="🎨",
    layout="centered"
)

st.markdown("""
<style>
    .summary {
        background: #eaf4fc;
        border-left: 4px solid #3498db;
        padding: 15px 20px;
        margin: 20px 0;
        border-radius: 0 8px 8px 0;
        line-height: 1.6;
    }
    .result {
        background: #f5f5f5;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    giga = GigaChat(
        credentials=st.secrets["GIGA_KEY"],  # берём из secrets Streamlit
        verify_ssl_certs=False
    )
    return vector_store, giga

SYSTEM_PROMPT = (
    "You are assistant with the corpus of Van Gogh's letters to brother Theo. "
    "Give answers to user based on texts fragments. "
    "If there is no information tell about it"
)

def summarize(query, docs, giga):
    chunks = "\n\n".join(
        f"Fragment {i+1} ({doc.metadata.get('filename', '')}):\n{doc.page_content}"
        for i, (doc, _score) in enumerate(docs[:3])
    )
    prompt = f"{SYSTEM_PROMPT}\n\nAnswer: {query}\n\n{chunks}"
    try:
        response = giga.chat(prompt)
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка GigaChat: {e}"

# интерфейс
st.title("🎨 The Letters of Vincent van Gogh")
st.caption("Semantic search powered by GigaChat")

vector_store, giga = load_resources()

query = st.text_input("", placeholder="For example: Paris, landscape, yellow color")

if query:
    with st.spinner("Searching..."):
        results = vector_store.similarity_search_with_score(query, k=5)

        if results:
            summary = summarize(query, results, giga)

            if summary:
                st.markdown("**Answer:**")
                st.markdown(f"""
                <div class="summary">
                    {summary}
                    <div style="font-size:12px; color:#7f8c8d; margin-top:10px">
                        Ответ сгенерирован ИИ на основе найденных фрагментов
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### Найденные фрагменты:")
            for doc, score in results:
                with st.expander(f"📄 {doc.metadata.get('filename', 'fragment')} | score: {score:.2f}"):
                    st.write(doc.page_content)
