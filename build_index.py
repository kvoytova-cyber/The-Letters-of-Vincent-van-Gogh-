import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


url = "van_gogh_letters_data.csv"
df = pd.read_csv(url)
df = df.dropna(subset=["letter_text"])

print(f"Писем загружено: {len(df)}")


documents = []
for _, row in df.iterrows():
    doc = Document(
        page_content=row["letter_text"],
        metadata={
            "date": str(row.get("date", "Дата неизвестна")),
            "letter_id": str(row.get("letter_id", "")),
        }
    )
    documents.append(doc)

# разбиваем длинные письма на чанки
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Чанков после разбивки: {len(chunks)}")

# строим индекс
print("Строю FAISS индекс...")
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_index")
print("Индекс сохранён в faiss_index/")