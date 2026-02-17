from flask import Flask, request, render_template_string
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from gigachat import GigaChat
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

print("Загружаю индекс...")
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
print("Индекс загружен!")

print("Подключаю GigaChat...")
giga = GigaChat(
    credentials=os.getenv("GIGA_KEY"),
    verify_ssl_certs=False
)
print("Готово!")

SYSTEM_PROMPT = (
    "You are assistant with the corpus of Van Gogh's letters to brother Theo. "
    "Give answers to user based on texts fragments. "
    "If there is no information tell about it"
)

def summarize(query, docs):
    chunks = "\n\n".join(
        f"Fragment {i+1} ({doc.metadata.get('filename', '')}):\n{doc.page_content}"
        for i, (doc, _score) in enumerate(docs[:3])
    )
    prompt = f"{SYSTEM_PROMPT}\n\nAnswer: {query}\n\n{chunks}"
    try:
        response = giga.chat(prompt)
        return response.choices[0].message.content
    except Exception:
        return None

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>The Letters of Vincent van Gogh | GigaChat</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { color: #2c3e50; }
        input[type=text] { width: 70%; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; background: #3498db; color: white; border: none; cursor: pointer; }
        .summary {
            background: #eaf4fc;
            border-left: 4px solid #3498db;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
            line-height: 1.6;
        }
        .summary-label {
            font-size: 12px;
            color: #7f8c8d;
            margin-top: 10px;
        }
        .result { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 8px; }
        .filename { color: #3498db; font-weight: bold; }
    </style>
</head>
<body>
    <h1>🎬 The Letters of Vincent van Gogh + GigaChat</h1>
    <form method="POST">
        <input type="text" name="query" placeholder="Foe example: Paris, landscape, painting" value="{{ query }}">
        <button type="submit">Найти</button>
    </form>

    {% if summary %}
    <div class="summary">
        <strong>Answer:</strong><br>
        {{ summary }}
        <div class="summary-label">Ответ сгенерирован ИИ на основе найденных фрагментов</div>
    </div>
    {% endif %}

    {% if results %}
    <h2>Найденные фрагменты:</h2>
    {% for doc, score in results %}
    <div class="result">
        <div class="filename">📹 {{ doc.metadata.filename }}</div>
        <p>{{ doc.page_content }}</p>
    </div>
    {% endfor %}
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    query = ""
    results = []
    summary = None
    if request.method == "POST":
        query = request.form.get("query", "")
        if query:
            results = vector_store.similarity_search_with_score(query, k=5)
            if results:
                summary = summarize(query, results)
    return render_template_string(HTML, query=query, results=results, summary=summary)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5002)
