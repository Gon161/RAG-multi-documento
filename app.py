from flask import Flask, request, jsonify, render_template, send_from_directory, session
from flask_cors import CORS
import os, io, base64, json
from datetime import datetime

import pdfplumber
from dotenv import load_dotenv

# LangChain + OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory

# =========================
# CONFIG
# =========================
load_dotenv()

OPENAI_CHAT_MODEL = "gpt-4o-mini"
CHROMA_DB_PATH = "./chroma_db_multi"
COLLECTION_NAME = "all_documents"
DOCUMENTS_FILE = "documents_metadata.json"
PDF_STORAGE_PATH = "./stored_pdfs"

app = Flask(__name__, template_folder="templates")
app.secret_key = os.urandom(24)  # Necesario para sesiones
CORS(app, supports_credentials=True)

documents_store = {}
conversation_memories = {}  # Almacenar memoria por sesión

# =========================
# UTILS
# =========================
def load_documents_store():
    global documents_store
    if os.path.exists(DOCUMENTS_FILE):
        with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
            documents_store = json.load(f)

def save_documents_store():
    with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(documents_store, f, ensure_ascii=False, indent=2)

def extract_text_from_pdf(pdf_base64: str) -> str:
    pdf_bytes = base64.b64decode(pdf_base64)
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            content = page.extract_text()
            if content:
                text += f"\n[Página {i}]\n{content}\n"
    return text

# =========================
# DOCUMENT PROCESSING
# =========================
def process_document(doc_id, file_name, pdf_bytes):
    # Convertir bytes a base64 para extracción de texto
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
    text = extract_text_from_pdf(pdf_base64)
    
    # Guardar texto extraído
    os.makedirs("TextoEstraido", exist_ok=True)
    with open(f"TextoEstraido/{file_name}.txt", "w", encoding="utf-8") as file:
        file.write(text)
    
    # Guardar PDF original con ID
    pdf_filename = f"{doc_id}_{file_name}"
    pdf_path = os.path.join(PDF_STORAGE_PATH, pdf_filename)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    
    # Procesar chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file_name, "doc_id": doc_id, "chunk": i} for i in range(len(chunks))]

    db.add_texts(texts=chunks, metadatas=metadatas, ids=ids)

    documents_store[doc_id] = {
        "fileName": file_name,
        "uploadDate": datetime.now().isoformat(),
        "chunkCount": len(chunks),
        "pdfPath": pdf_filename
    }
    save_documents_store()

# =========================
# QUERY CON MEMORIA
# =========================
def get_session_id():
    """Obtener o crear ID de sesión"""
    if 'session_id' not in session:
        session['session_id'] = str(int(datetime.now().timestamp() * 1000))
    return session['session_id']

def get_or_create_memory(session_id):
    """Obtener o crear memoria para una sesión"""
    if session_id not in conversation_memories:
        conversation_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    return conversation_memories[session_id]

def query_documents(question, k=6, doc_ids=None):
    session_id = get_session_id()
    memory = get_or_create_memory(session_id)
    
    embeddings = OpenAIEmbeddings()
    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )

    search_kwargs = {"k": k}
    if doc_ids and len(doc_ids) > 0:
        search_kwargs["filter"] = {"doc_id": {"$in": doc_ids}}

    retriever = db.as_retriever(search_kwargs=search_kwargs)
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.3)
    
    # Usar ConversationalRetrievalChain en lugar de RetrievalQA
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )

    result = qa({"question": question})
    answer = result["answer"]
    
    # Detectar si el modelo no sabe la respuesta basándose en los documentos
    uncertainty_phrases = [
        "no sé",
        "no lo sé",
        "no tengo información",
        "no puedo encontrar",
        "no está disponible",
        "no hay información",
        "no se menciona",
        "no se proporciona",
        "desconozco",
        "no conozco",
        "no dispongo"
    ]
    
    answer_lower = answer.lower()
    is_uncertain = any(phrase in answer_lower for phrase in uncertainty_phrases)
    
    # Si no encontró respuesta en los documentos, consultar al modelo directamente
    if is_uncertain:
        llm_direct = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.7)
        
        # Incluir historial de conversación en la consulta directa
        chat_history = memory.load_memory_variables({}).get("chat_history", [])
        context = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history[-4:]])  # Últimos 4 mensajes
        
        prompt = f"""Historial de conversación reciente:
{context}

Pregunta actual: {question}

Por favor, responde basándote en tu conocimiento general y el contexto de la conversación."""
        
        direct_response = llm_direct.invoke(prompt)
        answer = f"📚 No encontré información específica en los documentos proporcionados.\n\n🤖 Sin embargo, puedo responder basándome en mi conocimiento general:\n\n{direct_response.content}"
        sources = []
    else:
        sources = [{
            "document": d.metadata.get("source"),
            "chunk": d.metadata.get("chunk", 0),
            "content_preview": d.page_content[:200]
        } for d in result["source_documents"]]

    return {"answer": answer, "sources": sources}

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/health")
def health():
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    
    return jsonify({
        "success": True,
        "documents": len(documents_store),
        "chat_model": OPENAI_CHAT_MODEL,
        "config": {
            "hasApiKey": has_api_key
        }
    })

@app.route("/api/documents")
def documents():
    docs = [{"id": k, **v} for k, v in documents_store.items()]
    return jsonify({"success": True, "documents": docs, "total": len(docs)})

@app.route("/api/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"success": False, "error": "No se recibió archivo"})

    # Generar ID único basado en timestamp para evitar conflictos
    import time
    doc_id = str(int(time.time() * 1000))  # Timestamp en milisegundos
    file_name = file.filename
    pdf_bytes = file.read()

    try:
        process_document(doc_id, file_name, pdf_bytes)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        question = data.get("question")
        doc_ids = data.get("docs", [])
        
        result = query_documents(question, doc_ids=doc_ids if doc_ids else None)
        
        return jsonify({
            "success": True,
            "answer": result["answer"],
            "sources": result["sources"]
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route("/api/clear-memory", methods=["POST"])
def clear_memory():
    """Endpoint para limpiar la memoria de conversación"""
    try:
        session_id = get_session_id()
        if session_id in conversation_memories:
            del conversation_memories[session_id]
        return jsonify({"success": True, "message": "Memoria limpiada correctamente"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/delete-document", methods=["DELETE"])
def delete_document():
    try:
        data = request.json
        doc_id = data.get("docId")
        
        if not doc_id or doc_id not in documents_store:
            return jsonify({"success": False, "error": "Documento no encontrado"})
        
        # Eliminar PDF físico
        pdf_filename = documents_store[doc_id].get("pdfPath")
        if pdf_filename:
            pdf_path = os.path.join(PDF_STORAGE_PATH, pdf_filename)
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
        
        # Eliminar chunks de ChromaDB
        embeddings = OpenAIEmbeddings()
        db = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        
        chunk_count = documents_store[doc_id]["chunkCount"]
        ids_to_delete = [f"{doc_id}_{i}" for i in range(chunk_count)]
        
        db.delete(ids=ids_to_delete)
        
        # Eliminar del store
        del documents_store[doc_id]
        save_documents_store()
        
        return jsonify({"success": True, "message": "Documento eliminado correctamente"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/view-pdf/<doc_id>")
def view_pdf(doc_id):
    """Endpoint para visualizar el PDF"""
    if doc_id not in documents_store:
        return jsonify({"success": False, "error": "Documento no encontrado"}), 404
    
    pdf_filename = documents_store[doc_id].get("pdfPath")
    if not pdf_filename:
        return jsonify({"success": False, "error": "PDF no encontrado"}), 404
    
    return send_from_directory(PDF_STORAGE_PATH, pdf_filename)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
    os.makedirs("TextoEstraido", exist_ok=True)
    load_documents_store()

    print("="*80)
    print("🚀 RAG MULTI-DOCUMENTO - OPENAI")
    print(f"🤖 Chat: {OPENAI_CHAT_MODEL}")
    print("📦 Vector DB: Chroma (persistencia automática)")
    print(f"📄 PDFs almacenados en: {PDF_STORAGE_PATH}")
    print("="*80)

    # Usar puerto del entorno o 5000 por defecto
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', debug=False, port=port)