# 📚 RAG Multi-Documento - Chatbot Inteligente

Sistema de Recuperación de Información Aumentada (RAG) que permite hacer preguntas sobre múltiples documentos PDF simultáneamente, utilizando embeddings de OpenAI y ChromaDB como base de datos vectorial.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-3.1.1-green)
![LangChain](https://img.shields.io/badge/LangChain-0.3.27-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-purple)

## 🌟 Demo en Vivo

Prueba el sistema aquí: **[https://rag-multi-documento.onrender.com/](https://rag-multi-documento.onrender.com/)**

## 🎯 ¿Qué es este proyecto?

Un sistema RAG (Retrieval-Augmented Generation) que combina la búsqueda semántica en documentos con la capacidad de generación de respuestas de modelos de lenguaje grandes (LLMs). Permite:

- 📄 Subir múltiples documentos PDF
- 🔍 Hacer preguntas sobre uno o varios documentos
- 📊 Obtener respuestas con referencias a las fuentes
- 🎯 Filtrar búsquedas por documentos específicos
- 👁️ Visualizar PDFs almacenados
- 🧠 Memoria conversacional que mantiene el contexto

## 🚀 Características Principales

### ✨ Funcionalidades

- **Procesamiento de PDFs**: Extrae y segmenta texto de documentos PDF
- **Embeddings Vectoriales**: Utiliza OpenAI Embeddings para búsqueda semántica
- **Base de Datos Vectorial**: ChromaDB para almacenamiento persistente
- **Chat Inteligente**: Respuestas contextuales usando GPT-4o-mini
- **Memoria Conversacional**: Recuerda el contexto de preguntas anteriores
- **Búsqueda Selectiva**: Consulta todos los documentos o solo los seleccionados
- **Visualización de PDFs**: Visor integrado de documentos
- **Gestión de Documentos**: Sube, visualiza y elimina documentos fácilmente

### 🎨 Interfaz

- Diseño moderno con gradientes y efectos visuales
- Panel lateral para gestión de documentos
- Área de chat con formato de mensajes
- Indicadores de estado en tiempo real
- Notificaciones tipo toast
- Visualizador de PDF modal

## 🛠️ Tecnologías Utilizadas

### Backend
- **Flask**: Framework web Python
- **LangChain**: Framework para aplicaciones LLM
- **OpenAI**: API para embeddings y generación
- **ChromaDB**: Base de datos vectorial
- **PDFPlumber**: Extracción de texto de PDFs

### Frontend
- **HTML5/CSS3**: Interfaz moderna y responsiva
- **JavaScript Vanilla**: Sin dependencias de frameworks
- **Fetch API**: Comunicación asíncrona con el backend

## 📋 Requisitos Previos

- Python 3.10.0
- Cuenta de OpenAI con API Key
- Navegador web moderno

## 🔧 Instalación Local (Sin Entorno Virtual)

### 1. Clonar el Repositorio

```bash
git clone <tu-repositorio>
cd rag-multi-documento
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
OPENAI_API_KEY=tu_api_key_aqui
```

> ⚠️ **Importante**: Nunca subas tu API key a GitHub. El archivo `.env` está incluido en `.gitignore`.

### 4. Ejecutar la Aplicación

```bash
python app.py
```

La aplicación estará disponible en: `http://localhost:5000`

## 🌐 Despliegue en Render

Este proyecto está configurado para desplegarse en Render automáticamente:

1. Conecta tu repositorio de GitHub con Render
2. Render detectará automáticamente el archivo `render.yaml`
3. Configura la variable de entorno `OPENAI_API_KEY` en el dashboard de Render
4. El despliegue se iniciará automáticamente

## 📖 Cómo Funciona

### Arquitectura del Sistema

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Usuario   │────▶│   Frontend   │────▶│   Backend   │
│             │◀────│   (HTML/JS)  │◀────│   (Flask)   │
└─────────────┘     └──────────────┘     └─────────────┘
                                                │
                                                ▼
                                         ┌─────────────┐
                                         │   OpenAI    │
                                         │  Embeddings │
                                         └─────────────┘
                                                │
                                                ▼
                                         ┌─────────────┐
                                         │  ChromaDB   │
                                         │   Vectores  │
                                         └─────────────┘
```

### Flujo de Procesamiento

1. **Carga de Documento**:
   - Usuario sube un PDF
   - PDFPlumber extrae el texto
   - RecursiveCharacterTextSplitter divide en chunks (1000 chars)
   - OpenAI Embeddings genera vectores
   - ChromaDB almacena chunks con metadatos

2. **Consulta**:
   - Usuario hace una pregunta
   - Sistema genera embedding de la pregunta
   - ChromaDB busca chunks más relevantes (k=6)
   - ConversationalRetrievalChain genera respuesta
   - Se incluye contexto conversacional previo

3. **Respuesta Inteligente**:
   - Si hay info en documentos → Respuesta con fuentes
   - Si no hay info → Consulta directa al modelo GPT-4o-mini
   - Memoria conversacional mantiene contexto

## 📁 Estructura del Proyecto

```
rag-multi-documento/
│
├── app.py                      # Backend Flask principal
├── requirements.txt            # Dependencias Python
├── render.yaml                # Configuración Render
├── .env                       # Variables de entorno (NO SUBIR)
├── .gitignore                 # Archivos ignorados por Git
├── .python-version            # Versión Python (3.10.0)
│
├── templates/
│   └── index.html             # Frontend único
│
├── chroma_db_multi/           # Base de datos vectorial
├── stored_pdfs/               # PDFs almacenados
├── TextoEstraido/             # Textos extraídos de PDFs
└── documents_metadata.json    # Metadatos de documentos
```

## 🎮 Uso del Sistema

### 1. Subir Documentos

1. Haz clic en "📎 Seleccionar PDF"
2. Elige un archivo PDF
3. Presiona "🚀 Procesar y Entrenar"
4. Espera la confirmación ✅

### 2. Hacer Preguntas

1. Selecciona modo de búsqueda:
   - 🌐 **Todos los documentos**: Busca en toda la biblioteca
   - 🎯 **Documentos seleccionados**: Solo en los marcados

2. Escribe tu pregunta en el chat
3. Presiona "📤 Enviar" o Enter

### 3. Gestionar Documentos

- **Ver PDF**: Clic en 👁️ para visualizar
- **Eliminar**: Clic en 🗑️ para borrar
- **Seleccionar**: Checkbox para búsquedas filtradas

### 4. Memoria Conversacional

- El sistema recuerda el contexto de la conversación
- Puedes hacer preguntas de seguimiento
- Usa "🧹 Limpiar Memoria" para reiniciar el contexto

## 🔍 Ejemplos de Uso

### Pregunta Simple
```
Usuario: "¿Cuáles son los objetivos principales del proyecto?"
Bot: Responde basándose en los documentos + fuentes citadas
```

### Pregunta Conversacional
```
Usuario: "¿Cuál es el presupuesto del proyecto?"
Bot: "El presupuesto total es de $500,000..."

Usuario: "¿Y cuánto se ha gastado hasta ahora?"
Bot: [Recuerda el contexto del presupuesto] "Se ha gastado $350,000..."
```

### Pregunta sin Información
```
Usuario: "¿Qué es la teoría de cuerdas?"
Bot: 📚 No encontré información específica en los documentos...
     🤖 Sin embargo, puedo responder con mi conocimiento general: [...]
```

## ⚙️ Configuración Avanzada

### Ajustar Parámetros RAG

En `app.py`, puedes modificar:

```python
# Modelo de chat
OPENAI_CHAT_MODEL = "gpt-4o-mini"  # O "gpt-4", "gpt-3.5-turbo"

# Tamaño de chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Caracteres por chunk
    chunk_overlap=200     # Overlap entre chunks
)

# Número de chunks recuperados
def query_documents(question, k=6):  # k = número de chunks
```

### Memoria Conversacional

```python
# Ajustar ventana de memoria
context = "\n".join([f"{msg.type}: {msg.content}" 
                     for msg in chat_history[-4:]])  # Últimos 4 mensajes
```

## 🐛 Solución de Problemas

### Error: "No se detectó API key"
- Verifica que el archivo `.env` existe
- Confirma que `OPENAI_API_KEY` está configurado
- Reinicia la aplicación

### Error al procesar PDF
- Verifica que el PDF no esté protegido
- Asegúrate que el PDF contiene texto (no solo imágenes)
- Revisa los logs para más detalles

### Base de datos vectorial corrupta
```bash
# Eliminar y recrear
rm -rf chroma_db_multi/
python app.py  # Se recrea automáticamente
```

## 📊 Rendimiento

- **Procesamiento**: ~5-10 segundos por PDF (depende del tamaño)
- **Respuestas**: ~2-5 segundos (depende de API OpenAI)
- **Capacidad**: Ilimitados documentos (limitado por disco/RAM)

## 🔐 Seguridad

- ✅ Variables de entorno para API keys
- ✅ `.gitignore` configurado
- ✅ CORS habilitado para desarrollo
- ⚠️ En producción: configurar CORS específicos
- ⚠️ Implementar autenticación si es público

## 🤝 Contribuciones

Las contribuciones son bienvenidas:

1. Fork el proyecto
2. Crea tu rama (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 👨‍💻 Autor

Desarrollado con ❤️ para facilitar la búsqueda y análisis de documentos mediante IA.

## 🙏 Agradecimientos

- [LangChain](https://langchain.com/) - Framework RAG
- [OpenAI](https://openai.com/) - Embeddings y LLM
- [ChromaDB](https://www.trychroma.com/) - Base de datos vectorial
- [Flask](https://flask.palletsprojects.com/) - Framework web

---

⭐ Si te gusta este proyecto, dale una estrella en GitHub!