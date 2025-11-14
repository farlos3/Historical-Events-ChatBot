# 📚 Historical Events ChatBot

A Streamlit-based AI chatbot that uses FAISS vector database and Groq's Llama model to answer questions about historical events through Retrieval-Augmented Generation (RAG).

## 🚀 Features

- **Vector Search**: FAISS-powered similarity search for relevant historical events
- **RAG Pipeline**: Combines vector search with LLM for contextual responses
- **Interactive Chat**: Streamlit web interface with chat history
- **Environment Variables**: Easy configuration through .env files
- **Cloud Ready**: Deploy to Streamlit Cloud, Heroku, Railway, and more

## 🛠️ Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy the template and configure:
```bash
cp .env.template .env
```

Edit `.env` with your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run the App
```bash
streamlit run app.py
```

## ⚙️ Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *Required* | Your Groq API key from [console.groq.com](https://console.groq.com/) |
| `DATA_PATH` | `descriptions.txt` | Historical events data file |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformers embedding model |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq LLM model |
| `TEMPERATURE` | `0.3` | Response creativity (0.0-1.0) |
| `TOP_K` | `5` | Default sources to retrieve |
| `APP_TITLE` | `Historical Events ChatBot` | Application title |
| `PAGE_ICON` | `📚` | Page icon |

- 🔍 **Vector Search**: Semantic similarity search using FAISS and SentenceTransformers
- 🤖 **AI ChatBot**: RAG-powered conversational AI using Groq LLM
- 📊 **Interactive Interface**: Clean Streamlit web interface
- 💾 **Persistent Storage**: Save/load vector database for quick startup
- 📈 **Real-time Analytics**: Database statistics and performance metrics

## Project Structure

```
Project/
│
├── backend.py              # Core vector database backend
├── app.py                  # Streamlit web interface
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── Data/
│   └── descriptions.txt   # Historical events data
└── faiss_historical_db/   # Saved vector database (auto-generated)
    ├── vector_index.faiss
    ├── chunks.pkl
    ├── metadata.pkl
    ├── original_texts.pkl
    └── config.json
```

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   - Place your historical events data in `Data/descriptions.txt`
   - Each line should contain one historical event description

3. **Configure API Key**:
   - Get a Groq API key from [Groq Console](https://console.groq.com/)
   - Update the API key in both `backend.py` and `app.py`

## Usage

### Running the Streamlit App

```bash
# Run Streamlit app
streamlit run app.py

# Alternative if streamlit command not found
python -m streamlit run app.py
```

### Using the Backend Directly

```python
from backend import create_historical_events_db

# Initialize database
db = create_historical_events_db(
    groq_api_key="your_groq_api_key",
    data_path="Data/descriptions.txt",
    load_saved=True
)

# Search for similar events
results = db.search_similar_events("World War I", top_k=5)

# Chat with AI
response = db.chat("What caused World War I?", top_k=3)
print(response['response'])
```

## Web Interface Features

### 🔍 Event Search
- Semantic similarity search across historical events
- Adjustable top-k results
- Similarity scores and detailed event information

### 💬 AI ChatBot
- RAG-powered responses using historical context
- Source citation and transparency
- Response time tracking

### 📊 Database Statistics
- Total events count
- Embedding dimensions and memory usage
- Event length statistics

### ⚙️ Customizable Settings
- Search parameters (top-k)
- Model configuration display
- Chat history management

## Technical Architecture

### Backend (`backend.py`)
- **HistoricalEventsVectorDB**: Main database class
- **SentenceTransformers**: all-MiniLM-L6-v2 for embeddings
- **FAISS**: IndexFlatIP for cosine similarity search
- **Groq**: Llama-3.3-70B-Versatile for LLM responses

### Frontend (`app.py`)
- **Streamlit**: Web interface framework
- **Caching**: Efficient database initialization
- **Real-time**: Live search and chat capabilities
- **Responsive**: Multi-column layout

## Configuration

### Model Settings
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **LLM Model**: Llama-3.3-70B-Versatile
- **Temperature**: 0.3 (balanced creativity/accuracy)
- **Similarity Threshold**: 0.1 (filter low relevance)

### Performance Settings
- **Batch Size**: 64 (embedding generation)
- **Top-K Search**: 1-10 (configurable)
- **Chat Context**: 1-10 events (configurable)

## Data Format

Your `descriptions.txt` should contain one historical event per line:
```
The assassination of Archduke Franz Ferdinand on June 28, 1914, sparked World War I...
The Russian Revolution of 1917 overthrew the Tsarist regime and led to the rise of the Soviet Union...
The Treaty of Versailles in 1919 officially ended World War I and imposed heavy reparations on Germany...
```

## Troubleshooting

### Common Issues

1. **Streamlit command not found**:
   ```bash
   python -m streamlit run app.py
   ```

2. **FAISS installation issues**:
   ```bash
   pip install faiss-cpu --no-cache-dir
   ```

3. **Memory issues with large datasets**:
   - Reduce batch_size in embedding generation
   - Use FAISS index with compression

4. **Groq API errors**:
   - Check your API key is valid
   - Verify rate limits
   - Check internet connection

### Performance Tips

1. **First Run**: Database creation may take 2-5 minutes
2. **Subsequent Runs**: Saved database loads in seconds
3. **Large Datasets**: Consider using GPU-accelerated FAISS
4. **Memory Usage**: Monitor with database statistics

## Development

### Adding New Features
1. Extend `HistoricalEventsVectorDB` class in `backend.py`
2. Add UI components in `app.py`
3. Update requirements if needed

### Testing
```python
# Test backend functionality
python backend.py

# Test with different queries
db.chat("Your test question", top_k=5)
```

## License

This project is for educational purposes (CSS371 NLP Course).

## Credits

- **FAISS**: Facebook AI Similarity Search
- **SentenceTransformers**: Hugging Face
- **Streamlit**: Web framework
- **Groq**: LLM API service
