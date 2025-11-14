"""
Historical Events Vector Database Backend
Converted from Jupyter Notebook for Streamlit integration
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os
import json
from typing import List, Dict, Any
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class HistoricalEventsVectorDB:
    """Vector Database for Historical Events with RAG capabilities"""
    
    def __init__(self, groq_api_key: str):
        """Initialize the vector database system"""
        self.model = None
        self.index = None
        self.chunks = []
        self.metadata = []
        self.raw_texts = []
        self.embeddings_matrix = None
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=groq_api_key)
        
        # System prompt for the chatbot
        self.SYSTEM_PROMPT = """You are a knowledgeable historian and AI assistant specializing in historical events. 
You have access to a database of historical events and can provide detailed, accurate information.

Guidelines:
1. Use the provided historical context to answer questions accurately
2. If the context doesn't contain enough information, say so clearly
3. Provide specific dates, names, and details when available
4. Be educational and engaging in your responses
5. If asked about events not in the context, acknowledge the limitation
6. Always cite or reference the historical events you're discussing

Context format: Each historical event will be provided with its content and metadata."""
    
    def load_data(self, data_path: str) -> bool:
        """Load historical events data from text file"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.raw_texts = [line.strip() for line in f if line.strip()]
            
            print(f"Successfully loaded {len(self.raw_texts)} events")
            return True
            
        except FileNotFoundError:
            print(f"File not found: {data_path}")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def process_events(self):
        """Process events without chunking"""
        if not self.raw_texts:
            print("No raw texts available for processing")
            return False
        
        # Use whole events without chunking
        self.chunks = self.raw_texts.copy()
        
        # Create metadata for each event
        self.metadata = []
        for doc_id, text in enumerate(self.chunks):
            self.metadata.append({
                'original_doc_id': doc_id,
                'event_id': doc_id,
                'event_length': len(text),
                'is_whole_event': True
            })
        
        print(f"Event Processing Results:")
        print(f"  Total events: {len(self.chunks)}")
        print(f"  Average event length: {np.mean([len(event) for event in self.chunks]):.0f} chars")
        print(f"  Min event length: {min([len(event) for event in self.chunks])} chars")
        print(f"  Max event length: {max([len(event) for event in self.chunks])} chars")
        
        return True
    
    def initialize_model(self):
        """Initialize the sentence transformer model"""
        try:
            model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            self.model = SentenceTransformer(model_name)
            print(f"Model '{model_name}' loaded successfully!")
            print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            return True
        except Exception as e:
            print(f"Error initializing model: {e}")
            return False
    
    def generate_embeddings(self, batch_size: int = 64):
        """Generate embeddings for all events"""
        if not self.chunks or self.model is None:
            print("No chunks available or model not initialized")
            return False
        
        try:
            all_embeddings = []
            
            for i in range(0, len(self.chunks), batch_size):
                batch_events = self.chunks[i:i+batch_size]
                batch_embeddings = self.model.encode(
                    batch_events,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # for cosine similarity
                )
                all_embeddings.extend(batch_embeddings)
                print(f"  Processed batch {i//batch_size + 1}/{(len(self.chunks)-1)//batch_size + 1}")
            
            # Convert to numpy array
            self.embeddings_matrix = np.array(all_embeddings, dtype=np.float32)
            
            print(f"Embedding generation completed!")
            print(f"Embeddings matrix shape: {self.embeddings_matrix.shape}")
            print(f"Memory usage: ~{self.embeddings_matrix.nbytes / 1024 / 1024:.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return False
    
    def create_faiss_index(self):
        """Create FAISS index for similarity search"""
        if self.embeddings_matrix is None or self.embeddings_matrix.size == 0:
            print("No embeddings available for indexing")
            return False
        
        try:
            dimension = self.embeddings_matrix.shape[1]
            
            # Use IndexFlatIP for cosine similarity (since embeddings are normalized)
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings_matrix)
            
            print(f"FAISS index created successfully!")
            print(f"Dimension: {self.index.d}")
            print(f"Total vectors: {self.index.ntotal}")
            print(f"Index type: IndexFlatIP (Cosine Similarity)")
            
            return True
            
        except Exception as e:
            print(f"Error creating FAISS index: {e}")
            return False
    
    def search_similar_events(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar events (for testing and visualization)"""
        if self.index is None or not self.chunks or self.model is None:
            return []
        
        try:
            query_embedding = self.model.encode([query], normalize_embeddings=True)
            similarities, indices = self.index.search(query_embedding.astype(np.float32), top_k)
            
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx < len(self.chunks):
                    result = {
                        'similarity': float(similarity),
                        'event_index': int(idx),
                        'event_text': self.chunks[idx],
                        'metadata': self.metadata[idx],
                        'event_id': self.metadata[idx]['event_id']
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error searching similar events: {e}")
            return []
    
    def retrieve_relevant_events(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant historical events for RAG (for ChatBot)"""
        if self.index is None or not self.chunks or self.model is None:
            return []
        
        try:
            # Convert query to embedding
            query_embedding = self.model.encode([query], normalize_embeddings=True)
            
            # Search in FAISS index
            similarities, indices = self.index.search(query_embedding.astype(np.float32), top_k)
            
            # Format results
            relevant_events = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx < len(self.chunks) and similarity > 0.1:  # Filter low similarity
                    event_data = {
                        'content': self.chunks[idx],
                        'similarity': float(similarity),
                        'event_id': self.metadata[idx]['event_id'],
                        'event_length': self.metadata[idx]['event_length']
                    }
                    relevant_events.append(event_data)
            
            return relevant_events
            
        except Exception as e:
            print(f"Error retrieving events: {e}")
            return []
    
    def format_context(self, relevant_events: List[Dict]) -> str:
        """Format retrieved events as context for the LLM"""
        if not relevant_events:
            return "No relevant historical events found in the database."
        
        context = "Relevant Historical Events:\n\n"
        for i, event in enumerate(relevant_events, 1):
            context += f"Event {i} (Similarity: {event['similarity']:.3f}):\n"
            context += f"{event['content']}\n\n"
        
        return context
    
    def generate_response(self, user_query: str, relevant_events: List[Dict]) -> str:
        """Generate response using Groq API with RAG context"""
        try:
            # Format context
            context = self.format_context(relevant_events)
            
            # Create messages for the conversation
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"""Context: {context}

Question: {user_query}

Please answer the question using the provided historical context. If the context doesn't contain sufficient information, please acknowledge this limitation."""}
            ]
            
            # Get model configuration from environment variables
            model_name = os.getenv('LLM_MODEL', 'llama-3.3-70b-versatile')
            temperature = float(os.getenv('TEMPERATURE', '0.3'))
            
            # Call Groq API
            response = self.groq_client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=temperature,
                stream=False
            )
            
            answer = response.choices[0].message.content
            return answer
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def chat(self, user_query: str, top_k: int = 3) -> Dict:
        """Main chat function"""
        # Retrieve relevant events
        relevant_events = self.retrieve_relevant_events(user_query, top_k)
        
        # Generate response
        response = self.generate_response(user_query, relevant_events)
        
        return {
            'user_query': user_query,
            'response': response,
            'relevant_events': relevant_events,
            'sources_count': len(relevant_events)
        }
    
    def save_vector_db(self, save_dir: str = "./faiss_db") -> bool:
        """Save FAISS index and related data"""
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            if self.index is not None and self.chunks:
                faiss.write_index(self.index, os.path.join(save_dir, "vector_index.faiss"))
            
                with open(os.path.join(save_dir, "chunks.pkl"), "wb") as f:
                    pickle.dump(self.chunks, f)
                
                with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
                    pickle.dump(self.metadata, f)
                
                with open(os.path.join(save_dir, "original_texts.pkl"), "wb") as f:
                    pickle.dump(self.raw_texts, f)
                
                config = {
                    "model_name": os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
                    "llm_model": os.getenv('LLM_MODEL', 'llama-3.3-70b-versatile'),
                    "temperature": float(os.getenv('TEMPERATURE', '0.3')),
                    "embedding_dimension": self.embeddings_matrix.shape[1] if self.embeddings_matrix.size > 0 else 384,
                    "total_events": len(self.chunks),
                    "total_original_docs": len(self.raw_texts),
                    "uses_whole_events": True,
                    "no_chunking": True
                }
                
                with open(os.path.join(save_dir, "config.json"), "w") as f:
                    json.dump(config, f, indent=2)
                
                print(f"Vector database saved to: {save_dir}")
                return True
            else:
                print("No data to save")
                return False
                
        except Exception as e:
            print(f"Error saving vector database: {e}")
            return False
    
    def load_vector_db(self, save_dir: str = "./faiss_db") -> bool:
        """Load FAISS index and related data"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(os.path.join(save_dir, "vector_index.faiss"))
            
            # Load other data
            with open(os.path.join(save_dir, "chunks.pkl"), "rb") as f:
                self.chunks = pickle.load(f)
            
            with open(os.path.join(save_dir, "metadata.pkl"), "rb") as f:
                self.metadata = pickle.load(f)
            
            with open(os.path.join(save_dir, "original_texts.pkl"), "rb") as f:
                self.raw_texts = pickle.load(f)
            
            with open(os.path.join(save_dir, "config.json"), "r") as f:
                config = json.load(f)
            
            print(f"Vector database loaded from: {save_dir}")
            print(f"Config: {config}")
            
            # Initialize model if not already done
            if self.model is None:
                self.initialize_model()
            
            return True
            
        except Exception as e:
            print(f"Error loading vector database: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if the system is ready for queries"""
        return (self.index is not None and 
                len(self.chunks) > 0 and 
                self.model is not None)
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        if not self.is_ready():
            return {"status": "not_ready"}
        
        return {
            "status": "ready",
            "total_events": len(self.chunks),
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "index_vectors": self.index.ntotal,
            "memory_usage_mb": self.embeddings_matrix.nbytes / 1024 / 1024 if self.embeddings_matrix is not None else 0,
            "average_event_length": np.mean([len(event) for event in self.chunks]),
            "min_event_length": min([len(event) for event in self.chunks]),
            "max_event_length": max([len(event) for event in self.chunks])
        }


# Factory function for easy initialization
def create_historical_events_db(groq_api_key: str, data_path: str = None, load_saved: bool = True) -> HistoricalEventsVectorDB:
    """
    Create and initialize Historical Events Vector Database
    
    Args:
        groq_api_key: API key for Groq
        data_path: Path to the data file (if loading from scratch)
        load_saved: Whether to try loading saved database first
    
    Returns:
        HistoricalEventsVectorDB instance
    """
    # Use environment variables for default configuration
    if data_path is None:
        data_path = os.getenv("DATA_PATH", "descriptions.txt")
    
    db = HistoricalEventsVectorDB(groq_api_key)
    
    # Try to load saved database first
    if load_saved and os.path.exists("./faiss_historical_db"):
        print("Attempting to load saved vector database...")
        if db.load_vector_db("./faiss_historical_db"):
            print("✅ Saved database loaded successfully!")
            return db
        else:
            print("❌ Failed to load saved database, creating new one...")
    
    # Create new database
    if data_path:
        print("Creating new vector database...")
        
        # Load data
        if not db.load_data(data_path):
            print("❌ Failed to load data")
            return db
        
        # Process events
        if not db.process_events():
            print("❌ Failed to process events")
            return db
        
        # Initialize model
        if not db.initialize_model():
            print("❌ Failed to initialize model")
            return db
        
        # Generate embeddings
        if not db.generate_embeddings():
            print("❌ Failed to generate embeddings")
            return db
        
        # Create FAISS index
        if not db.create_faiss_index():
            print("❌ Failed to create FAISS index")
            return db
        
        # Save for future use
        db.save_vector_db("./faiss_historical_db")
        print("✅ New database created and saved successfully!")
    
    return db