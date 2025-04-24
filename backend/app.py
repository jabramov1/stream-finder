import json
import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import time
from collections import defaultdict
import re
import logging
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors
from sklearn.preprocessing import normalize

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set ROOT_PATH for linking files
os.environ["ROOT_PATH"] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script (backend folder)
current_directory = os.path.dirname(os.path.abspath(__file__))

# ===== Configuration =====
# Choose a smaller Word2Vec model
MODEL_NAME = "glove-wiki-gigaword-300"  # or can use the much smaller glove-twitter-25 (must also change the below vector size as well to 25)
VECTOR_SIZE = 300  # The dimension of this model

# Paths for stored data
MODELS_DIR = os.path.join(current_directory, "models", "word2vec")
WIKI_VECTORS_PATH = os.path.join(MODELS_DIR, f"wiki_doc_vectors_{VECTOR_SIZE}.npy")
WIKI_LOOKUP_PATH = os.path.join(MODELS_DIR, f"wiki_doc_lookup_{VECTOR_SIZE}.pkl")
MODEL_INFO_PATH = os.path.join(MODELS_DIR, "model_info.json")
BOOLEAN_INDEX_PATH = os.path.join(MODELS_DIR, "boolean_index.pkl")

# Search weights
WIKI_SEMANTIC_WEIGHT = 0.5
BOOLEAN_WEIGHT = 0.5
TOP_K = 50  # Number of results to return from each search method

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# ===== Load Data =====
# Define models directory
models_dir = os.path.join(current_directory, "models")

# Specify the path to the JSON file (init.json) in the backend folder
json_path = os.path.join(current_directory, "init.json")

try:
    # Load the JSON data with UTF-8 encoding (still needed for streamer info)
    with open(json_path, "r", encoding="utf-8") as file:
        combined_data = json.load(file)

    # Extract the individual datasets (needed for document details)
    reddit_data = combined_data["reddit"]
    twitter_data = combined_data["twitter"]
    wiki_data = combined_data["wiki"]
    details_data = combined_data["details"]
except Exception as e:
    logger.error(f"Failed to load init.json: {e}")
    reddit_data = {}
    twitter_data = {}
    wiki_data = {}
    details_data = {}

try:
    # Load CSV data about streamers for additional details
    csv_path = os.path.join(current_directory, "streamer_details.csv")
    streamer_csv = pd.read_csv(csv_path).fillna("")  # Safely fill NaNs with empty strings

    # Convert CSV rows into a dict keyed by uppercase Name
    streamer_csv_data = {}
    for _, row in streamer_csv.iterrows():
        name_upper = str(row["Name"]).upper().strip()
        streamer_csv_data[name_upper] = dict(row)
except Exception as e:
    logger.error(f"Failed to load streamer_details.csv: {e}")
    streamer_csv_data = {}

# ===== Word2Vec Wiki Search =====
class LightWord2VecSearch:
    def __init__(self, model_name=MODEL_NAME, vector_size=VECTOR_SIZE):
        """Initialize the Word2Vec search engine with a pre-trained model"""
        self.model_name = model_name
        self.vector_size = vector_size
        self.word2vec_model = None
        self.wiki_docs = []
        self.doc_vectors = None
        self.doc_lookup = {}  # Maps index to original document data
        
    def load_pretrained_model(self):
        """Load a pre-trained Word2Vec model from Gensim"""
        logger.info(f"Loading pre-trained Word2Vec model: {self.model_name}")
        start_time = time.time()
        
        try:
            # Import gensim.downloader only when needed
            try:
                import gensim.downloader as api
                self.word2vec_model = api.load(self.model_name)
                logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
                return True
            except ImportError as e:
                logger.error(f"Error importing gensim.downloader: {e}")
                # Try creating a simple word vector model from scratch
                self._create_simple_model()
                return self.word2vec_model is not None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to simple model
            self._create_simple_model()
            return self.word2vec_model is not None
            
    def _create_simple_model(self):
        """Create a very simple word vector model from scratch using the documents"""
        logger.info("Creating a simple word vector model from scratch")
        
        try:
            # Extract documents
            all_docs = []
            
            # Add wiki documents
            if isinstance(wiki_data, dict):
                for streamer, entry in wiki_data.items():
                    if isinstance(entry, dict) and "content" in entry:
                        all_docs.append(entry["content"])
            elif isinstance(wiki_data, list):
                for entry in wiki_data:
                    if isinstance(entry, dict) and "content" in entry:
                        all_docs.append(entry["content"])
            
            # Add other text sources for better vocabulary
            for streamer, posts in reddit_data.items():
                if isinstance(posts, list):
                    for post in posts:
                        if isinstance(post, dict) and "Title" in post:
                            all_docs.append(post["Title"])
            
            # Tokenize documents
            all_tokens = []
            for doc in all_docs:
                tokens = re.findall(r"\w+", doc.lower())
                all_tokens.extend(tokens)
            
            # Count token frequencies
            token_freq = {}
            for token in all_tokens:
                if token in token_freq:
                    token_freq[token] += 1
                else:
                    token_freq[token] = 1
            
            # Keep only tokens that appear at least twice
            vocab = {token: idx for idx, (token, freq) in enumerate(
                sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
            ) if freq >= 2}
            
            # Create random vectors for each token
            vectors = np.random.randn(len(vocab), self.vector_size).astype(np.float32)
            vectors = normalize(vectors)
            
            # Create a simple Word2VecKeyedVectors object
            model = Word2VecKeyedVectors(self.vector_size)
            model.add_vectors(list(vocab.keys()), vectors)
            
            self.word2vec_model = model
            logger.info(f"Created simple model with {len(vocab)} tokens")
            return True
        except Exception as e:
            logger.error(f"Error creating simple model: {e}")
            return False
    
    def extract_wiki_documents(self, wiki_data):
        """Extract Wikipedia documents from the data"""
        logger.info("Extracting Wikipedia documents")
        
        try:
            doc_idx = 0
            if isinstance(wiki_data, dict):
                # Handle dictionary format
                for streamer, entry in wiki_data.items():
                    if isinstance(entry, dict) and "content" in entry:
                        self.wiki_docs.append(entry["content"])
                        self.doc_lookup[doc_idx] = {
                            "source": "wiki", 
                            "streamer": streamer, 
                            "text": entry["content"]
                        }
                        doc_idx += 1
            elif isinstance(wiki_data, list):
                # Handle list format
                for idx, entry in enumerate(wiki_data):
                    if isinstance(entry, dict) and "content" in entry and "streamer" in entry:
                        self.wiki_docs.append(entry["content"])
                        self.doc_lookup[doc_idx] = {
                            "source": "wiki", 
                            "streamer": entry["streamer"], 
                            "text": entry["content"]
                        }
                        doc_idx += 1
            
            logger.info(f"Extracted {len(self.wiki_docs)} Wikipedia documents")
            return True
        except Exception as e:
            logger.error(f"Error extracting Wikipedia documents: {e}")
            return False
    
    def compute_document_vectors(self):
        """Compute document vectors by averaging word vectors"""
        if not self.word2vec_model or not self.wiki_docs:
            logger.error("Model or documents not loaded")
            return False
        
        logger.info("Computing document vectors...")
        start_time = time.time()
        
        # Initialize document vectors array
        self.doc_vectors = np.zeros((len(self.wiki_docs), self.vector_size), dtype=np.float32)
        
        # Compute document vectors by averaging word vectors
        for i, doc in enumerate(self.wiki_docs):
            words = re.findall(r"\w+", doc.lower())
            vectors = []
            
            for word in words:
                try:
                    if word in self.word2vec_model:
                        vectors.append(self.word2vec_model[word])
                except Exception as e:
                    continue  # Skip words that cause errors
            
            if vectors:
                doc_vector = np.mean(vectors, axis=0)
                self.doc_vectors[i] = doc_vector
        
        # Normalize document vectors for cosine similarity
        self.doc_vectors = normalize(self.doc_vectors)
        
        logger.info(f"Document vectors computed in {time.time() - start_time:.2f} seconds")
        return True
    
    def save_model(self, directory):
        """Save the computed document vectors, lookup table, and model info"""
        os.makedirs(directory, exist_ok=True)
        
        # Save document vectors with dimension in filename
        vectors_path = os.path.join(directory, f"wiki_doc_vectors_{self.vector_size}.npy")
        np.save(vectors_path, self.doc_vectors)
        
        # Save document lookup table with dimension in filename
        lookup_path = os.path.join(directory, f"wiki_doc_lookup_{self.vector_size}.pkl")
        with open(lookup_path, "wb") as f:
            pickle.dump(self.doc_lookup, f)
        
        # Save model info
        model_info = {
            "model_name": self.model_name,
            "vector_size": self.vector_size,
            "doc_count": len(self.wiki_docs),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "vectors_path": vectors_path,
            "lookup_path": lookup_path
        }
        
        with open(os.path.join(directory, "model_info.json"), "w") as f:
            json.dump(model_info, f)
        
        logger.info(f"Model components saved to {directory} with dimension {self.vector_size}")
        return True
    
    def load_model(self, directory):
        """Load pre-computed document vectors and lookup table"""
        # Construct paths with vector dimension
        vectors_path = os.path.join(directory, f"wiki_doc_vectors_{self.vector_size}.npy")
        lookup_path = os.path.join(directory, f"wiki_doc_lookup_{self.vector_size}.pkl")
        
        logger.info(f"Loading model components from {directory}")
        start_time = time.time()
        
        try:
            # Load document vectors
            self.doc_vectors = np.load(vectors_path)
            
            # Load document lookup table
            with open(lookup_path, "rb") as f:
                self.doc_lookup = pickle.load(f)
            
            logger.info(f"Model components loaded in {time.time() - start_time:.2f} seconds")
            
            # Check if dimensions match
            if self.doc_vectors.shape[1] != self.vector_size:
                logger.warning(f"Dimension mismatch: loaded vectors have dimension {self.doc_vectors.shape[1]} but expected {self.vector_size}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error loading model components: {e}")
            return False
    
    def check_model_compatibility(self):
        """Check if any existing model is compatible with current configuration"""
        model_info_path = os.path.join(MODELS_DIR, "model_info.json")
        
        # Check if model info exists
        if os.path.exists(model_info_path):
            try:
                with open(model_info_path, "r") as f:
                    model_info = json.load(f)
                
                stored_size = model_info.get("vector_size")
                stored_name = model_info.get("model_name")
                
                logger.info(f"Found existing model: {stored_name} with {stored_size} dimensions")
                logger.info(f"Current configuration: {self.model_name} with {self.vector_size} dimensions")
                
                if stored_size == self.vector_size and stored_name == self.model_name:
                    logger.info("Model configuration matches")
                    return True
                else:
                    logger.info("Model configuration doesn't match")
                    return False
            except Exception as e:
                logger.error(f"Error checking model info: {e}")
                return False
        else:
            logger.info("No existing model info found")
            return False
    
    def rebuild_model_if_needed(self):
        """Check if model needs to be rebuilt and rebuild if necessary"""
        vectors_path = os.path.join(MODELS_DIR, f"wiki_doc_vectors_{self.vector_size}.npy")
        lookup_path = os.path.join(MODELS_DIR, f"wiki_doc_lookup_{self.vector_size}.pkl")
        
        # Check if model files with correct dimensions already exist
        if os.path.exists(vectors_path) and os.path.exists(lookup_path):
            logger.info(f"Found model files with matching dimension {self.vector_size}")
            return self.load_model(MODELS_DIR)
        
        # Check if any model exists but with wrong dimensions
        if not self.check_model_compatibility():
            logger.info("No matching model found, rebuilding...")
            
            # Load word2vec model
            if not self.word2vec_model and not self.load_pretrained_model():
                logger.error("Failed to load word2vec model")
                return False
            
            # Extract documents
            if not self.wiki_docs and not self.extract_wiki_documents(wiki_data):
                logger.error("Failed to extract wiki documents")
                return False
            
            # Compute document vectors
            if not self.compute_document_vectors():
                logger.error("Failed to compute document vectors")
                return False
            
            # Save model with updated info
            return self.save_model(MODELS_DIR)
        
        return True
    
    def query(self, query_text, top_k=TOP_K):
        """Find the most similar documents to the query"""
        if self.doc_vectors is None:
            logger.error("Document vectors not loaded")
            return []
        
        if self.word2vec_model is None:
            logger.error("Word2Vec model not loaded")
            return []
        
        start_time = time.time()
        
        # Get all the words from the query
        words = re.findall(r"\w+", query_text.lower())
        vectors = []
        
        for word in words:
            try:
                if word in self.word2vec_model:
                    vectors.append(self.word2vec_model[word])
            except Exception as e:
                logger.warning(f"Error getting vector for word '{word}': {e}")
                continue
        
        if not vectors:
            # Fallback to boolean matching if no vectors
            logger.warning("No word vectors found in query, falling back to boolean matching")
            return self._boolean_fallback(query_text, top_k)
        
        # Compute average query vector
        query_vector = np.mean(vectors, axis=0)
        
        # Normalize query vector
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # Debug dimensions
        logger.info(f"Query vector dimension: {query_vector.shape}, Document vectors dimension: {self.doc_vectors.shape}")
        
        # Compute cosine similarity with all documents
        similarities = np.dot(self.doc_vectors, query_vector)
        
        # Get top-k most similar document indices
        top_indices = np.argsort(-similarities)[:top_k]
        
        results = []
        for idx in top_indices:
            if idx >= len(self.doc_lookup):
                continue
                
            doc_info = self.doc_lookup[idx]
            similarity_score = float(similarities[idx])
            
            # Skip very low similarity scores
            if similarity_score < 0.1:
                continue
                
            results.append({
                "source": doc_info["source"],
                "name": doc_info["streamer"],
                "doc": doc_info["text"][:150] + "..." if len(doc_info["text"]) > 150 else doc_info["text"],
                "sim_score": round(similarity_score * 100, 2),
                "semantic_score": round(similarity_score * 100, 2)
            })
        
        # If we got no results, fall back to boolean matching
        if not results:
            logger.warning("No semantic results found, falling back to boolean matching")
            return self._boolean_fallback(query_text, top_k)
            
        logger.info(f"Query processed in {time.time() - start_time:.4f} seconds")
        return results
        
    def _boolean_fallback(self, query_text, top_k=TOP_K):
        """Simple boolean search fallback if vector search fails"""
        results = []
        query_terms = set(re.findall(r"\w+", query_text.lower()))
        
        for idx, doc_info in self.doc_lookup.items():
            text = doc_info["text"].lower()
            term_matches = sum(1 for term in query_terms if term in text)
            
            if term_matches > 0:
                score = term_matches * 10
                # Extra points for exact phrase match
                if query_text.lower() in text:
                    score += 20
                    
                results.append({
                    "source": doc_info["source"],
                    "name": doc_info["streamer"],
                    "doc": doc_info["text"][:150] + "..." if len(doc_info["text"]) > 150 else doc_info["text"],
                    "sim_score": score,
                    "semantic_score": score
                })
        
        # Sort by score
        results.sort(key=lambda x: x["sim_score"], reverse=True)
        return results[:top_k]

# ===== Boolean Search =====
def create_boolean_index():
    """Create an inverted index for boolean search"""
    index = defaultdict(list)
    
    # Index Reddit data
    for streamer, posts in reddit_data.items():
        if isinstance(posts, list):
            for i, post in enumerate(posts):
                if isinstance(post, dict) and "Title" in post:
                    words = re.findall(r"\w+", post["Title"].lower())
                    for w in words: 
                        index[w].append(("reddit", streamer, i))
    
    # Index Twitter data
    for streamer, tweets in twitter_data.items():
        if isinstance(tweets, list):
            for i, tweet in enumerate(tweets):
                words = re.findall(r"\w+", str(tweet).lower())
                for w in words: 
                    index[w].append(("twitter", streamer, i))
    
    # Index details data
    for streamer, details in details_data.items():
        if isinstance(details, dict):
            description = str(details.get("Description", ""))
            words = re.findall(r"\w+", description.lower())
            for w in words: 
                index[w].append(("details", streamer, 0))
    
    return index

def boolean_search(query, index, top_k=TOP_K):
    """Perform boolean search using the inverted index"""
    if index is None:
        logger.error("Boolean index is None, cannot perform search")
        return []
        
    terms = re.findall(r"\w+", query.strip().lower())
    if not terms: 
        return []
    
    doc_matches = defaultdict(int)
    doc_info = {}
    
    for term in terms:
        if term in index:
            for doc_ref in index[term]:
                source, streamer, idx = doc_ref
                doc_id = f"{source}:{streamer}:{idx}"
                doc_matches[doc_id] += 1
                
                if doc_id not in doc_info:
                    try:
                        if source == "reddit" and streamer in reddit_data and len(reddit_data[streamer]) > idx:
                            post = reddit_data[streamer][idx]
                            text = post.get("Title", "")
                            score = post.get("Score", 1)
                            doc_info[doc_id] = {
                                "source": source,
                                "name": streamer,
                                "doc": text,
                                "score": score,
                                "term_matches": 0
                            }
                        elif source == "twitter" and streamer in twitter_data and len(twitter_data[streamer]) > idx:
                            text = str(twitter_data[streamer][idx])
                            doc_info[doc_id] = {
                                "source": source,
                                "name": streamer,
                                "doc": text,
                                "score": 1,
                                "term_matches": 0
                            }
                        elif source == "details" and streamer in details_data:
                            text = str(details_data[streamer].get("Description", ""))
                            doc_info[doc_id] = {
                                "source": source,
                                "name": streamer,
                                "doc": text,
                                "score": 1,
                                "term_matches": 0
                            }
                    except Exception as e:
                        logger.error(f"Error processing document {doc_id}: {e}")
                        continue
    
    # Update term match counts
    for doc_id, count in doc_matches.items():
        if doc_id in doc_info:
            doc_info[doc_id]["term_matches"] = count
    
    # Score and sort results
    results = []
    for doc_id, info in doc_info.items():
        # Calculate a relevance score
        relevance = info["term_matches"] * 10
        if info["source"] == "reddit":
            relevance += min(info["score"] / 100, 10)
        
        # Add exact match bonus
        if any(term in info["doc"].lower() for term in terms):
            relevance += 5
        
        # Add source type bonus
        if info["source"] == "twitter":
            relevance += 2
        elif info["source"] == "details":
            relevance += 3
        
        info["boolean_score"] = round(relevance, 2)
        results.append(info)
    
    # Sort by relevance score
    results.sort(key=lambda x: x["boolean_score"], reverse=True)
    return results[:top_k]

# ===== Hybrid Search =====
def combine_search_results(semantic_results, boolean_results, semantic_weight=WIKI_SEMANTIC_WEIGHT, boolean_weight=BOOLEAN_WEIGHT):
    """Combine semantic and boolean search results with weighting"""
    combined_results = {}
    
    # Process semantic results
    for result in semantic_results:
        streamer = result["name"]
        if streamer not in combined_results:
            combined_results[streamer] = {
                "name": streamer,
                "documents": [],
                "max_score": 0
            }
        # Normalize and weight the semantic score
        weighted_score = result["semantic_score"] * semantic_weight
        result["final_score"] = round(weighted_score, 2)
        combined_results[streamer]["documents"].append(result)
        combined_results[streamer]["max_score"] = max(combined_results[streamer]["max_score"], weighted_score)
    
    # Process boolean results
    for result in boolean_results:
        streamer = result["name"]
        if streamer not in combined_results:
            combined_results[streamer] = {
                "name": streamer,
                "documents": [],
                "max_score": 0
            }
        # Normalize and weight the boolean score
        weighted_score = result["boolean_score"] * boolean_weight
        result["final_score"] = round(weighted_score, 2)
        combined_results[streamer]["documents"].append(result)
        combined_results[streamer]["max_score"] = max(combined_results[streamer]["max_score"], weighted_score)
    
    # Sort streamers by max score
    sorted_results = sorted(combined_results.values(), key=lambda x: x["max_score"], reverse=True)
    
    # Sort documents within each streamer by score
    for streamer_data in sorted_results:
        streamer_data["documents"].sort(key=lambda x: x["final_score"], reverse=True)
        # Limit to top 5 documents per streamer
        streamer_data["documents"] = streamer_data["documents"][:5]
    
    return sorted_results

# ===== Helper Functions =====
def get_twitch_info(streamer_name):
    """Get Twitch page info for a streamer if available."""
    variants = [
        streamer_name,
        streamer_name.upper(),
        streamer_name.lower(),
        streamer_name.title(),
        streamer_name.replace(" ", "")
    ]
    for name_variant in variants:
        if name_variant in streamer_csv_data:
            data = streamer_csv_data[name_variant]
            if "Twitch URL" in data and data["Twitch URL"].strip():
                return data
            else:
                default_url = f"https://www.twitch.tv/{streamer_name}"
                data["url"] = default_url
                return data
    logger.info(f"No Twitch data found for streamer: {streamer_name}")
    return None

def get_streamer_image_path(streamer_name):
    """Get the image path for a streamer if available."""
    image_paths = [
        f"images/streamer_images/{streamer_name.upper()}.jpg",
        f"images/streamer_images/{streamer_name}.jpg",
        f"images/streamer_images/{streamer_name.lower()}.jpg",
        f"images/streamer_images/{streamer_name.replace(' ', '')}.jpg"
    ]
    return image_paths[0]

def get_csv_streamer_info(streamer_name):
    """Look up extra CSV info for the streamer from streamer_details.csv."""
    name_upper = streamer_name.upper().strip()
    return streamer_csv_data.get(name_upper, None)

# ===== Initialize Search System =====
logger.info("Initializing search system...")

# Initialize Word2Vec search for Wikipedia
wiki_search = LightWord2VecSearch(model_name=MODEL_NAME, vector_size=VECTOR_SIZE)

# Check if model needs to be rebuilt based on configuration
if not wiki_search.rebuild_model_if_needed():
    logger.warning("Failed to load or rebuild Word2Vec model. Using fallback only...")

# Load the word2vec model for queries (needed even if vectors are pre-computed)
wiki_search.load_pretrained_model()

# Initialize boolean search index
boolean_index = None
try:
    if os.path.exists(BOOLEAN_INDEX_PATH):
        logger.info("Loading boolean index...")
        with open(BOOLEAN_INDEX_PATH, "rb") as f:
            boolean_index = pickle.load(f)
    else:
        logger.info("Building boolean index...")
        boolean_index = create_boolean_index()
        os.makedirs(os.path.dirname(BOOLEAN_INDEX_PATH), exist_ok=True)
        with open(BOOLEAN_INDEX_PATH, "wb") as f:
            pickle.dump(boolean_index, f)
except Exception as e:
    logger.error(f"Error with boolean index: {e}")
    boolean_index = create_boolean_index()

# ===== Initialize Flask App =====
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("base.html", title="Streamer Search")

@app.route("/search")
def search_streamer():
    query = request.args.get("name", "")
    if not query:
        return jsonify([])
    
    logger.info(f"Search query: {query}")
    start_time = time.time()
    
    try:
        # Perform Word2Vec search on Wikipedia data
        semantic_results = wiki_search.query(query)
        logger.info(f"Semantic search: {len(semantic_results)} results in {time.time() - start_time:.4f} seconds")
        
        # Perform boolean search on other data
        boolean_results = boolean_search(query, boolean_index)
        logger.info(f"Boolean search: {len(boolean_results)} results in {time.time() - start_time:.4f} seconds")
        
        # Combine results
        combined_results = combine_search_results(semantic_results, boolean_results)
        logger.info(f"Combined search: {len(combined_results)} streamers found")
        
        # Format final results
        final_results = []
        for streamer_data in combined_results[:10]:  # Limit to top 10 streamers
            streamer_name = streamer_data["name"]
            final_results.append({
                "name": streamer_name,
                "documents": streamer_data["documents"],
                "twitch_info": get_twitch_info(streamer_name),
                "image_path": get_streamer_image_path(streamer_name),
                "csv_data": get_csv_streamer_info(streamer_name)
            })
        
        logger.info(f"Search completed in {time.time() - start_time:.4f} seconds")
        return jsonify(final_results)
    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)
        return jsonify([])  # Return empty results rather than error

# Additional endpoint for Wikipedia-only search
@app.route("/wiki-search")
def wiki_search_endpoint():
    query = request.args.get("name", "")
    if not query:
        return jsonify([])
    
    logger.info(f"Wiki search query: {query}")
    start_time = time.time()
    
    try:
        # Perform Word2Vec search on Wikipedia data
        semantic_results = wiki_search.query(query)
        logger.info(f"Wiki search: {len(semantic_results)} results in {time.time() - start_time:.4f} seconds")
        # Group by streamer
        streamer_results = {}
        for result in semantic_results:
            streamer = result["name"]
            if streamer not in streamer_results:
                streamer_results[streamer] = {
                    "name": streamer,
                    "documents": [],
                    "max_score": 0
                }
            result["final_score"] = result["semantic_score"]  # Use semantic score as final score
            streamer_results[streamer]["documents"].append(result)
            streamer_results[streamer]["max_score"] = max(
                streamer_results[streamer]["max_score"], 
                result["semantic_score"]
            )
        
        # Sort streamers by max score
        sorted_results = sorted(streamer_results.values(), key=lambda x: x["max_score"], reverse=True)
        
        # Format final results
        final_results = []
        for streamer_data in sorted_results[:10]:  # Limit to top 10 streamers
            streamer_name = streamer_data["name"]
            final_results.append({
                "name": streamer_name,
                "documents": streamer_data["documents"][:5],  # Limit to top 5 documents per streamer
                "twitch_info": get_twitch_info(streamer_name),
                "image_path": get_streamer_image_path(streamer_name),
                "csv_data": get_csv_streamer_info(streamer_name)
            })
        
        logger.info(f"Wiki search completed in {time.time() - start_time:.4f} seconds")
        return jsonify(final_results)
    except Exception as e:
        logger.error(f"Error during wiki search: {e}", exc_info=True)
        return jsonify([])  # Return empty results rather than error

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)