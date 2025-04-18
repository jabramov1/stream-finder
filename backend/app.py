import json
import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import normalize
import time

# Set ROOT_PATH for linking files
os.environ["ROOT_PATH"] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script (backend folder)
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define models directory
models_dir = os.path.join(current_directory, "models")

# Specify the path to the JSON file (init.json) in the backend folder
json_path = os.path.join(current_directory, "init.json")

# Load the JSON data with UTF-8 encoding (still needed for streamer info)
with open(json_path, "r", encoding="utf-8") as file:
    combined_data = json.load(file)

# Extract the individual datasets (needed for document details)
reddit_data = combined_data["reddit"]
twitter_data = combined_data["twitter"]
wiki_data = combined_data["wiki"]
details_data = combined_data["details"]

# Load CSV data about streamers for additional details
csv_path = os.path.join(current_directory, "streamer_details.csv")
streamer_csv = pd.read_csv(csv_path).fillna("")  # Safely fill NaNs with empty strings

# Convert CSV rows into a dict keyed by uppercase Name
streamer_csv_data = {}
for _, row in streamer_csv.iterrows():
    name_upper = str(row["Name"]).upper().strip()
    streamer_csv_data[name_upper] = dict(row)


class OptimizedTFIDFSVDSearch:
    """Optimized version of TFIDFSVDSearch that loads pre-computed models"""
    
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.vectorizer = None
        self.u = None
        self.s = None
        self.vt = None
        self.docs_compressed = None
        self.doc_lookup = {}
        self.index_to_word = {}
        self.word_to_index = {}
        self.dimension_labels = []
        
    def load_model(self):
        """Load all model components from disk"""
        print("Loading pre-computed model components...")
        start_time = time.time()
        
        # Load the vectorizer
        vectorizer_path = os.path.join(self.models_dir, "vectorizer.pkl")
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
            self.word_to_index = self.vectorizer.vocabulary_
        
        # Load SVD components
        self.u = np.load(os.path.join(self.models_dir, "u_matrix.npy"))
        self.s = np.load(os.path.join(self.models_dir, "s_values.npy"))
        self.vt = np.load(os.path.join(self.models_dir, "vt_matrix.npy"))
        
        # Load normalized document vectors
        self.docs_compressed = np.load(os.path.join(self.models_dir, "docs_compressed.npy"))
        
        # Load document lookup mappings
        with open(os.path.join(self.models_dir, "doc_lookup.pkl"), "rb") as f:
            self.doc_lookup = pickle.load(f)
        
        # Load word mappings
        with open(os.path.join(self.models_dir, "index_to_word.pkl"), "rb") as f:
            self.index_to_word = pickle.load(f)
        
        # Load dimension labels
        with open(os.path.join(self.models_dir, "dimension_labels.pkl"), "rb") as f:
            self.dimension_labels = pickle.load(f)
        
        print(f"Model loading completed in {time.time() - start_time:.2f} seconds")
        return self
    
    def query(self, query_text, top_k=10):
        """Transform a query and find the most similar documents - optimized version"""
        start_time = time.time()
        
        # Transform query to TF-IDF space
        query_tfidf = self.vectorizer.transform([query_text])
        
        # Project query to concept space
        query_vec = query_tfidf @ self.vt.T
        
        # Scale query vector by singular values to weight important dimensions more
        weighted_query_vec = query_vec @ np.diag(self.s)
        
        # Normalize for cosine similarity
        query_vec_norm = normalize(weighted_query_vec)
        
        # Compute cosine similarity with all documents - this is a single matrix operation
        # Shape of docs_compressed: [n_docs, n_components]
        # Shape of query_vec_norm.T: [n_components, 1]
        # Result shape: [n_docs, 1]
        similarities = self.docs_compressed @ query_vec_norm.T
        
        # Get top-k most similar document indices (fastest part)
        top_indices = np.argsort(-similarities.flatten())[:top_k]
        
        print(f"Found top {top_k} matches in {time.time() - start_time:.4f} seconds")
        
        # Format results
        results = []
        for doc_idx in top_indices:
            source, streamer, idx, data = self.doc_lookup[doc_idx]
            similarity_score = float(similarities[doc_idx, 0])
            
            # Find top contributing dimensions for this document
            doc_factors = self.u[doc_idx]
            query_factors = query_vec_norm[0]
            
            # Calculate contribution of each dimension to similarity score
            dimension_contributions = doc_factors * query_factors
            top_dim_indices = np.argsort(-dimension_contributions)[:3]  # Top 3 dimensions
            
            top_dimensions = [
                {
                    "index": int(dim_idx),
                    "label": self.dimension_labels[dim_idx],
                    "contribution": float(dimension_contributions[dim_idx])
                }
                for dim_idx in top_dim_indices
            ]
            
            # Create document text representation
            if source == "reddit":
                text = data["Title"]
                score = data["Score"]
                reddit_id = data["ID"]
                result = {
                    "source": source,
                    "name": streamer,
                    "doc": text[:150] + "..." if len(text) > 150 else text,
                    "sim_score": round(similarity_score * 100, 2),
                    "reddit_score": score,
                    "id": reddit_id,
                    "top_dimensions": top_dimensions
                }
            elif source == "twitter":
                text = data
                result = {
                    "source": source,
                    "name": streamer,
                    "doc": text[:150] + "..." if len(text) > 150 else text,
                    "sim_score": round(similarity_score * 100, 2),
                    "top_dimensions": top_dimensions
                }
            elif source == "wiki":
                text = data["wikipedia_summary"] if isinstance(data, dict) else str(data)
                result = {
                    "source": source,
                    "name": streamer,
                    "doc": text[:150] + "..." if len(text) > 150 else text,
                    "sim_score": round(similarity_score * 100, 2),
                    "top_dimensions": top_dimensions
                }
            elif source == "details":
                text = data.get("Description", "")
                result = {
                    "source": source,
                    "name": streamer,
                    "doc": text[:150] + "..." if len(text) > 150 else text,
                    "sim_score": round(similarity_score * 100, 2),
                    "top_dimensions": top_dimensions
                }
            
            results.append(result)
            
        print(f"Total search time: {time.time() - start_time:.4f} seconds")
        return results
    
    def analyze_svd_components(self, n_terms=10):
        """Analyze the top terms in each SVD dimension"""
        results = []
        for i in range(len(self.dimension_labels)):
            dimension = self.vt[i, :]
            top_indices = np.argsort(-dimension)[:n_terms]
            top_terms = [self.index_to_word[idx] for idx in top_indices]
            results.append((i, top_terms, self.dimension_labels[i]))
        return results
        
    def plot_singular_values(self):
        """Return the singular values for plotting"""
        return self.s


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
    print(f"No Twitch data found for streamer: {streamer_name}")
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

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Check if pre-computed models exist
if os.path.exists(models_dir) and os.path.isfile(os.path.join(models_dir, "vectorizer.pkl")):
    print("Found pre-computed models. Loading optimized search engine...")
    search_engine = OptimizedTFIDFSVDSearch(models_dir)
    search_engine.load_model()
else:
    print("Pre-computed models not found. Please run preprocess_data.py first.")
    print("Falling back to in-memory computation (slower startup)...")
    from preprocess_data import TFIDFSVDSearch
    search_engine = TFIDFSVDSearch(n_components=30)
    search_engine.preprocess_documents(reddit_data, twitter_data, wiki_data, details_data)
    search_engine.fit()

@app.route("/")
def home():
    return render_template("base.html", title="Streamer Search")

@app.route("/search")
def search_streamer():
    query = request.args.get("name", "")
    if not query:
        return jsonify([])
    
    # Use the SVD-powered search
    results = search_engine.query(query, top_k=50)  # Get top 50 results
    
    # Group results by streamer
    streamer_results = {}
    for result in results:
        streamer = result["name"]
        if streamer not in streamer_results:
            streamer_results[streamer] = {
                "documents": [],
                "twitch_info": get_twitch_info(streamer),
                "total_score": 0
            }
        streamer_results[streamer]["documents"].append(result)
        streamer_results[streamer]["total_score"] += result["sim_score"]
    
    final_results = []
    for streamer, data in streamer_results.items():
        csv_info = get_csv_streamer_info(streamer)
        final_results.append({
            "name": streamer,
            "documents": data["documents"][:5],  # Limit to top 5 documents per streamer
            "twitch_info": data["twitch_info"],
            "image_path": get_streamer_image_path(streamer),
            "csv_data": csv_info
        })
    
    # Sort by total similarity score
    final_results.sort(
        key=lambda x: sum([doc["sim_score"] for doc in x["documents"]]) if x["documents"] else 0, 
        reverse=True
    )
    
    # Return the top results (limited to improve performance)
    return jsonify(final_results[:10])

# Additional endpoint for SVD analysis
@app.route("/analyze_svd")
def analyze_svd():
    components = search_engine.analyze_svd_components(n_terms=15)
    singular_values = search_engine.plot_singular_values().tolist()
    return jsonify({
        "components": components,
        "singular_values": singular_values
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)