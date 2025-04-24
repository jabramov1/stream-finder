import os, json, pickle, sys, numpy as np, re
from sentence_transformers import SentenceTransformer
import faiss, torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from collections import defaultdict
import gc
from tqdm import tqdm


print("👋 app.py launched", flush=True)
app_dir = os.path.abspath(os.path.dirname(__file__))
print("App directory:", app_dir, flush=True)

# Filter out static folder when listing contents
contents = [item for item in os.listdir(app_dir) if item != "static"]
print("Contents:", contents, flush=True)

# Skip the static folder when walking through directories
for root, _, files in os.walk(app_dir):
    if "static" not in root:
        for f in files:
            print(os.path.join(root, f), flush=True)

print("sys.path =", sys.path, flush=True)
print("ENV PATH =", os.environ.get("PATH", ""), flush=True)



# CONFIG
MODEL_NAME             = "intfloat/e5-base-v2"
EMBED_DIM, TOP_K       = 768, 25
META_PATH              = "models/metadata.pkl"
BOOLEAN_INDEX_PATH     = "models/boolean_index.pkl"
EMBED_PATH             = "models/embeddings.npy"
SEMANTIC_WEIGHT        = 0.5
BOOLEAN_WEIGHT         = 0.5
TWITCH_USERNAME_REGEX = r'^[a-z0-9_]{4,25}'

STOP_WORDS = {"the", "and", "a", "of", "to", "in", "is", "you", "that", "it", "was", "for", "on", "streamer"}

# LOAD DATA
BACK = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(BACK, "init.json"), "r", encoding="utf-8") as f:
        init = json.load(f)
        reddit_data, twitter_data = init.get("reddit", {}), init.get("twitter", {})
        wiki_data, details_data = init.get("wiki", {}), init.get("details", {})
except Exception as e:
    print(f"Error loading init.json: {e}. Exiting."); sys.exit(1)

# Load CSV data if available
streamer_csv_data = {}
print(f"Looking for embeddings at: {os.path.join(BACK, EMBED_PATH)}")
print(f"Looking for metadata at: {os.path.join(BACK, META_PATH)}")
print(f"Directory exists: {os.path.exists(BACK)}")
if os.path.exists(CSV_PATH := os.path.join(BACK, "streamer_details.csv")):
    try:
        streamer_csv = pd.read_csv(CSV_PATH).fillna("")
        streamer_csv_data = {str(r["Name"]).upper().strip(): dict(r) for _, r in streamer_csv.iterrows()}
    except: pass

# HELPER FUNCTIONS
def is_valid_username(name):
    return bool(re.match(TWITCH_USERNAME_REGEX, name.lower().replace(' ', '')))

# At the top of your file, define:
SOURCE_WEIGHT = {
    "wiki":    2.0,   # double-weight wiki
    "details": 1.5,   # 1.5× for your “details” section
    "reddit":  1.0,   # default
    "twitter": 1.0,   # default
}

def embed_streamer(full_text: str,
                   chunk_size: int = 300,
                   overlap:    int = 75,
                   batch_size: int = 16) -> np.ndarray:
    """
    1) Slide a chunk_size-word window with `overlap` over full_text
    2) SBERT-encode & normalize each window
    3) Weight each window by (SOURCE_WEIGHT × window_length)
    4) Softmax-pool → one EMBED_DIM vector
    """
    words = full_text.split()
    if not words:
        return np.zeros(EMBED_DIM, dtype=np.float32)

    step = max(1, chunk_size - overlap)
    windows, raw_w = [], []

    # build windows & raw weights
    for i in range(0, len(words), step):
        w = " ".join(words[i : i + chunk_size])
        if not w:
            continue
        windows.append(w)

        # detect source label in your text if you prefix with e.g. "wiki:" or "reddit:"
        src = w.split(":", 1)[0].lower()
        bias = SOURCE_WEIGHT.get(src, 1.0)
        raw_w.append(bias * len(w.split()))

    # encode + normalize
    embs = model.encode(
        windows,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True
    )  # shape (n_windows, EMBED_DIM)

    # softmax over raw_w → attention weights
    r    = np.array(raw_w, dtype=np.float32)
    exps = np.exp(r - r.max())
    attn = exps / exps.sum()

    # weighted mean + final normalize
    vec = attn @ embs            # (EMBED_DIM,)
    vec /= np.linalg.norm(vec) + 1e-8
    return vec



def split_text(text, min_len=100, max_len=600):
    chunks, current = [], ""
    for sent in re.split(r'(?<=[.!?])\s+', text):
        if len(current) + len(sent) <= max_len:
            current += " " + sent if current else sent
        else:
            if current and len(current) >= min_len: chunks.append(current)
            current = sent
    if current and len(current) >= min_len: chunks.append(current)
    return chunks

def make_snippet(text, terms, max_len=300):
    sentences = re.split(r'(?<=[.!?])\s+', text.replace("\n", " "))
    for sent in sentences:
        for t in terms:
            if t.lower() in sent.lower():
                sent = sent.strip()
                return sent if len(sent) <= max_len else sent[:max_len].rstrip() + "…"
    return text[:max_len].rstrip() + ("…" if len(text) > max_len else "")

def get_twitch_info(name):
    sun = name.upper().strip()
    if sun in streamer_csv_data:
        data = streamer_csv_data[sun].copy()
        if not data.get("url") and (tu := data.get("Twitch URL", "")):
            data["url"] = tu.strip()
        else: data["url"] = f"https://www.twitch.tv/{name.replace(' ', '').lower()}"
        if "Name" not in data: data["Name"] = name
        return data
    return {"url": f"https://www.twitch.tv/{name.replace(' ', '').lower()}", "Name": name}

def get_image_path(name):
    for v in [name.upper(), name, name.lower(), name.replace(" ", ""), name.replace(" ", "_")]:
        for e in [".jpg", ".png", ".jpeg", ".webp"]:
            if os.path.exists(path := os.path.join(BACK, "static/images/streamer_images", f"{v}{e}")):
                return f"images/streamer_images/{v}{e}"
    return "images/streamer_images/default.png"

# BOOLEAN SEARCH
def create_boolean_index():
    index = defaultdict(list)
    # Reddit
    for streamer, posts in reddit_data.items():
        if not is_valid_username(streamer) or not isinstance(posts, list): continue
        for i, post in enumerate(posts):
            if not isinstance(post, dict): continue
            for w in [w for w in re.findall(r"\w+", post.get("Title", "").lower()) if w not in STOP_WORDS]:
                index[w].append(("reddit", streamer, i))
    
    # Twitter
    for streamer, tweets in twitter_data.items():
        if not is_valid_username(streamer) or not isinstance(tweets, list): continue
        for i, tweet in enumerate(tweets):
            for w in [w for w in re.findall(r"\w+", str(tweet).lower()) if w not in STOP_WORDS]:
                index[w].append(("twitter", streamer, i))
    
    # Wiki
    if isinstance(wiki_data, dict):
        for streamer, entry in wiki_data.items():
            if not is_valid_username(streamer) or not isinstance(entry, dict): continue
            for i, chunk in enumerate(split_text(entry.get("content", ""))):
                for w in [w for w in re.findall(r"\w+", chunk.lower()) if w not in STOP_WORDS]:
                    index[w].append(("wiki", streamer, i))
    
    # Details
    for streamer, details in details_data.items():
        if not is_valid_username(streamer) or not isinstance(details, dict): continue
        for w in [w for w in re.findall(r"\w+", str(details.get("Description", "")).lower()) if w not in STOP_WORDS]:
            index[w].append(("details", streamer, 0))
    
    return index

def boolean_search(query, index):
    terms = [t for t in re.findall(r"\w+", query.lower()) if t not in STOP_WORDS]
    if not terms: return []
    
    matches, info = defaultdict(int), {}
    for term in terms:
        if term not in index: continue
        for source, streamer, idx in index[term]:
            doc_id = f"{source}:{streamer}:{idx}"
            matches[doc_id] += 1
            
            if doc_id in info: continue
            
            # Extract document text based on source
            text, score = "", 1
            try:
                if source == "reddit" and streamer in reddit_data and isinstance(reddit_data[streamer], list):
                    if idx < len(reddit_data[streamer]) and isinstance(post := reddit_data[streamer][idx], dict):
                        text, score = post.get("Title", ""), post.get("Score", 1)
                        info[doc_id] = {"source": source, "streamer": streamer, "data": post, 
                                       "text": text, "score": score, "idx": idx, "term_matches": 0}
                
                elif source == "twitter" and streamer in twitter_data and isinstance(twitter_data[streamer], list):
                    if idx < len(twitter_data[streamer]):
                        text = str(twitter_data[streamer][idx])
                        info[doc_id] = {"source": source, "streamer": streamer, "data": twitter_data[streamer][idx],
                                       "text": text, "score": 1, "idx": idx, "term_matches": 0}
                
                elif source == "wiki" and streamer in wiki_data and isinstance(wiki_data[streamer], dict):
                    chunks = split_text(wiki_data[streamer].get("content", ""))
                    if idx < len(chunks):
                        text = chunks[idx]
                        info[doc_id] = {"source": source, "streamer": streamer, "data": wiki_data[streamer],
                                       "text": text, "score": 2, "idx": idx, "term_matches": 0}
                
                elif source == "details" and streamer in details_data and isinstance(details_data[streamer], dict):
                    text = str(details_data[streamer].get("Description", ""))
                    info[doc_id] = {"source": source, "streamer": streamer, "data": details_data[streamer],
                                   "text": text, "score": 3, "idx": idx, "term_matches": 0}
            except: continue
    
    for doc_id, count in matches.items():
        if doc_id in info: info[doc_id]["term_matches"] = count
    
    return list(info.values())

def score_boolean_results(results, query):
    terms = re.findall(r"\w+", query.lower())
    scored = []
    
    for doc in results:
        if not isinstance(doc, dict): continue
        
        text = doc.get("text", "")
        score = doc.get("term_matches", 0) * 15
        score += sum(text.lower().count(term) * 5 for term in terms)
        
        ss = doc.get("score", 1)
        if doc["source"] == "reddit" and isinstance(ss, (int, float)):
            score += min(ss / 500.0, 20) if ss > 0 else 0
        elif doc["source"] == "wiki": score += 15
        elif doc["source"] == "details": score += 10
        
        if query.lower() in text.lower(): score += 50
        
        snippet = make_snippet(text, terms)
        fmt = {
            "source": doc["source"],
            "name": doc["streamer"],
            "snippet": snippet,
            "idx": doc["idx"],
            "doc": snippet,
            "boolean_score": round(score, 2),
            "term_matches": doc.get("term_matches", 0)
        }
        
        if doc["source"] == "reddit" and isinstance(doc.get("data"), dict):
            fmt["reddit_score"] = ss
            fmt["id"] = doc["data"].get("ID", "")
        
        scored.append((fmt, score))
    
    return [d for d, s in sorted(scored, key=lambda x: x[1], reverse=True)]

# SEMANTIC SEARCH
def gather_documents():
    docs = []
    
    # Process data with identity prefixes and chunking
    for streamer, posts in reddit_data.items():
        if not is_valid_username(streamer) or not isinstance(posts, list): continue
        for idx, post in enumerate(posts):
            if not isinstance(post, dict) or "Title" not in post: continue
            docs.append({"text": f"{streamer}: {post['Title']}", "source": "reddit", 
                        "streamer": streamer, "idx": idx, "data": post, "score": post.get("Score", 1)})
    
    for streamer, tweets in twitter_data.items():
        if not is_valid_username(streamer) or not isinstance(tweets, list): continue
        for idx, tweet in enumerate(tweets):
            if not isinstance(tweet, str): continue
            docs.append({"text": f"{streamer}: {tweet}", "source": "twitter", 
                        "streamer": streamer, "idx": idx, "data": tweet, "score": 1})
    
    for streamer, entry in wiki_data.items():
        if not is_valid_username(streamer) or not isinstance(entry, dict): continue
        for idx, chunk in enumerate(split_text(entry.get("content", ""))):
            docs.append({"text": f"{streamer}: {chunk}", "source": "wiki", 
                        "streamer": streamer, "idx": idx, "data": entry, "score": 2})
    
    for streamer, details in details_data.items():
        if not is_valid_username(streamer) or not isinstance(details, dict): continue
        if desc := str(details.get("Description", "")):
            docs.append({"text": f"{streamer}: {desc}", "source": "details", 
                        "streamer": streamer, "idx": 0, "data": details, "score": 3})
    
    return docs

def score_semantic_results(results, query):
    terms = set(re.findall(r"\w+", query.lower()))
    scored = []
    
    for doc in results:
        if not isinstance(doc, dict): continue
        
        score = doc.get("sim_score", 0.0)
        text = doc.get("text", "").lower()
        
        # Boost scores based on source
        if doc["source"] == "wiki": score += 15
        elif doc["source"] == "details": score += 10
        elif doc["source"] == "reddit" and isinstance(ss := doc.get("score", 1), (int, float)):
            score += min(ss / 500.0, 20) if ss > 0 else 0
        
        # Term frequency bonus
        score += sum(text.count(term) * 2 for term in terms if term)
        
        # Exact phrase bonus
        if query and query.lower() in text: score += 30
        
        fmt = {
            "source": doc.get("source", "unknown"),
            "name": doc.get("streamer", "unknown"),
            "doc": text[:150] + ("..." if len(text) > 150 else ""),
            "semantic_score": round(score, 2),
            "sim_score": doc.get("sim_score", 0.0),
            "idx": doc.get("idx", 0)
        }
        
        if doc["source"] == "reddit" and isinstance(doc.get("data"), dict):
            fmt["reddit_score"] = doc.get("score", 1)
            fmt["id"] = doc["data"].get("ID", "")
        
        scored.append((fmt, score))
    
    return [d for d, s in sorted(scored, key=lambda x: x[1], reverse=True)]

# HYBRID SEARCH
def combine_results(bool_res, sem_res, threshold=5.0):
    combined = {}
    
    # Helper to create doc key
    def get_key(doc):
        if not isinstance(doc, dict): return None
        source = doc.get('source', 'unk')
        streamer = doc.get('name', 'unk')
        idx = doc.get('idx', -1)
        doc_id = doc.get('id', None)
        return (f"{source}:{streamer}:{doc_id}" if source == 'reddit' and doc_id else
                f"{source}:{streamer}:{idx}" if idx != -1 else
                f"{source}:{streamer}:{doc.get('doc', '')[:20]}")
    
    # Process boolean results
    for doc in bool_res:
        if key := get_key(doc):
            combined[key] = {"doc_info": doc.copy(), "boolean_score": doc.get("boolean_score", 0.0),
                           "semantic_score": 0.0, "sim_score": 0.0}
    
    # Process semantic results
    for doc in sem_res:
        if key := get_key(doc):
            if key not in combined:
                combined[key] = {"doc_info": doc.copy(), "boolean_score": 0.0,
                               "semantic_score": doc.get("semantic_score", 0.0),
                               "sim_score": doc.get("sim_score", 0.0)}
            else:
                combined[key]["semantic_score"] = max(combined[key]["semantic_score"], 
                                                    doc.get("semantic_score", 0.0))
                combined[key]["sim_score"] = max(combined[key]["sim_score"], doc.get("sim_score", 0.0))
                combined[key]["doc_info"] = doc.copy()  # Use semantic version for better info
    
    # Calculate final scores
    scored_docs = []
    for data in combined.values():
        b_score, s_score = data["boolean_score"], data["semantic_score"]
        
        # Calculate final score based on presence and thresholds
        if b_score > 0 and s_score > 0:
            # Both scores present - use weighted sum
            final_score = (b_score * BOOLEAN_WEIGHT) + (s_score * SEMANTIC_WEIGHT)
        elif b_score > threshold:
            # Only boolean above threshold
            final_score = b_score
        elif s_score > threshold:
            # Only semantic above threshold
            final_score = s_score
        else:
            # Neither meets criteria
            continue
        
        # Add final score to doc and collect
        doc = data["doc_info"]
        doc["final_score"] = round(final_score, 2)
        doc["sim_score"] = round(data.get("sim_score", 0.0), 2)
        scored_docs.append(doc)
    
    # Group by streamer
    streamer_results = defaultdict(lambda: {"name": "", "documents": [], "max_final_score": 0.0})
    for doc in scored_docs:
        name = doc.get("name", "unknown")
        if name != "unknown":
            streamer_results[name]["name"] = name
            streamer_results[name]["documents"].append(doc)
            streamer_results[name]["max_final_score"] = max(
                streamer_results[name]["max_final_score"],
                doc.get("final_score", 0.0)
            )
    
    # Sort streamers by max score
    results = list(streamer_results.values())
    results.sort(key=lambda x: x["max_final_score"], reverse=True)
    
    # Sort documents within each streamer
    for sd in results:
        sd["documents"].sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    
    return results

# INITIALIZE SEARCH SYSTEM
print("Initializing search system...")

# Load or build boolean index
try:
    if os.path.exists(BOOLEAN_INDEX_PATH):
        with open(BOOLEAN_INDEX_PATH, "rb") as f:
            boolean_index = pickle.load(f)
    else:
        print("Building boolean index...")
        boolean_index = create_boolean_index()
        with open(BOOLEAN_INDEX_PATH, "wb") as f:
            pickle.dump(boolean_index, f)
except Exception as e:
    print(f"Error with boolean index: {e}")
    boolean_index = create_boolean_index()

# Initialize model
device = "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)

# Load or build embeddings
# Load or build embeddings
DOCS, EMBEDDINGS = [], None
if os.path.exists(EMBED_PATH) or os.path.exists(META_PATH):
    try:
        print(f"Attempting to load embeddings from {EMBED_PATH}")
        EMBEDDINGS = np.load(EMBED_PATH)
        print(f"Successfully loaded embeddings with shape: {EMBEDDINGS.shape}")
        
        print(f"Attempting to load metadata from {META_PATH}")
        with open(META_PATH, "rb") as f:
            DOCS = pickle.load(f)
        print(f"Successfully loaded metadata with {len(DOCS)} documents")
        
        if EMBEDDINGS.shape[1] != EMBED_DIM or len(DOCS) != EMBEDDINGS.shape[0]:
            raise ValueError(f"Dimension mismatch: EMBEDDINGS shape: {EMBEDDINGS.shape}, DOCS length: {len(DOCS)}, Expected EMBED_DIM: {EMBED_DIM}")
    except Exception as e:
        print(f"ERROR LOADING FILES: {e}")
        EMBEDDINGS, DOCS = None, []

# Build embeddings if needed
if EMBEDDINGS is None:
    print("Building new embeddings...")
    # 1) gather all raw texts per streamer
    per_stream = defaultdict(list)
    # collect from every source
    for d in gather_documents():
        per_stream[d["streamer"]].append(d["text"])

    DOCS, vectors = [], []
    total_streamers = len(per_stream)
    
    # Add progress bar
    with tqdm(total=total_streamers, desc="Embedding streamers") as pbar:
        for streamer, texts in per_stream.items():
            full_text = " ".join(texts)
            vec = embed_streamer(full_text, chunk_size=200)
            DOCS.append({"streamer": streamer})
            vectors.append(vec)
            pbar.update(1)
            
    # 2) stack into (N,768) array
    EMBEDDINGS = np.vstack(vectors).astype(np.float32)
    print(f"\nBuilt {len(DOCS)} streamer-level embeddings via chunk+pool")

    # 3) save
    np.save(EMBED_PATH, EMBEDDINGS)
    with open(META_PATH, "wb") as f:
        pickle.dump(DOCS, f)

    print(f"Saved embeddings to {EMBED_PATH} and metadata to {META_PATH}")

# Initialize FAISS index with IP for cosine similarity
try:
    EMBEDDINGS = np.ascontiguousarray(EMBEDDINGS) if not EMBEDDINGS.flags['C_CONTIGUOUS'] else EMBEDDINGS
    index = faiss.IndexFlatIP(EMBED_DIM)  # Using IP for normalized embeddings = cosine similarity
    index.add(EMBEDDINGS)
    del EMBEDDINGS; gc.collect()  # Release memory
except Exception as e:
    print(f"Error initializing FAISS: {e}"); sys.exit(1)

# FLASK SERVER
app = Flask(__name__, static_folder='static')
CORS(app)

@app.route("/")
def home():
    return render_template("base.html", title="Streamer Search")

@app.route("/search")
def search_streamer():
    query = request.args.get("name", "").strip()
    if not query: return jsonify([])
    
    # Boolean search
    bool_res = score_boolean_results(boolean_search(query, boolean_index), query)
    
    # Semantic search
    sem_raw = []
    try:
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        # Get semantically similar documents (higher score = better match with IndexFlatIP)
        scores, ids = index.search(q_emb, TOP_K)
        
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1 or idx >= len(DOCS): continue
            
            meta = DOCS[idx]
            if not isinstance(meta, dict): continue
            
            # Normalize cosine similarity to 0-100 range
            sim_score = max(0.0, min(1.0, score)) * 100
            
            sem_raw.append({
                "source": meta.get("source", "unknown"),
                "streamer": meta.get("streamer", "unknown"),
                "text": meta.get("text", ""),
                "idx": meta.get("idx", idx),
                "score": meta.get("score", 1),
                "data": meta.get("data", {}),
                "sim_score": round(sim_score, 2)
            })
    except Exception as e:
        print(f"Error in semantic search: {e}")
    
    sem_res = score_semantic_results(sem_raw, query)
    
    # Combine results
    comb_res = combine_results(bool_res, sem_res)
    
    # Format final output
    final_res = []
    for sd in comb_res[:10]:  # Limit to top 10 streamers
        name = sd.get("name", "Unknown")
        docs = sd.get("documents", [])[:4]  # Limit to top 4 docs per streamer
        
        final_res.append({
            "name": name,
            "documents": docs,
            "twitch_info": get_twitch_info(name),
            "image_path": get_image_path(name),
            "csv_data": streamer_csv_data.get(name.upper().strip(), {}),
            "max_combined_score": sd.get("max_final_score", 0.0)
        })
    def sanitize(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Unserializable object {type(o)}")

    clean = json.loads(json.dumps(final_res, default=sanitize))
    return jsonify(clean)
    


if __name__ == "__main__":

    app.run(debug=False, host="0.0.0.0", port=5001)