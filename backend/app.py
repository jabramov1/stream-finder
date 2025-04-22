import os
import json
import pickle
import sys
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import faiss
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from collections import defaultdict
import gc
import math # Import math for isnan/isinf checks

# ───────────────────────────────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────────────────────────
MODEL_NAME   = "sentence-transformers/all-MiniLM-L12-v2"
META_PATH    = "metadata.pkl"
BOOLEAN_INDEX_PATH = "boolean_index.pkl"

# --- Float32 Configuration ---
EMBED_PATH_FP32 = "embeddings_fp32.npy" # Preferred path
EXPECTED_EMBED_DTYPE = np.float32

BUILD_INDEX_ON_MISSING = True
TOP_K        = 20 # Adjusted from 25 based on previous code
EMBED_DIM    = 384 # Will be verified
SEMANTIC_WEIGHT = 0.5
BOOLEAN_WEIGHT = 0.5

# ───────────────────────────────────────────────────────────────────────────────
# DATA LOADING (As before)
# ───────────────────────────────────────────────────────────────────────────────
BACK = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(BACK, "init.json"), "r", encoding="utf-8") as f: init = json.load(f)
    reddit_data, twitter_data, wiki_data, details_data = init.get("reddit",{}), init.get("twitter",{}), init.get("wiki",{}), init.get("details",{})
except FileNotFoundError: print(f"Error: init.json not found in {BACK}. Exiting."); sys.exit(1)
except Exception as e: print(f"Error loading init.json: {e}. Exiting."); sys.exit(1)

streamer_csv_data = {}
CSV_PATH = os.path.join(BACK, "streamer_details.csv")
if os.path.exists(CSV_PATH):
    try:
        streamer_csv = pd.read_csv(CSV_PATH).fillna("")
        streamer_csv_data = {str(r["Name"]).upper().strip(): dict(r) for _, r in streamer_csv.iterrows()}
    except Exception as e: print(f"Warning: Could not load/parse {CSV_PATH}: {e}")
else: print(f"Info: {CSV_PATH} not found.")

# ───────────────────────────────────────────────────────────────────────────────
# BOOLEAN SEARCH FUNCTIONS (As before)
# ───────────────────────────────────────────────────────────────────────────────

def make_snippet_sentence(text: str, terms: list[str], max_len: int = 300) -> str:
    """
    Split `text` into sentences, find the first one containing any term,
    and return it (trimmed to max_len if needed, with ellipsis).
    """
    clean = text.replace("\n", " ")
    # split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', clean)
    for sent in sentences:
        for t in terms:
            if t.lower() in sent.lower():
                sent = sent.strip()
                if len(sent) <= max_len:
                    return sent
                # still return the full sentence head-trimmed
                return sent[:max_len].rstrip() + "…"
    # fallback to the first max_len chars
    head = clean[:max_len].rstrip()
    return head + ("…" if len(clean) > max_len else "")


def create_boolean_index():
    # (Function content remains the same as previous minimal version)
    index = defaultdict(list)
    for streamer, data in reddit_data.items(): # Reddit
        if isinstance(data, list): # Added check
            for i, post in enumerate(data):
                if isinstance(post, dict): # Added check
                     words = re.findall(r"\w+", post.get("Title", "").lower())
                     for w in words: index[w].append(("reddit", streamer, i))
    for streamer, tweets in twitter_data.items(): # Twitter
        if isinstance(tweets, list): # Added check
            for i, tweet in enumerate(tweets):
                words = re.findall(r"\w+", str(tweet).lower())
                for w in words: index[w].append(("twitter", streamer, i))
    wiki_idx_counter = 0 # Defined for list case
    if isinstance(wiki_data, dict): # Wiki (dict format)
        for streamer, entry in wiki_data.items():
            if isinstance(entry, dict) and "content" in entry:
                words = re.findall(r"\w+", entry["content"].lower())
                for w in words: index[w].append(("wiki", streamer, 0)) # Use 0 for dict case idx
    elif isinstance(wiki_data, list):  # Wiki (list format)
        for i, entry in enumerate(wiki_data):
            if isinstance(entry, dict) and "content" in entry and "streamer" in entry:
                sn = entry["streamer"]
                words = re.findall(r"\w+", entry["content"].lower())
                for w in words:
                    index[w].append(("wiki", sn, i))


    for streamer, details in details_data.items(): # Details
        if isinstance(details, dict): # Added check
            words = re.findall(r"\w+", str(details.get("Description", "")).lower())
            for w in words: index[w].append(("details", streamer, 0)) # Use 0 for details idx
    return index


def boolean_search(query, index):
    # (Function content remains the same as previous minimal version)
    terms = re.findall(r"\w+", query.strip().lower());
    if not terms: return []
    doc_matches, doc_info = defaultdict(int), {}
    for term in terms:
        if term in index:
            for doc_ref in index[term]:
                source, streamer, idx = doc_ref; doc_id = f"{source}:{streamer}:{idx}"; doc_matches[doc_id] += 1
                if doc_id not in doc_info:
                    doc_entry, text, score = None, "", 1
                    try:
                        # Added more specific checks for data types and existence
                        if source == "reddit" and streamer in reddit_data and isinstance(reddit_data[streamer], list) and idx < len(reddit_data[streamer]):
                             entry = reddit_data[streamer][idx]
                             if isinstance(entry, dict):
                                 doc_entry, text, score = entry, entry.get("Title", ""), entry.get("Score", 1)
                        elif source == "twitter" and streamer in twitter_data and isinstance(twitter_data[streamer], list) and idx < len(twitter_data[streamer]):
                             doc_entry, text = twitter_data[streamer][idx], str(twitter_data[streamer][idx])
                        elif source == "wiki":
                             we = None
                             if isinstance(wiki_data, dict) and streamer in wiki_data and isinstance(wiki_data[streamer], dict):
                                 we = wiki_data[streamer]
                             elif isinstance(wiki_data, list) and 0 <= idx < len(wiki_data) and isinstance(wiki_data[idx], dict) and wiki_data[idx].get("streamer") == streamer:
                                 we = wiki_data[idx]

                             if we and "content" in we:
                                 doc_entry, text, score = we, we["content"], 2
                        elif source == "details" and streamer in details_data and isinstance(details_data[streamer], dict):
                             doc_entry, text, score = details_data[streamer], str(details_data[streamer].get("Description", "")), 3

                        if text: # Ensure text was actually found
                             doc_info[doc_id] = {"source": source, "streamer": streamer, "data": doc_entry, "text": text, "score": score, "idx": idx, "term_matches": 0}
                    except (KeyError, IndexError, TypeError) as e: print(f"Warning: Bool access {doc_id}: {e}"); continue
    for doc_id, count in doc_matches.items():
         if doc_id in doc_info: doc_info[doc_id]["term_matches"] = count
    return list(doc_info.values())

def score_boolean_results(results, query):
    """
    Score your boolean hits and return only a sentence snippet
    around the matched term instead of the full doc.
    """
    qt     = re.findall(r"\w+", query.lower())
    scored = []

    for doc in results:
        if not isinstance(doc, dict):
            continue

        text = doc.get("text", "")  # full article

        # 1) compute your original boolean score
        score = doc.get("term_matches", 0) * 15
        score += sum(text.lower().count(term) * 5 for term in qt)
        ss = doc.get("score", 1)
        if doc["source"] == "reddit" and isinstance(ss, (int, float)):
            score += min(ss / 500.0, 20) if ss > 0 else 0
        elif doc["source"] == "wiki":
            score += 15
        elif doc["source"] == "details":
            score += 10
        if query.lower() in text.lower():
            score += 50

        # 2) extract just the sentence snippet
        snippet = make_snippet_sentence(text, qt)
        fmt = {
            "source":        doc["source"],
            "name":          doc["streamer"],
            "snippet":       snippet,
            "doc":           snippet,             # ← add this line
            "boolean_score": round(score, 2),
            "term_matches":  doc.get("term_matches", 0)
        }

        if doc["source"] == "reddit" and isinstance(doc.get("data"), dict):
            fmt["reddit_score"] = ss
            fmt["id"]           = doc["data"].get("ID", "")

        scored.append((fmt, score))

    # sort and return only the dicts
    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored]

# ───────────────────────────────────────────────────────────────────────────────
# SEMANTIC SEARCH FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────
def gather_documents():
    # (Function content remains the same as previous minimal version)
    docs = []
    for streamer, posts in reddit_data.items(): # Reddit
        if isinstance(posts, list): # Added check
            for idx, post in enumerate(posts):
                 if isinstance(post, dict) and "Title" in post: docs.append({"text": post["Title"], "source": "reddit", "streamer": streamer, "idx": idx, "data": post, "score": post.get("Score", 1)})
    for streamer, tweets in twitter_data.items(): # Twitter
        if isinstance(tweets, list): # Added check
            for idx, tweet in enumerate(tweets):
                 if isinstance(tweet, str): docs.append({"text": tweet, "source": "twitter", "streamer": streamer, "idx": idx, "data": tweet, "score": 1})
    wiki_idx_counter = 0
    if isinstance(wiki_data, dict): # Wiki (dict)
        for streamer, entry in wiki_data.items():
            if isinstance(entry, dict) and "content" in entry: docs.append({"text": entry['content'], "source": "wiki", "streamer": streamer, "idx": 0, "data": entry, "score": 2}) # Removed adding streamer name to text
    elif isinstance(wiki_data, list): # Wiki (list)
         for i, entry in enumerate(wiki_data): # Use i
             if isinstance(entry, dict) and "content" in entry and "streamer" in entry:
                 sn=entry["streamer"]
                 docs.append({"text": entry['content'], "source": "wiki", "streamer": sn, "idx": i, "data": entry, "score": 2}) # Use i, Removed adding streamer name
                 # wiki_idx_counter += 1 # Not needed
    for streamer, det in details_data.items(): # Details
         if isinstance(det, dict): docs.append({"text": str(det.get('Description', '')), "source": "details", "streamer": streamer, "idx": 0, "data": det, "score": 3}) # Removed adding streamer name
    return docs

def score_semantic_results(results, query):
    qt = set(re.findall(r"\w+", query.lower())); scored = []
    for doc in results:
        # --- FIX: Ensure doc is a dictionary and sim_score exists ---
        if not isinstance(doc, dict):
            print(f"Warning: Skipping non-dict item in score_semantic_results: {doc}")
            continue
        # Use .get() with a default for the base score
        score = doc.get("sim_score", 0.0)
        # --- End FIX ---

        text = doc.get("text", "").lower() # Default to empty string
        ss = doc.get("score", 1) # Source score (Reddit score or type score)

        # --- FIX: Check if source score ss is numeric before calculation ---
        is_numeric_ss = isinstance(ss, (int, float))
        # --- End FIX ---

        if doc["source"] == "wiki": score += 15
        elif doc["source"] == "details": score += 10
        elif doc["source"] == "reddit" and is_numeric_ss: # Check if numeric
             score += min(ss / 500.0, 20) if ss > 0 else 0 # Avoid division by zero if ss=0
        # Term frequency bonus
        score += sum(text.count(term) * 2 for term in qt if term) # Added check 'if term'
        # Exact phrase bonus
        if query and query.lower() in text: score += 30 # Added check 'if query'

        fmt = {
            "source": doc.get("source", "unknown"), # Added default
            "name": doc.get("streamer", "unknown"), # Added default
            "doc": text[:150] + ("..." if len(text) > 150 else ""),
            "semantic_score": round(score, 2),
            # Optionally keep the original sim_score if needed downstream
            "sim_score": doc.get("sim_score", 0.0)
        }
        if doc.get("source") == "reddit" and isinstance(doc.get("data"), dict):
             fmt["reddit_score"] = ss # Use the retrieved ss
             fmt["id"] = doc["data"].get("ID", "")
        scored.append((fmt, score))

    scored.sort(key=lambda x: x[1], reverse=True); return [d for d, _ in scored]


# ───────────────────────────────────────────────────────────────────────────────
# HYBRID SEARCH COMBINING RESULTS (As before)
# ───────────────────────────────────────────────────────────────────────────────
def combine_search_results_weighted_simple(
    boolean_results,
    semantic_results,
    boolean_weight=0.5, # Re-introduce weight
    semantic_weight=0.9, # Re-introduce weight
    score_threshold=5.0   # Keep threshold for single-source results
    ):

    combined_docs = {} # Key: unique_doc_id, Value: dict storing scores and doc_info

    # Helper to create a unique key (same as before)
    def get_doc_key(doc):
        if not isinstance(doc, dict): return None
        source = doc.get('source', 'unk')
        streamer = doc.get('name', 'unk')
        idx = doc.get('idx', -1)
        doc_id = doc.get('id', None) # Reddit ID

        if source == 'reddit' and doc_id:
            return f"{source}:{streamer}:{doc_id}"
        elif idx != -1:
            return f"{source}:{streamer}:{idx}"
        else:
            return f"{source}:{streamer}:{doc.get('doc', '')[:20]}" # Fallback

    # --- Step 1: Process Boolean Results ---
    for doc in boolean_results:
        key = get_doc_key(doc)
        if not key: continue
        if key not in combined_docs:
            combined_docs[key] = {
                "doc_info": doc.copy(),
                "boolean_score": doc.get("boolean_score", 0.0),
                "semantic_score": 0.0,
                "sim_score": 0.0
            }
        else:
            combined_docs[key]["boolean_score"] = max(combined_docs[key]["boolean_score"], doc.get("boolean_score", 0.0))

    # --- Step 2: Process Semantic Results ---
    for doc in semantic_results:
        key = get_doc_key(doc)
        if not key: continue

        semantic_score = doc.get("semantic_score", 0.0) # Boosted semantic score
        sim_score = doc.get("sim_score", 0.0) # Raw 0-100 similarity

        if key not in combined_docs:
             combined_docs[key] = {
                 "doc_info": doc.copy(),
                 "boolean_score": 0.0,
                 "semantic_score": semantic_score,
                 "sim_score": sim_score
             }
        else:
            combined_docs[key]["semantic_score"] = max(combined_docs[key]["semantic_score"], semantic_score)
            combined_docs[key]["sim_score"] = max(combined_docs[key].get("sim_score", 0.0), sim_score)
            # Overwrite doc_info with semantic version to ensure sim_score is present and doc content might be better
            combined_docs[key]["doc_info"] = doc.copy()

    # --- Step 3: Calculate Final Score based on Weighted/Threshold Rules & Filter ---
    scored_results = []
    for key, data in combined_docs.items():
        b_score = data["boolean_score"]
        s_score = data["semantic_score"] # Use the boosted semantic score
        final_score = 0.0

        # Apply the combination logic
        is_b_significant = b_score > 0 # Basic check if score exists
        is_s_significant = s_score > 0 # Basic check if score exists

        if is_b_significant and is_s_significant:
            # --- Rule 1: Both present, use weighted sum ---
            final_score = (b_score * boolean_weight) + (s_score * semantic_weight)
        elif b_score > score_threshold and not is_s_significant:
            # --- Rule 2a: Only boolean is above threshold ---
            final_score = b_score
        elif s_score > score_threshold and not is_b_significant:
             # --- Rule 2b: Only semantic is above threshold ---
             final_score = s_score
        # Else: final_score remains 0.0 (both zero, or neither meets criteria)

        # Only include results with a positive final score
        if final_score > 0:
            doc_info = data["doc_info"]
            doc_info["final_score"] = round(final_score, 2)
            # Ensure raw sim_score is present for display
            doc_info["sim_score"] = round(data.get("sim_score", 0.0), 2)
            scored_results.append(doc_info)

    # --- Step 4: Group by Streamer and Sort (Same as before) ---
    streamer_results = defaultdict(lambda: {"name": "", "documents": [], "max_final_score": 0.0})

    for doc_info in scored_results:
        streamer_name = doc_info.get("name", "unknown")
        if streamer_name != "unknown":
            streamer_results[streamer_name]["name"] = streamer_name
            streamer_results[streamer_name]["documents"].append(doc_info)
            streamer_results[streamer_name]["max_final_score"] = max(
                streamer_results[streamer_name]["max_final_score"],
                doc_info.get("final_score", 0.0)
            )

    final_list = list(streamer_results.values())
    final_list.sort(key=lambda x: x["max_final_score"], reverse=True)
    for streamer_data in final_list:
        streamer_data["documents"].sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    return final_list

# ───────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS (As before)
# ───────────────────────────────────────────────────────────────────────────────
def get_twitch_info(streamer_name):
    # (Function content remains the same as previous minimal version)
    sun = streamer_name.upper().strip()
    if sun in streamer_csv_data:
        data = streamer_csv_data[sun].copy(); tu = data.get("Twitch URL", "")
        if isinstance(tu, str) and tu.strip():
            if "url" not in data or not data["url"]: data["url"] = tu.strip()
            if "Name" not in data: data["Name"] = streamer_name # Ensure Name field exists
            return data
        else:
            data["url"] = f"https://www.twitch.tv/{streamer_name.replace(' ', '').lower()}"
            if "Name" not in data: data["Name"] = streamer_name # Ensure Name field exists
            return data
    else: return {"url": f"https://www.twitch.tv/{streamer_name.replace(' ', '').lower()}", "Name": streamer_name}


def get_streamer_image_path(streamer_name):
    # (Function content remains the same as previous minimal version)
    bp = os.path.join("static", "images", "streamer_images") # Use os.path.join
    vs = [streamer_name.upper(), streamer_name, streamer_name.lower(), streamer_name.replace(" ", ""), streamer_name.replace(" ", "_")]
    es = [".jpg", ".png", ".jpeg", ".webp"]
    for v in vs:
        for e in es:
            # Check existence using absolute path for reliability
            abs_pp = os.path.join(BACK, bp, f"{v}{e}") # Construct absolute path
            if os.path.exists(abs_pp):
                # Return relative path for HTML
                return f"images/streamer_images/{v}{e}"
    # Ensure default exists using absolute path check if possible
    default_img_abs = os.path.join(BACK, bp, "default.png")
    if os.path.exists(default_img_abs):
         return "images/streamer_images/default.png"
    else:
         print("Warning: Default image 'default.png' not found.")
         return "" # Return empty string or placeholder path if default is missing


def get_csv_streamer_info(streamer_name):
    # (Function content remains the same as previous minimal version)
    return streamer_csv_data.get(streamer_name.upper().strip(), {}).copy()

# ───────────────────────────────────────────────────────────────────────────────
# INITIALIZE SEARCH INDEXES
# ───────────────────────────────────────────────────────────────────────────────
print("Initializing search system...")

# --- Boolean Index (Loads or builds as before) ---
boolean_index = None
try:
    if os.path.exists(BOOLEAN_INDEX_PATH):
        with open(BOOLEAN_INDEX_PATH, "rb") as f: boolean_index = pickle.load(f); print(f"Loaded boolean index: {BOOLEAN_INDEX_PATH}")
    else:
        print("Building boolean index..."); boolean_index = create_boolean_index()
        with open(BOOLEAN_INDEX_PATH, "wb") as f: pickle.dump(boolean_index, f); print(f"Boolean index saved: {BOOLEAN_INDEX_PATH}")
except Exception as e:
    print(f"Error boolean index: {e}. Rebuilding..."); boolean_index = create_boolean_index() # Attempt rebuild
    try:
        with open(BOOLEAN_INDEX_PATH, "wb") as f: pickle.dump(boolean_index, f); print(f"Boolean index rebuilt: {BOOLEAN_INDEX_PATH}")
    except Exception as e_rebuild: print(f"FATAL: Failed rebuild boolean index: {e_rebuild}. Exiting."); sys.exit(1)

# --- SentenceTransformer Model ---
print(f"Loading SentenceTransformer model: {MODEL_NAME}")
try: model = SentenceTransformer(MODEL_NAME, device="cpu")
except Exception as e: print(f"Error loading SBERT model: {e}. Exiting."); sys.exit(1)

# Verify embedding dimension
try:
    actual_dim = model.encode(["Test"], convert_to_numpy=True).shape[1]
    if actual_dim != EMBED_DIM: print(f"Warning: Model dim ({actual_dim}) != config ({EMBED_DIM}). Using {actual_dim}."); EMBED_DIM = actual_dim
except Exception as e: print(f"Warning: Could not verify model dimension: {e}. Using config ({EMBED_DIM})")

# --- Load or Build FP16 Embeddings ---
DOCS = []
EMBEDDINGS = None
embeddings_loaded = False
loaded_embedding_dtype = None

build_target_path = EMBED_PATH_FP32
build_target_dtype = np.float32
loaded_path = None

if os.path.exists(EMBED_PATH_FP32) and os.path.exists(META_PATH):
    print(f"Attempting load FP32 embeddings: {EMBED_PATH_FP32}...")
    EMBEDDINGS = np.load(EMBED_PATH_FP32)
    with open(META_PATH, "rb") as f: DOCS = pickle.load(f)
    loaded_path = EMBED_PATH_FP32
    embeddings_loaded = True

# if not embeddings_loaded and os.path.exists(EMBED_PATH_FP32_FALLBACK) and os.path.exists(META_PATH):
#      try:
#         print(f"Attempting load fallback embeddings: {EMBED_PATH_FP32_FALLBACK}...")
#         EMBEDDINGS = np.load(EMBED_PATH_FP32_FALLBACK)
#         if EMBEDDINGS.dtype != EXPECTED_EMBED_DTYPE:
#              print(f"Warning: Loaded fallback '{EMBED_PATH_FP32_FALLBACK}' dtype {EMBEDDINGS.dtype}, expected {EXPECTED_EMBED_DTYPE}.")
#              print(f"Attempting conversion to {EXPECTED_EMBED_DTYPE}...")
#              EMBEDDINGS = EMBEDDINGS.astype(EXPECTED_EMBED_DTYPE)
#              print(f"Conversion complete. New dtype: {EMBEDDINGS.dtype}")
#              gc.collect()
#         with open(META_PATH, "rb") as f: DOCS = pickle.load(f)
#         loaded_path = EMBED_PATH_FP32_FALLBACK
#         embeddings_loaded = True
#      except Exception as e:
#          print(f"Warning: Failed load fallback {EMBED_PATH_FP32_FALLBACK}: {e}. Will attempt build.")
#          EMBEDDINGS, DOCS, embeddings_loaded = None, [], False

if embeddings_loaded:
    loaded_embedding_dtype = EMBEDDINGS.dtype
    print(f"Embeddings loaded from {loaded_path}. Shape: {EMBEDDINGS.shape}, Dtype: {loaded_embedding_dtype}")
    if loaded_embedding_dtype != EXPECTED_EMBED_DTYPE:
         print(f"Warning: Final loaded dtype {loaded_embedding_dtype} doesn't match expected {EXPECTED_EMBED_DTYPE}. Forcing rebuild.")
         embeddings_loaded = False
    # Ensure loaded dimension matches the *verified* model dimension
    elif EMBEDDINGS.shape[1] != EMBED_DIM:
         print(f"Error: Loaded dim ({EMBEDDINGS.shape[1]}) != verified model dim ({EMBED_DIM}). Forcing rebuild.")
         embeddings_loaded = False
    elif len(DOCS) != EMBEDDINGS.shape[0]:
        print(f"Error: Mismatch metadata ({len(DOCS)}) vs embeddings ({EMBEDDINGS.shape[0]}). Forcing rebuild.")
        embeddings_loaded = False
    else: print(f"OK: {len(DOCS)} documents and embeddings match.")

if not embeddings_loaded and BUILD_INDEX_ON_MISSING:
    print(f"Building semantic embeddings (Target: {build_target_dtype})...")
    DOCS = gather_documents();
    if not DOCS: print("Error: No documents to embed. Exiting."); sys.exit(1)
    print(f"Encoding {len(DOCS)} documents...")
    try:
        EMBEDDINGS = model.encode(
            [d.get("text","") for d in DOCS], # Use .get for safety
             batch_size=32, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True
        ).astype(build_target_dtype)

        loaded_embedding_dtype = EMBEDDINGS.dtype
        print(f"Embeddings generated: Shape {EMBEDDINGS.shape}, Dtype {loaded_embedding_dtype}")
        if loaded_embedding_dtype != EXPECTED_EMBED_DTYPE:
             print(f"Warning: Built dtype {loaded_embedding_dtype} != expected {EXPECTED_EMBED_DTYPE}?")

        np.save(build_target_path, EMBEDDINGS); print(f"Embeddings saved: {build_target_path}")
        with open(META_PATH, "wb") as f: pickle.dump(DOCS, f); print(f"Metadata saved: {META_PATH}")
        embeddings_loaded = True
    except Exception as e: print(f"Error during embedding build: {e}. Exiting."); sys.exit(1)
elif not embeddings_loaded:
    print("Error: Embeddings missing & build disabled/failed. Exiting."); sys.exit(1)

if not embeddings_loaded or EMBEDDINGS is None or EMBEDDINGS.dtype != EXPECTED_EMBED_DTYPE or len(DOCS) != EMBEDDINGS.shape[0]:
    print(f"FATAL: Embeddings not ready or inconsistent before FAISS init. Exiting.")
    sys.exit(1)

# --- Initialize FAISS Index with FP32 and L2 Distance ---
print(f"Initializing FAISS index with L2 distance for {EXPECTED_EMBED_DTYPE} data...")
index = None
embeddings_successfully_added = False
try:
    if not EMBEDDINGS.flags['C_CONTIGUOUS']:
        print("Embeddings not C-contiguous. Making copy...")
        EMBEDDINGS = np.ascontiguousarray(EMBEDDINGS)
        gc.collect()

    if EMBEDDINGS.shape[1] != EMBED_DIM:
         print(f"FATAL Error: Data dim ({EMBEDDINGS.shape[1]}) mismatch config ({EMBED_DIM}). Exiting.")
         sys.exit(1)

    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(EMBEDDINGS)
    print(f"FAISS index created (IndexFlatL2) with {index.ntotal} vectors (dtype: {EMBEDDINGS.dtype}).")
    if index.ntotal != len(DOCS):
         print(f"Warning: FAISS index count ({index.ntotal}) != DOCS count ({len(DOCS)}).")
    embeddings_successfully_added = True

except Exception as e:
    print(f"Error initializing/populating FAISS index: {e}")
    if EMBEDDINGS is not None: # Check if EMBEDDINGS exists before accessing attributes
        print(f"Context: Data shape: {EMBEDDINGS.shape}, dtype: {EMBEDDINGS.dtype}, dim: {EMBED_DIM}")
    else:
        print("Context: EMBEDDINGS variable is None.")
    sys.exit(1)

# --- Attempt to release original embeddings memory ---
if embeddings_successfully_added:
    if 'EMBEDDINGS' in locals() and EMBEDDINGS is not None:
        print("Attempting to release original embeddings array memory...")
        try:
            del EMBEDDINGS
            gc.collect()
            print("Original embeddings array deleted, gc collected.")
        except Exception as e_del:
            print(f"Warning: Error occurred during embeddings deletion/gc: {e_del}")
    else:
         print("Embeddings variable already released or not assigned.")
else:
    print("Skipping embeddings deletion because FAISS add failed.")

# ───────────────────────────────────────────────────────────────────────────────
# FLASK SERVER
# ───────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static')
CORS(app)

@app.route("/")
def home(): return render_template("base.html", title="Streamer Search")

@app.route("/search")
def search_streamer():
    query = request.args.get("name", "").strip();
    if not query: return jsonify([])
    print(f"\n--- Query: '{query}' ---")

    # 1. Boolean Search
    bool_raw = boolean_search(query, boolean_index)
    bool_res = score_boolean_results(bool_raw, query)

    # 2. Semantic Search
    sem_raw = []
    try:
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(EXPECTED_EMBED_DTYPE)

        if q_emb.shape[1] != index.d:
            print(f"Error: Query dim ({q_emb.shape[1]}) != index dim ({index.d}). Skipping semantic.")
        else:
            D, I = index.search(q_emb, TOP_K)

            for L2_dist_sq, idx in zip(D[0], I[0]):
                if idx == -1 or idx >= len(DOCS): continue
                # --- FIX: Check if L2_dist_sq is valid ---
                if not math.isfinite(L2_dist_sq) or L2_dist_sq < 0:
                    print(f"Warning: Skipping result with invalid L2 distance squared: {L2_dist_sq} for index {idx}")
                    continue
                # --- End FIX ---

                meta = DOCS[idx]
                # Ensure meta is a dict before proceeding
                if not isinstance(meta, dict):
                    print(f"Warning: Skipping result with non-dict metadata at index {idx}")
                    continue

                cosine_similarity = 1.0 - (L2_dist_sq / 2.0)
                # Clamp similarity to [0, 1] range before scaling
                sim_score = max(0.0, min(1.0, cosine_similarity)) * 100

                sem_raw.append({
                    "source": meta.get("source", "unknown"), # Use .get
                    "streamer": meta.get("streamer", "unknown"), # Use .get
                    "text": meta.get("text", ""), # Use .get
                    "score": meta.get("score", 1), # Original source score
                    "data": meta.get("data", {}),
                    "sim_score": round(sim_score, 2), # The 0-100 similarity score
                })
    except Exception as e: print(f"Error semantic search: {e}")

    sem_res = score_semantic_results(sem_raw, query)

    # 3. Combine Results
    comb_res = combine_search_results_weighted_simple(
    bool_res, sem_res,
    boolean_weight=BOOLEAN_WEIGHT,      # Use weight from config
    semantic_weight=SEMANTIC_WEIGHT,    # Use weight from config
    score_threshold=5.0                 # Adjust threshold if needed
)
    # 4. Format Final Output
    final_res = []
    for sd in comb_res[:10]: # Still limit streamers to top 10 overall
        sn = sd.get("name", "Unknown Streamer") # Use .get
        # --- FIX: Limit documents per streamer to 4 ---
        docs_limited = sd.get("documents", [])[:4]
        # --- End FIX ---

        final_res.append({
            "name": sn,
            "documents": docs_limited, # Use the limited list
            "twitch_info": get_twitch_info(sn),
            "image_path": get_streamer_image_path(sn),
            "csv_data": get_csv_streamer_info(sn),
            "max_combined_score": sd.get("max_final_score", 0.0) # Use max_final_score here
        })

    print(f"Returning {len(final_res)} combined streamer results.")
    return jsonify(final_res)

if __name__ == "__main__":
    if boolean_index is None or index is None or model is None or not DOCS:
        print("Error: Indexes or model not loaded properly. Exiting."); sys.exit(1)
    print("\n--- Starting Flask Server ---")
    app.run(debug=False, host="0.0.0.0", port=5001) #
