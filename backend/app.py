import os
import json
import pickle
import sys
import numpy as np
import re
import math
from sentence_transformers import SentenceTransformer
import faiss
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from collections import defaultdict
import gc

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────
MODEL_NAME           = "sentence-transformers/all-MiniLM-L6-v2"
META_PATH            = "metadata.pkl"
BOOLEAN_INDEX_PATH   = "boolean_index.pkl"
EMBED_PATH_FP16      = "embeddings_fp16.npy"
EMBED_PATH_FP32_FALLBACK = "embeddings.npy"
EXPECTED_EMBED_DTYPE = np.float16
BUILD_INDEX_ON_MISSING   = True
TOP_K                = 20
EMBED_DIM            = 768
SEMANTIC_WEIGHT      = 0.5
BOOLEAN_WEIGHT       = 0.5

# ───────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ───────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load init.json
with open(os.path.join(BASE_DIR, "init.json"), "r", encoding="utf-8") as f:
    init = json.load(f)
reddit_data  = init.get("reddit", {})
twitter_data = init.get("twitter", {})
wiki_data    = init.get("wiki", {})
details_data = init.get("details", {})

# Load twitchtracker.json
with open(os.path.join(BASE_DIR, "..", "twitchtracker.json"), "r", encoding="utf-8") as f:
    raw_tt = json.load(f)
twitch_data = {}
for slug, info in raw_tt.items():
    summary = info.get("channel_summary_30d") or {}
    summary["games_summary"] = info.get("games_summary_30d") or {}
    twitch_data[slug.lower()] = summary

# Load streamer_details.csv
csv_path = os.path.join(BASE_DIR, "streamer_details.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path).fillna("")
    streamer_csv_data = {str(r["Name"]).upper().strip(): dict(r) for _, r in df.iterrows()}
else:
    streamer_csv_data = {}

# ───────────────────────────────────────────────────────────────────────────────
# BOOLEAN SEARCH INDEX
# ───────────────────────────────────────────────────────────────────────────────
def create_boolean_index():
    idx = defaultdict(list)
    # Reddit
    for s, posts in reddit_data.items():
        if isinstance(posts, list):
            for i, p in enumerate(posts):
                if isinstance(p, dict):
                    for w in re.findall(r"\w+", p.get("Title", "").lower()):
                        idx[w].append(("reddit", s, i))
    # Twitter
    for s, tweets in twitter_data.items():
        if isinstance(tweets, list):
            for i, t in enumerate(tweets):
                for w in re.findall(r"\w+", str(t).lower()):
                    idx[w].append(("twitter", s, i))
    # Wiki
    if isinstance(wiki_data, dict):
        for s, e in wiki_data.items():
            if isinstance(e, dict):
                txt = e.get("wikipedia_summary", "").lower()
                for w in re.findall(r"\w+", txt): idx[w].append(("wiki", s, 0))
    else:
        for i, e in enumerate(wiki_data or []):
            if isinstance(e, dict) and "streamer" in e:
                txt = e.get("wikipedia_summary", "").lower()
                for w in re.findall(r"\w+", txt): idx[w].append(("wiki", e["streamer"], i))
    # Details
    for s, d in details_data.items():
        if isinstance(d, dict):
            txt = str(d.get("Description", "")).lower()
            for w in re.findall(r"\w+", txt): idx[w].append(("details", s, 0))
    return idx

def boolean_search(query, idx):
    terms = re.findall(r"\w+", query.lower())
    if not terms: return []
    matches, info = defaultdict(int), {}
    for t in terms:
        for src, s, i in idx.get(t, []):
            did = f"{src}:{s}:{i}"
            matches[did] += 1
            if did not in info:
                # retrieve text and score
                ent = None; text=""; score=1
                try:
                    if src=="reddit": ent = reddit_data[s][i]; text=ent.get("Title",""); score=ent.get("Score",1)
                    elif src=="twitter": text=str(twitter_data[s][i])
                    elif src=="wiki":
                        ent = wiki_data.get(s) if isinstance(wiki_data, dict) else wiki_data[i]
                        text = ent.get("wikipedia_summary","") if isinstance(ent, dict) else ""
                        score=2
                    else:
                        ent = details_data.get(s); text=str(ent.get("Description","")) if isinstance(ent, dict) else ""; score=3
                except: pass
                info[did] = {"source":src,"streamer":s,"text":text,"score":score,"idx":i,"term_matches":0}
    for did, cnt in matches.items():
        if did in info: info[did]["term_matches"] = cnt
    # scoring
    scored=[]
    for doc in info.values():
        s=doc["term_matches"]*15 + sum(doc["text"].lower().count(t)*5 for t in terms)
        if doc["source"]=="reddit": s+=min(doc["score"]/500,20)
        if doc["source"]=="wiki": s+=15
        if query.lower() in doc["text"].lower(): s+=50
        fmt={"source":doc["source"],"name":doc["streamer"],"doc":doc["text"][:150]+("..." if len(doc["text"])>150 else ""),"boolean_score":round(s,2),"idx":doc["idx"]}
        if doc["source"]=="reddit": fmt.update({"reddit_score":doc["score"],"id":doc.get("id")})
        scored.append((fmt,s))
    scored.sort(key=lambda x:-x[1])
    return [d for d,_ in scored]

# ───────────────────────────────────────────────────────────────────────────────
# SEMANTIC SEARCH UTILITIES
# ───────────────────────────────────────────────────────────────────────────────
def gather_documents():
    docs=[]
    for s, posts in reddit_data.items():
        for i, p in enumerate(posts or []):
            if isinstance(p, dict): docs.append({"text":p.get("Title",""),"source":"reddit","streamer":s,"idx":i,"data":p,"score":p.get("Score",1)})
    for s, tw in twitter_data.items():
        for i, t in enumerate(tw or []): docs.append({"text":str(t),"source":"twitter","streamer":s,"idx":i,"data":t,"score":1})
    if isinstance(wiki_data, dict):
        for s, e in wiki_data.items():
            if isinstance(e, dict): docs.append({"text":e.get("wikipedia_summary",""),"source":"wiki","streamer":s,"idx":0,"data":e,"score":2})
    else:
        for i, e in enumerate(wiki_data or []):
            if isinstance(e, dict): docs.append({"text":e.get("wikipedia_summary",""),"source":"wiki","streamer":e.get("streamer",""),"idx":i,"data":e,"score":2})
    for s, d in details_data.items():
        if isinstance(d, dict): docs.append({"text":str(d.get("Description","")),"source":"details","streamer":s,"idx":0,"data":d,"score":3})
    return docs

def score_semantic_results(results, query):
    terms=set(re.findall(r"\w+",query.lower()))
    scored=[]
    for doc in results:
        score=doc.get("sim_score",0)
        text=doc.get("doc","?").lower()
        ss=doc.get("score",1)
        if doc["source"]=="reddit": score+=min(ss/500,20) if isinstance(ss,(int,float)) else 0
        if doc["source"]=="wiki": score+=15
        if doc["source"]=="details": score+=10
        score+=sum(text.count(t)*2 for t in terms)
        if query.lower() in text: score+=30
        fmt={"source":doc["source"],"name":doc["name"],"doc":text[:150]+("..." if len(text)>150 else ""),"semantic_score":round(score,2),"sim_score":doc.get("sim_score",0),"idx":doc.get("idx"),"data":doc.get("data")}
        if doc["source"]=="reddit": fmt["reddit_score"]=ss; fmt["id"]=doc.get("id")
        scored.append((fmt,score))
    scored.sort(key=lambda x:-x[1])
    return [d for d,_ in scored]

# ───────────────────────────────────────────────────────────────────────────────
# HYBRID COMBINATION
# ───────────────────────────────────────────────────────────────────────────────
def combine_search_results_weighted_simple(bool_res, sem_res, boolean_weight, semantic_weight, score_threshold):
    combined={}
    def key(doc): return f"{doc['source']}:{doc['name']}:{doc.get('id',doc.get('idx',''))}"
    for doc in bool_res:
        k=key(doc)
        combined.setdefault(k,{"doc_info":doc.copy(),"boolean_score":doc.get("boolean_score",0),"semantic_score":0,"sim_score":0,"max_final_score":0})
        combined[k]["boolean_score"]=max(combined[k]["boolean_score"], doc.get("boolean_score",0))
    for doc in sem_res:
        k=key(doc); s=doc.get("semantic_score",0); sim=doc.get("sim_score",0)
        if k not in combined:
            combined[k]={"doc_info":doc.copy(),"boolean_score":0,"semantic_score":s,"sim_score":sim,"max_final_score":0}
        else:
            combined[k]["semantic_score"]=max(combined[k]["semantic_score"],s)
            combined[k]["sim_score"]=max(combined[k]["sim_score"],sim)
            combined[k]["doc_info"]=doc.copy()
    results=[]
    for v in combined.values():
        b=v["boolean_score"]; s=v["semantic_score"]; final=0
        if b>0 and s>0: final=b*boolean_weight + s*semantic_weight
        elif b>score_threshold and s==0: final=b
        elif s>score_threshold and b==0: final=s
        if final>0:
            di=v["doc_info"]; di["final_score"]=round(final,2); di["sim_score"]=round(v.get("sim_score",0),2)
            results.append(di)
    # group by streamer
    sr=defaultdict(lambda:{"name":"","documents":[],"max_final_score":0})
    for d in results:
        nm=d.get("name","?"); sr[nm]["name"]=nm; sr[nm]["documents"].append(d)
        sr[nm]["max_final_score"]=max(sr[nm]["max_final_score"], d.get("final_score",0))
    lst=list(sr.values()); lst.sort(key=lambda x:-x["max_final_score"])
    for i in lst: i["documents"].sort(key=lambda x:-x.get("final_score",0))
    return lst

# ───────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────
def get_twitch_info(name):
    key=name.upper().strip(); info=streamer_csv_data.get(key,{}).copy()
    tu=info.get("Twitch URL","")
    if isinstance(tu,str) and tu.strip(): info.setdefault("url",tu); info.setdefault("Name",name); return info
    info.setdefault("url",f"https://www.twitch.tv/{name.replace(' ','').lower()}"); info.setdefault("Name",name); return info

def get_streamer_image_path(name):
    return f"images/streamer_images/{name.upper()}.jpg"

def get_csv_streamer_info(name):
    return streamer_csv_data.get(name.upper().strip(),{})

# ───────────────────────────────────────────────────────────────────────────────
# INITIALIZATION & INDEX BUILDING
# ───────────────────────────────────────────────────────────────────────────────
boolean_index = None
if os.path.exists(BOOLEAN_INDEX_PATH):
    try:
        with open(BOOLEAN_INDEX_PATH,'rb') as f: boolean_index=pickle.load(f)
    except: boolean_index=create_boolean_index(); pickle.dump(boolean_index, open(BOOLEAN_INDEX_PATH,'wb'))
else:
    boolean_index=create_boolean_index(); pickle.dump(boolean_index, open(BOOLEAN_INDEX_PATH,'wb'))

print(f"Loading SBERT model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device="cpu")
# FAISS index initialization omitted for brevity

# ───────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ───────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static')
CORS(app)

@app.route("/")
def home():
    return render_template("base.html", title="Streamer Search")

@app.route("/search")
def search_streamer():
    query = request.args.get("name","" ).strip()
    cat_filter = request.args.get("category","all").lower()
    if not query: return jsonify([])
    bool_raw = boolean_search(query, boolean_index)
    bool_scored = score_boolean_results(bool_raw, query)
    # semantic search and combining omitted for brevity
    combined = combine_search_results_weighted_simple(bool_scored, sem_scored, BOOLEAN_WEIGHT, SEMANTIC_WEIGHT, 5.0)
    final=[]
    for sd in combined[:10]:
        name=sd['name']; tt=twitch_data.get(name.lower(),{}); avg=tt.get('avg_viewers')
        if avg is None: category='unknown'
        elif avg>20000: category='big'
        elif avg>=10000: category='medium'
        else: category='small'
        if cat_filter!='all' and category!=cat_filter: continue
        docs=sd['documents'][:4]
        final.append({
            'name':name,'documents':docs,'twitch_info':get_twitch_info(name),'image_path':get_streamer_image_path(name),
            'csv_data':get_csv_streamer_info(name),'twitchtracker':tt,'category':category,'max_combined_score':sd.get('max_final_score')
        })
    return jsonify(final)

if __name__=="__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)