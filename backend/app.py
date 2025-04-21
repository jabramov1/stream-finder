import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from collections import defaultdict
import re

# Get the directory of this script (backend/)
current_directory = os.path.dirname(os.path.abspath(__file__))

# Load init.cleaned.json
init_path = os.path.join(current_directory, "init.cleaned.json")
with open(init_path, "r", encoding="utf-8") as f:
    combined_data = json.load(f)
reddit_data   = combined_data["reddit"]
twitter_data  = combined_data["twitter"]
wiki_data     = combined_data["wiki"]
details_data  = combined_data["details"]

# Load twitchtracker.json (one level up from backend/)
tt_path = os.path.abspath(os.path.join(current_directory, "..", "twitchtracker.json"))
with open(tt_path, "r", encoding="utf-8") as f:
    raw_tt = json.load(f)

# Flatten TwitchTracker data
twitch_data = {}
for slug, info in raw_tt.items():
    ch = info.get("channel_summary_30d") or {}
    ch["games_summary"] = info.get("games_summary_30d") or {}
    twitch_data[slug.lower()] = ch


# Load streamer_details.csv
csv_path = os.path.join(current_directory, "streamer_details.csv")
streamer_csv = pd.read_csv(csv_path).fillna("")
streamer_csv_data = {
    str(r["Name"]).upper().strip(): dict(r)
    for _, r in streamer_csv.iterrows()
}

def create_index():
    idx = defaultdict(list)
    # reddit
    for s, posts in reddit_data.items():
        for i, p in enumerate(posts):
            for w in re.findall(r"\w+", p["Title"].lower()):
                idx[w].append(("reddit", s, i))
    # twitter
    for s, tweets in twitter_data.items():
        for i, t in enumerate(tweets):
            for w in re.findall(r"\w+", t.lower()):
                idx[w].append(("twitter", s, i))
    # wiki
    if isinstance(wiki_data, dict):
        for s, e in wiki_data.items():
            txt = e.get("wikipedia_summary","").lower()
            for w in re.findall(r"\w+", txt):
                idx[w].append(("wiki", s, 0))
    else:
        for i, e in enumerate(wiki_data):
            txt = e.get("wikipedia_summary","").lower()
            s = e.get("streamer","")
            for w in re.findall(r"\w+", txt):
                idx[w].append(("wiki", s, i))
    # details
    for s, d in details_data.items():
        txt = str(d.get("Description","")).lower()
        for w in re.findall(r"\w+", txt):
            idx[w].append(("details", s, 0))
    return idx

def search(query, index):
    terms = re.findall(r"\w+", query.lower())
    if not terms:
        return []
    matches, info = defaultdict(int), {}
    for t in terms:
        for src, s, i in index.get(t, []):
            did = f"{src}:{s}:{i}"
            matches[did] += 1
            if did not in info:
                if src=="reddit":
                    post = reddit_data[s][i]
                    info[did] = {"source":src,"name":s,"text":post["Title"],"score":post["Score"],"idx":i}
                elif src=="twitter":
                    txt = twitter_data[s][i]
                    info[did] = {"source":src,"name":s,"text":txt,"score":1,"idx":i}
                else:
                    ent = wiki_data.get(s) if isinstance(wiki_data, dict) else wiki_data[i]
                    txt = ent.get("wikipedia_summary","") if ent else ""
                    info[did] = {"source":src,"name":s,"text":txt,"score":2,"idx":i}
    results=[]
    for did, cnt in matches.items():
        d = info[did]
        score = cnt*15 + sum(d["text"].lower().count(t)*5 for t in terms)
        if d["source"]=="reddit":
            score += min(d["score"]/500,20)
        elif d["source"]=="wiki":
            score += 15
        if " ".join(terms) in d["text"].lower():
            score += 50
        results.append((d,score))
    results.sort(key=lambda x:x[1], reverse=True)
    out=[]
    for d,_ in results:
        entry = {
            "source":d["source"],
            "name":d["name"],
            "doc":d["text"][:150]+"..." if len(d["text"])>150 else d["text"],
            "sim_score":round(_,2)
        }
        if d["source"]=="reddit":
            entry.update({
                "reddit_score":d["score"],
                "id":reddit_data[d["name"]][d["idx"]]["ID"]
            })
        out.append(entry)
    return out

def get_twitch_info(name):
    variants = [name,name.upper(),name.lower(),name.title(),name.replace(" ","")]
    for v in variants:
        if v in streamer_csv_data:
            d = streamer_csv_data[v]
            if d.get("Twitch URL"):
                return d
            d["url"] = f"https://www.twitch.tv/{name}"
            return d
    return None

def get_streamer_image_path(name):
    return f"images/streamer_images/{name.upper()}.jpg"

def get_csv_streamer_info(name):
    return streamer_csv_data.get(name.upper().strip())

app = Flask(__name__)
CORS(app)
search_index = create_index()

@app.route("/")
def home():
    return render_template("base.html")

@app.route("/search")
def search_streamer():
    q = request.args.get("name","").strip()
    if not q:
        return jsonify([])
    raw = search(q, search_index)[:50]
    grouped = {}
    for doc in raw:
        s = doc["name"]
        grouped.setdefault(s, {
            "documents":[],
            "twitch_info": get_twitch_info(s),
            "twitchtracker": twitch_data.get(s.lower())
        })
        grouped[s]["documents"].append(doc)
    out=[]
    for s, data in grouped.items():
        out.append({
            "name": s,
            "documents": data["documents"],
            "twitch_info": data["twitch_info"],
            "image_path": get_streamer_image_path(s),
            "csv_data": get_csv_streamer_info(s),
            "twitchtracker": data["twitchtracker"]
        })
    out.sort(key=lambda x: max(d["sim_score"] for d in x["documents"]), reverse=True)
    return jsonify(out)

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
