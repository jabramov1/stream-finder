import json
import os
import pandas as pd

def combine_data():
    # ─── Paths ────────────────────────────────────────────────────────────────
    root = os.path.dirname(os.path.abspath(__file__))
    reddit_json_path  = os.path.join(root, "reddit.json")
    twitter_json_path = os.path.join(root, "twitter.json")
    wiki_json_path    = os.path.join(root, "wikitest.json")
    details_csv_path  = os.path.join(root, "backend", "streamer_details.csv")
    init_json_path    = os.path.join(root, "backend", "init.json")

    # ─── Load JSON sources ─────────────────────────────────────────────────────
    with open(reddit_json_path,  "r", encoding="utf-8") as f:
        reddit_data = json.load(f)
    with open(twitter_json_path, "r", encoding="utf-8") as f:
        twitter_data = json.load(f)
    with open(wiki_json_path,    "r", encoding="utf-8") as f:
        wiki_data = json.load(f)

    # ─── Load CSV AS STRINGS & DROP NaN ─────────────────────────────────────────
    # dtype=str forces every column to be read as a string,
    # then fillna("") replaces any missing cell with the empty string.
    details_df = pd.read_csv(details_csv_path, dtype=str).fillna("")

    # Strip whitespace from Name column, then build the dict
    details_df["Name"] = details_df["Name"].str.strip()
    details_data = details_df.set_index("Name").to_dict(orient="index")

    # ─── Combine & Dump ─────────────────────────────────────────────────────────
    combined = {
        "reddit":  reddit_data,
        "twitter": twitter_data,
        "wiki":    wiki_data,
        "details": details_data
    }

    with open(init_json_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"✅ Successfully wrote all data to {init_json_path}")

if __name__ == "__main__":
    combine_data()
