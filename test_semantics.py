#!/usr/bin/env python3
"""
test_embeddings.py - Comprehensive embedding test suite
Tests embedding quality, search functionality, and performance
"""
import os
import sys
import time
from pathlib import Path
import pickle
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

# Initialize rich console
console = Console()

# Environment settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def find_data_files():
    """Find embedding and metadata files"""
    base_dir = Path(__file__).parent
    emb_files = list(base_dir.glob("**/embeddings.npy"))
    meta_files = list(base_dir.glob("**/metadata.pkl"))
    
    if not emb_files or not meta_files:
        console.print("[red]❌ Could not find embedding or metadata files!")
        sys.exit(1)
        
    return str(emb_files[0]), str(meta_files[0])

def check_embeddings(embeddings):
    """Validate embedding quality"""
    issues = []
    
    # Check for NaN values
    if np.isnan(embeddings).any():
        issues.append("Contains NaN values")
    
    # Check for zero vectors
    zero_vectors = np.where(np.all(embeddings == 0, axis=1))[0]
    if len(zero_vectors) > 0:
        issues.append(f"Contains {len(zero_vectors)} zero vectors")
    
    # Check normalization
    norms = np.linalg.norm(embeddings, axis=1)
    unnormalized = np.where(np.abs(norms - 1.0) > 1e-6)[0]
    if len(unnormalized) > 0:
        issues.append(f"{len(unnormalized)} vectors not normalized")
    
    return issues

def run_search_test(model, embeddings, metadata, query, k=5):
    """Test search functionality for a single query"""
    try:
        # Encode query
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        if q_emb.dtype != np.float32:
            q_emb = q_emb.astype(np.float32)
        
        # Search
        index = faiss.IndexFlatIP(embeddings.shape[1])
        if not embeddings.flags['C_CONTIGUOUS']:
            embeddings = np.ascontiguousarray(embeddings)
        index.add(embeddings)
        D, I = index.search(q_emb, k)
        
        # Create results table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Rank")
        table.add_column("Streamer")
        table.add_column("Similarity")
        
        for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
            if idx < 0 or idx >= len(metadata):
                continue
            sim_score = max(0.0, min(1.0, score)) * 100
            table.add_row(
                str(rank),
                metadata[idx]['streamer'],
                f"{sim_score:.1f}%"
            )
        
        return True, table
    except Exception as e:
        return False, str(e)

def run_benchmark(model, embeddings, n_queries=50):
    """Run performance benchmark"""
    test_queries = [
        "gaming", "streaming", "just chatting", "esports",
        "variety", "speedrun", "minecraft", "valorant"
    ]
    
    # Create index once
    index = faiss.IndexFlatIP(embeddings.shape[1])
    if not embeddings.flags['C_CONTIGUOUS']:
        embeddings = np.ascontiguousarray(embeddings)
    index.add(embeddings)
    
    # Benchmark
    times = []
    with Progress() as progress:
        task = progress.add_task("[cyan]Running benchmark...", total=n_queries)
        
        for _ in range(n_queries):
            query = np.random.choice(test_queries)
            
            start = time.perf_counter()
            # Encode
            q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            if q_emb.dtype != np.float32:
                q_emb = q_emb.astype(np.float32)
            # Search
            D, I = index.search(q_emb, 5)
            
            end = time.perf_counter()
            times.append(end - start)
            
            progress.update(task, advance=1)
    
    return np.array(times)

def main():
    console.print("\n[bold blue]=== Embedding System Test Suite ===[/bold blue]\n")

    # Load files
    try:
        EMBED_PATH, META_PATH = find_data_files()
        console.print(f"📁 Found embedding file: [green]{EMBED_PATH}[/green]")
        console.print(f"📁 Found metadata file: [green]{META_PATH}[/green]")
    except Exception as e:
        console.print(f"[red]❌ File loading failed: {e}")
        return

    # Load model
    console.print("\n[bold]1. Testing SBERT Model[/bold]")
    try:
        model = SentenceTransformer("intfloat/e5-base-v2", device="cpu")
        test_emb = model.encode(
            ["test query"],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        console.print(f"✅ Model test passed! Shape: {test_emb.shape}")
    except Exception as e:
        console.print(f"[red]❌ Model test failed: {e}")
        return

    # Load and validate data
    console.print("\n[bold]2. Loading & Validating Data[/bold]")
    try:
        embeddings = np.load(EMBED_PATH)
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
        
        console.print(f"✅ Loaded embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
        console.print(f"✅ Loaded metadata: {len(metadata)} entries")
        
        if embeddings.shape[0] != len(metadata):
            console.print("[red]❌ Warning: Embedding count doesn't match metadata count!")
        
        # Check embedding quality
        issues = check_embeddings(embeddings)
        if issues:
            for issue in issues:
                console.print(f"[red]❌ {issue}")
        else:
            console.print("✅ No quality issues found in embeddings")
            
    except Exception as e:
        console.print(f"[red]❌ Loading failed: {e}")
        return

    # Test search functionality
    console.print("\n[bold]3. Testing Search Functionality[/bold]")
    test_queries = [
        "coding python",
        "variety gaming fun",
        "competitive esports",
        "just chatting irl"
    ]
    
    for query in test_queries:
        console.print(f"\n[cyan]Query: '{query}'[/cyan]")
        success, result = run_search_test(model, embeddings, metadata, query)
        if success:
            console.print(result)
        else:
            console.print(f"[red]❌ Search failed: {result}")

    # Run benchmark
    console.print("\n[bold]4. Running Performance Benchmark[/bold]")
    times = run_benchmark(model, embeddings)
    
    console.print("\n[bold]Benchmark Results:[/bold]")
    console.print(f"Average time: {times.mean()*1000:.1f}ms")
    console.print(f"95th percentile: {np.percentile(times, 95)*1000:.1f}ms")
    console.print(f"Max time: {times.max()*1000:.1f}ms")

if __name__ == "__main__":
    main()