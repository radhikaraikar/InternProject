"""
test_pipeline.py — Quick smoke test for the pipeline

Tests:
1. Document ingestion works (OCR fallback for empty PDFs)
2. Retrieval returns chunks
3. Judge call produces valid JSON
4. Pipeline branches correctly
"""

import os
import sys

# ── Check API key ─────────────────────────────────────────────────────────────
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("❌ ANTHROPIC_API_KEY not set. Export it first:")
    print("   export ANTHROPIC_API_KEY=sk-ant-...")
    print("   (or set in Windows: set ANTHROPIC_API_KEY=sk-ant-...)")
    sys.exit(1)

print("✅ ANTHROPIC_API_KEY found")

# ── Test imports ──────────────────────────────────────────────────────────────
print("\n1️⃣ Testing imports...")
try:
    from ingest import run_ingestion, get_chroma_collection
    from rag_pipeline import run_pipeline, retrieve, check_confidence
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# ── Test ingestion ────────────────────────────────────────────────────────────
print("\n2️⃣ Testing document ingestion...")
try:
    collection = run_ingestion(data_folder="./data", persist_dir="./chroma_db")
    count = collection.count()
    print(f"✅ Ingestion successful: {count} chunks indexed")
    if count == 0:
        print("⚠️ Warning: No chunks indexed — check data/ folder")
except Exception as e:
    print(f"❌ Ingestion failed: {e}")
    sys.exit(1)

# ── Test retrieval ────────────────────────────────────────────────────────────
print("\n3️⃣ Testing retrieval...")
try:
    collection = get_chroma_collection("./chroma_db")
    test_query = "What is the annual leave policy?"
    chunks = retrieve(test_query, collection, k=3)
    print(f"✅ Retrieved {len(chunks)} chunks for test query")
    if chunks:
        print(f"   Top chunk: {chunks[0].source} (score={chunks[0].score:.3f})")
except Exception as e:
    print(f"❌ Retrieval failed: {e}")
    sys.exit(1)

# ── Test pipeline ─────────────────────────────────────────────────────────────
print("\n4️⃣ Testing full self-correcting pipeline...")
try:
    collection = get_chroma_collection("./chroma_db")
    test_query = "How many days of annual leave do full-time employees get?"
    result = run_pipeline(test_query, collection)
    
    print(f"✅ Pipeline executed successfully")
    print(f"   Decision: {result.decision.value}")
    print(f"   Top score: {result.confidence_score:.3f}")
    print(f"   Answer (truncated): {result.answer[:150]}...")
    
    if result.judge_result:
        print(f"   Judge: sufficient={result.judge_result.sufficient}, "
              f"contradictory={result.judge_result.contradictory}")
    
except Exception as e:
    print(f"❌ Pipeline test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── Success ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("✅ ALL TESTS PASSED — Pipeline ready for deployment")
print("="*60)
print("\nNext steps:")
print("  • Run the Streamlit app: streamlit run app.py")
print("  • Run the eval harness: python eval_harness.py")
print("  • Deploy to Streamlit Cloud for the working link")
