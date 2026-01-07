# RAG Parameter Optimization Evaluation Guide

## Understanding Your Results

### Why embed_top_k=10 Performed Best

**Your observation is partially correct, but there are important nuances:**

1. **Diminishing Returns**: Higher `embed_top_k` doesn't always mean better results because:
   - If the embedding model is good, the top 10 candidates already contain most/all relevant documents
   - Adding more candidates (50, 100) introduces noise (irrelevant documents)
   - The reranker can only work with what it's given - if relevant docs aren't in the pool, it can't find them
   - But if they're already in top 10, more candidates just add irrelevant ones

2. **Your Corpus Size**: 
   - 14 documents total
   - Most queries expect 1-2 relevant documents
   - With such a small corpus, the embedding model likely finds all relevant docs in the top 10
   - Going to 50 or 100 mostly adds irrelevant documents

3. **Quality vs Quantity Trade-off**:
   - `embed_top_k=10`: High precision pool for reranker
   - `embed_top_k=50`: More candidates, but lower average relevance
   - `embed_top_k=100`: Even more noise, reranker has to filter more

### Why rerank_top_k=10 Performed Best

**This is more nuanced than "bigger is always better":**

1. **Recall vs Precision Trade-off**:
   - `rerank_top_k=3`: Higher precision, but might miss relevant docs (lower recall)
   - `rerank_top_k=10`: Better recall (captures more relevant docs), but includes some irrelevant ones
   - Your metrics show: Precision@5=0.491, Recall@5=0.773
   - This suggests there ARE multiple relevant documents per query, so returning 10 helps recall

2. **Use Case Matters**:
   - **Information retrieval**: Higher recall (rerank_top_k=10) is often better
   - **Answer generation**: Lower precision might confuse the LLM
   - **Your use case**: RAG for documentation - you want to find all relevant chunks

3. **The "Bigger is Better" Fallacy**:
   - If you have 2 relevant documents, `rerank_top_k=10` returns 2 relevant + 8 irrelevant
   - If you have 2 relevant documents, `rerank_top_k=3` might return 2 relevant + 1 irrelevant (better precision)
   - But if you have 5 relevant documents, `rerank_top_k=3` misses 2 (worse recall)

## What Your Results Tell You

### 1. Embedding Model Quality
- **sentence-transformers/all-MiniLM-L6-v2** performed best
- This suggests it's doing a good job at initial ranking
- The fact that embed_top_k=10 works suggests the embedding model is effective

### 2. Corpus Characteristics
- Small corpus (14 documents)
- Most queries have 1-2 expected relevant documents
- This explains why embed_top_k=10 is sufficient

### 3. Reranker Effectiveness
- The reranker is working (composite score improved with reranking)
- rerank_top_k=10 suggests multiple relevant documents exist per query

## How to Properly Evaluate Parameters

### 1. **Understand Your Metrics**

- **Precision@k**: Of the top k results, how many are relevant?
  - Higher = less noise in results
  - Important for: Answer quality, user experience

- **Recall@k**: Of all relevant documents, how many did we find in top k?
  - Higher = we're not missing important information
  - Important for: Completeness, avoiding information loss

- **MRR (Mean Reciprocal Rank)**: Where is the first relevant document?
  - Higher = relevant docs appear earlier
  - Important for: User experience, answer quality

- **NDCG@k**: Ranking quality (considers position)
  - Higher = better ranking order
  - Important for: Overall retrieval quality

### 2. **Consider Your Use Case**

**For RAG (Retrieval-Augmented Generation):**
- **Recall is often more important** than precision
- You want to find ALL relevant information for the LLM to synthesize
- The LLM can filter out irrelevant information
- **Recommendation**: Optimize for Recall@k and NDCG@k

**For Direct Search/Display:**
- **Precision is more important**
- Users see results directly
- Too many irrelevant results hurt UX
- **Recommendation**: Optimize for Precision@k

### 3. **Test with Different Corpus Sizes**

Your current corpus is small (14 docs). To get more meaningful results:

**Small Corpus (< 50 docs):**
- embed_top_k: 10-20 is usually sufficient
- rerank_top_k: 5-10 is usually sufficient
- Focus on: Precision (less noise matters more)

**Medium Corpus (50-500 docs):**
- embed_top_k: 20-50 often works best
- rerank_top_k: 5-10
- Focus on: Balance of precision and recall

**Large Corpus (> 500 docs):**
- embed_top_k: 50-100 may be needed
- rerank_top_k: 10-20
- Focus on: Recall (more relevant docs exist)

### 4. **Parameter Interaction**

**embed_top_k and rerank_top_k interact:**

- **High embed_top_k + Low rerank_top_k**: 
  - Reranker has many candidates, picks best few
  - Good if embedding model is noisy
  - Example: embed_top_k=100, rerank_top_k=5

- **Low embed_top_k + High rerank_top_k**:
  - Fewer candidates, but return more
  - Good if embedding model is precise
  - Example: embed_top_k=10, rerank_top_k=10 (your best config!)

- **High embed_top_k + High rerank_top_k**:
  - Maximum recall, but more noise
  - Good for comprehensive search
  - Example: embed_top_k=100, rerank_top_k=20

### 5. **Evaluate Per Query Type**

Your test suite has different query types:
- **Easy queries**: 1 expected document → Lower rerank_top_k might be better
- **Hard queries**: 2+ expected documents → Higher rerank_top_k helps
- **Broad queries**: Many relevant documents → Higher rerank_top_k essential

**Recommendation**: Analyze metrics by query difficulty to understand parameter impact.

## Recommendations for Your Setup

### 1. **For Your Current Corpus (Small)**
Your results make sense:
- ✅ `embed_top_k=10`: Sufficient for small corpus
- ✅ `rerank_top_k=10`: Good recall without too much noise
- ✅ `sentence-transformers/all-MiniLM-L6-v2`: Best embedding model

### 2. **For Production (Larger Corpus)**
You should re-optimize when you have:
- More documents (100+)
- More diverse queries
- Real user queries

**Expected changes:**
- `embed_top_k` might increase to 20-50
- `rerank_top_k` might stay at 10 or increase to 15
- Different embedding model might perform better

### 3. **Improve Your Evaluation**

**Add these analyses:**

1. **Per-query analysis**: Which queries benefit from higher embed_top_k?
2. **Per-difficulty analysis**: Do hard queries need different parameters?
3. **Recall analysis**: Are you missing relevant documents with current settings?
4. **Precision analysis**: How much noise are you introducing?

### 4. **Composite Score Weighting**

Your current composite score weights all metrics equally (0.25 each). Consider:

**For RAG use case:**
```python
composite = (
    0.15 * precision +  # Less weight on precision
    0.35 * recall +     # More weight on recall
    0.25 * mrr +        # Keep MRR
    0.25 * ndcg         # Keep NDCG
)
```

**For direct search:**
```python
composite = (
    0.35 * precision +  # More weight on precision
    0.15 * recall +     # Less weight on recall
    0.25 * mrr +
    0.25 * ndcg
)
```

## Conclusion

**Your results DO make sense:**
1. ✅ Small corpus → embed_top_k=10 is sufficient
2. ✅ Good embedding model → finds relevant docs early
3. ✅ Multiple relevant docs per query → rerank_top_k=10 helps recall
4. ✅ Reranker is effective → improves ranking quality

**Next steps:**
1. Test with a larger corpus (your actual documentation)
2. Analyze per-query and per-difficulty metrics
3. Consider adjusting composite score weights for your use case
4. Re-optimize when you have production data

The optimization script is working correctly - your results reflect the characteristics of your test corpus!

