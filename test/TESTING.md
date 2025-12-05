## Testing On-Prem Docs MCP Server

This guide shows how to test the **retrieval-only MCP server** using the included Python test scripts.

## RAG Testing Best Practices

When testing Retrieval-Augmented Generation (RAG) systems, it's important to evaluate both the **retrieval** and **generation** components. This MCP server focuses on retrieval, so our tests emphasize retrieval accuracy and relevance.

### Key Evaluation Metrics

1. **Precision@k**: Proportion of relevant documents among the top-k retrieved
   - Measures how many of the retrieved documents are actually relevant
   - Higher is better (range: 0.0 to 1.0)

2. **Recall@k**: Proportion of relevant documents successfully retrieved within top-k
   - Measures how many relevant documents were found
   - Higher is better (range: 0.0 to 1.0)

3. **Mean Reciprocal Rank (MRR)**: Evaluates the rank position of the first relevant document
   - Measures how quickly the system finds the first relevant result
   - Higher is better (range: 0.0 to 1.0)

4. **Normalized Discounted Cumulative Gain (NDCG@k)**: Considers relevance and position of retrieved documents
   - Accounts for both relevance and ranking quality
   - Higher is better (range: 0.0 to 1.0)

### Test Query Types

A comprehensive RAG test suite should include:

- **Simple factual queries**: Direct questions with clear answers
- **Complex multi-part queries**: Questions requiring information from multiple documents
- **Edge cases**: Misspellings, ambiguous queries, negative queries
- **Domain coverage**: Queries spanning all relevant topics
- **Difficulty levels**: Easy, medium, and hard queries to challenge the system

### Test Corpus Requirements

- **Diverse formats**: PDF, DOCX, Markdown, TXT files
- **Large scale**: Sufficient documents to challenge retrieval (100+ documents)
- **Varied content**: Different domains, topics, and writing styles
- **Ground truth**: Known relevant documents for each query

## Test Suites

### Setup

First, install test dependencies:

```bash
source venv/bin/activate
pip install -r test/requirements.txt
```

### 1. Basic functionality (ingestion + index loading)

Verify ingestion and retrieval work using only embeddings and vector search (no LLM needed):

```bash
source venv/bin/activate
cd test
python test_retrieval_only.py
```

This will:
- Ingest documents from `DATA_DIR` (default `./data`)
- Build and persist the index into `STORAGE_DIR` (default `./storage`)
- Perform a retrieval-only test over the index

### 2. End-to-end retrieval test (MCP server logic)

To exercise the same retrieval logic that the MCP server uses (`retrieve_docs`), run the corresponding pytest:

```bash
source venv/bin/activate
cd test
pytest test_system.py
```

This will:
- Ensure the index exists (or rebuild it)
- Call into the MCP server logic (via `retrieve_docs`) to retrieve chunks
- Print sample queries and information about the retrieved chunks (count, scores, previews)

You can also run the full test suite (including ingestion and system tests) with:

```bash
source venv/bin/activate
cd test
pytest
```

### 3. Large-Scale Automated Test Suite ⭐

The comprehensive large-scale test suite downloads a diverse corpus of documents from the web, ingests them, and evaluates retrieval performance using standard RAG metrics.

**Features:**
- Automatically downloads documents in all supported formats (PDF, DOCX, Markdown, TXT)
- Tests with diverse query types (simple, complex, edge cases)
- Calculates Precision@k, Recall@k, MRR, and NDCG@k metrics
- Evaluates performance by difficulty level
- Generates detailed JSON reports

**Run the large-scale test suite:**

**Important**: The large-scale test suite requires online access to download test documents. Run it with `OFFLINE=0`:

```bash
source venv/bin/activate
cd test
OFFLINE=0 python test_large_scale.py
```

**Note**: All tests must be run with `OFFLINE=0` to allow downloading test documents and models if needed. The test suite will download documents from the web, so an internet connection is required.

**What it does:**
1. Downloads a diverse corpus of documents from public sources:
   - Academic papers (Transformer, BERT, GPT-3 papers)
   - Technical documentation
   - Open-source project READMEs
   - Literary texts
   - Generated DOCX test documents
2. Ingests all documents into the RAG index
3. Runs evaluation queries with known ground truth
4. Calculates retrieval metrics at multiple k values (1, 3, 5, 10)
5. Generates aggregate metrics and difficulty-based breakdowns
6. Saves detailed results to `./test_results/test_results_<timestamp>.json`

**Configuration:**

You can customize the test using environment variables:

```bash
export TEST_DATA_DIR="./test_data"      # Where test documents are downloaded
export TEST_STORAGE_DIR="./test_storage"  # Where test index is stored
export TEST_RESULTS_DIR="./test_results"  # Where results are saved
export ABORT_ON_DOWNLOAD_FAILURE=1     # Abort if any file fails to download (default: 1)
```

**Note**: By default, the test suite will abort if any required file fails to download. Set `ABORT_ON_DOWNLOAD_FAILURE=0` to continue despite download failures (not recommended, as this may affect test quality).

**Expected Output:**

The test suite will print:
- Download progress for each document type
- Index building progress
- Per-query evaluation results with metrics
- Aggregate metrics across all queries
- Metrics broken down by difficulty level

**Example output:**
```
Aggregate Metrics:
  Precision@1: 0.750
  Precision@5: 0.680
  Recall@5: 0.850
  MRR: 0.820
  NDCG@5: 0.790

Metrics by Difficulty:
  easy:
    Count: 4
    Avg Precision@5: 0.850
    Avg Recall@5: 0.900
    Avg MRR: 0.950
  medium:
    Count: 5
    Avg Precision@5: 0.650
    Avg Recall@5: 0.800
    Avg MRR: 0.750
  hard:
    Count: 2
    Avg Precision@5: 0.550
    Avg Recall@5: 0.750
    Avg MRR: 0.650
```

**Interpreting Results:**

- **Precision@5 > 0.7**: Good - most retrieved documents are relevant
- **Recall@5 > 0.8**: Good - system finds most relevant documents
- **MRR > 0.8**: Good - first relevant result appears near the top
- **NDCG@5 > 0.75**: Good - ranking quality is high

If metrics are lower, consider:
- Adjusting embedding model parameters
- Tuning reranking model
- Improving chunking strategy
- Adding more training data or fine-tuning

### 4. Manual MCP server test

You can also run the MCP server directly:

```bash
source venv/bin/activate
python mcp_server.py
```

The server speaks MCP over stdio and is normally launched by an MCP client (such as Continue). Running it manually is useful for debugging, but you'll need an MCP-aware client to actually send tool calls.

## Continuous Testing

For CI/CD pipelines, run the large-scale test suite and check that metrics meet minimum thresholds:

```bash
cd test
OFFLINE=0 python test_large_scale.py
# Check that aggregate Precision@5 > 0.6, Recall@5 > 0.7, MRR > 0.7
```

This ensures that changes to the retrieval system don't degrade performance.

**Note**: Test files are located in the `test/` directory and are excluded from the release package. The release package is a standalone, offline-ready MCP server that does not include any test dependencies or test files.
