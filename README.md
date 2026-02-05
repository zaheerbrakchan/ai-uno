# Mini RAG System

```
Time: 75 Minutes
```

## Problem

Build a Retrieval-Augmented Generation (RAG) pipeline that can process documents, retrieve relevant information, and generate answers to questions.

## Dataset

A dataset of 100 synthetic emails is provided in the `emails/` directory. Each email contains:
- Subject line
- Sender and receiver information (from a pool of 200 unique people)
- Body content (100+ words) with diverse topics including:
  - Project updates
  - Meeting requests
  - Budget approvals
  - Technical issues
  - Client feedback
  - Team announcements
  - Deadline extensions
  - Training opportunities
  - Vendor proposals
  - Performance reviews

Use this dataset to test your RAG system's ability to:
- Chunk and process email documents
- Retrieve relevant information based on queries
- Generate accurate answers using the retrieved context

## Requirements

Your RAG system should implement:

1. **Document Chunking**
   - Split documents into appropriate chunks
   - Handle different document types and sizes
   - Explain your chunking strategy

2. **Embedding**
   - Generate embeddings for document chunks
   - Choose an appropriate embedding model
   - Store embeddings efficiently

3. **Retrieval**
   - Implement similarity search to find relevant chunks
   - Handle query embedding and matching
   - Return top-k most relevant results

4. **Generation**
   - Use retrieved context to generate answers
   - Design effective prompts
   - Integrate with a language model

## Constraints

- Do not use end-to-end RAG frameworks (e.g., LangChain, LlamaIndex)
- Build core components yourself or use individual libraries
- Document your design choices and tradeoffs
- Explain your approach to quality evaluation

## Submission

- Create a public git repository containing your submission and share the repository link
- Do not fork this repository or create pull requests

---

## Solution Overview

This submission implements a **custom RAG pipeline** (no LangChain, LlamaIndex, or other end-to-end RAG frameworks). Core components use **Python 3.8+** with `sentence-transformers`, `numpy`, `torch`, and `transformers` (individual libraries only). The **single entry point** is `rag.py`: load emails → chunk → embed → retrieve → generate.

| Requirement | Implementation |
|-------------|----------------|
| Document Chunking | `load_email()`, `chunk_document()`, `chunk_documents()` in `rag.py` |
| Embedding | `SentenceTransformer` (all-MiniLM-L6-v2), `embed_chunks()`; stored as NumPy array |
| Retrieval | Cosine similarity, `retrieve()`; top-k configurable |
| Generation | FLAN-T5-small via `AutoModelForSeq2SeqLM` + `AutoTokenizer`; prompt in `build_prompt()` |

---

## Project Structure

| File / Folder | Description |
|---------------|--------------|
| `rag.py` | Full RAG implementation: document loading, chunking, embedding, retrieval, generation, and CLI |
| `requirements.txt` | Dependencies: sentence-transformers, numpy, torch, transformers, accelerate |
| `emails/` | 100 synthetic email documents (email_001.txt … email_100.txt) |
| `generate_emails.py` | Original script that generated the email dataset (unchanged) |
| `README.md` | This file — problem, requirements, and solution documentation |

---

## How to Run

**Prerequisites:** Python 3.8 or higher, and `pip`.

```bash
# Install dependencies (first time only; may take a few minutes)
pip install -r requirements.txt

# Run with a single question (example)
python rag.py "Who sent emails about training opportunities?"

# Interactive mode: run without arguments, then type questions and press Enter
python rag.py
```

**What to expect:** On first run, the embedding model (`all-MiniLM-L6-v2`) and the generation model (`google/flan-t5-small`) are downloaded from Hugging Face (~100MB+ each). Then the script indexes all 100 emails into chunks, embeds them, and answers your question using the retrieved context. You may see a one-time warning about symlinks on Windows; it is safe to ignore, or set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to hide it.

**Example output (conceptually):**
```
Indexing emails (first run may download models)...
Loaded 100 emails, 395 chunks.
Indexing complete.

Question: Who sent emails about training opportunities?
Answer: Tara Woods and Gray Price (and others) sent emails about training.

Top retrieved chunks (scores):
  0.xxx: Subject: Training Opportunity\nFrom: Tara Woods...
  ...
```

### Where to Look in `rag.py`

| Requirement | Functions / logic |
|-------------|------------------|
| Document loading | `load_email()`, `load_emails()` |
| Chunking | `chunk_document()`, `chunk_documents()` |
| Embedding | `embed_chunks()`; `SentenceTransformer` in `RAGPipeline.index()` |
| Retrieval | `cosine_similarity()`, `retrieve()`; used in `RAGPipeline.query()` |
| Generation | `build_prompt()`, `RAGPipeline._get_gen_model()`, `RAGPipeline.query()` (generate + decode) |

---

### 1. Document Chunking

**Strategy:** Emails are parsed into Subject, From, To, and Body. Each chunk keeps the **header (Subject, From, To)** in every chunk so that retrieved snippets are self-contained. The body is split by paragraphs first; if a paragraph exceeds the max chunk size (~400 characters), it is split by sentences with an **overlap** of ~80 characters to avoid cutting meaning at boundaries.

- **Max chunk size:** 400 chars (fits embedding context, keeps coherence).
- **Overlap:** 80 chars so adjacent chunks share context.
- **No LangChain/LlamaIndex:** Chunking is implemented in `chunk_document()` / `chunk_documents()` in `rag.py`.

### 2. Embedding

- **Model:** `all-MiniLM-L6-v2` (sentence-transformers). Good tradeoff between speed and quality for short text; runs locally, no API key.
- **Storage:** NumPy array of shape `(n_chunks, embedding_dim)` kept in memory. For 100 emails and small chunks this is efficient; for scale, you could persist with `np.save` or a vector DB (e.g. FAISS).
- **Libraries:** Only `sentence-transformers` (and its dependencies), no end-to-end RAG framework.

### 3. Retrieval

- **Similarity:** Cosine similarity between query embedding and all chunk embeddings.
- **Top-k:** Default `k=5`; configurable in `RAGPipeline(top_k=...)`.
- **Flow:** Query is embedded with the same model → similarity scores computed → top-k chunks returned with scores.

### 4. Generation

- **Model:** `google/flan-t5-small` (HuggingFace). Runs locally, no API key; suitable for short, factual answers from context. The implementation uses `AutoModelForSeq2SeqLM` and `AutoTokenizer` directly (no `transformers` pipeline) for compatibility across library versions.
- **Prompt design:** The prompt instructs the model to answer **only** from the provided email excerpts and to say *"I don't have that information in the provided emails."* when the answer isn’t in the context, to reduce hallucination.
- **Context:** Top retrieved chunks are concatenated (with a cap of ~2000 characters) and passed in the prompt. See `build_prompt()` in `rag.py`.

### Design Choices and Tradeoffs

| Choice | Tradeoff |
|--------|----------|
| Paragraph-then-sentence chunking | Preserves coherence; long paragraphs may still be split. |
| In-memory NumPy embeddings | Simple and fast for 100 emails; not suitable for very large corpora without persistence. |
| all-MiniLM-L6-v2 | Fast and lightweight; larger models (e.g. all-mpnet-base) would improve retrieval at higher compute cost. |
| FLAN-T5-small | Runs offline and quickly; larger or instruction-tuned models would give better generation quality. |
| No reranking | Keeps pipeline simple; a reranker could improve precision after first-stage retrieval. |
| Direct model/tokenizer for generation | Avoids pipeline task name changes across transformers versions; same behavior as a seq2seq pipeline. |

### Quality Evaluation Approach

1. **Retrieval quality:** For a set of test questions with known relevant emails, check whether the correct email(s) appear in top-k (e.g. recall@k). Can also measure MRR (Mean Reciprocal Rank) of the first relevant chunk.
2. **Answer faithfulness:** Ensure the generated answer only uses information from the retrieved context (e.g. NLI-based entailment or sentence-level attribution).
3. **Relevance:** Manually or with a model, score whether the answer addresses the question (e.g. 1–5 scale or binary).
4. **End-to-end:** Run queries that have clear answers in the corpus (e.g. “Who wrote about budget approval?”) and verify correctness.

A minimal evaluation could be a small script that loads the RAG, runs a fixed set of queries, and prints retrieved doc IDs and answers for human review.
