#!/usr/bin/env python3
"""
Mini RAG System - Retrieval-Augmented Generation pipeline.
No end-to-end RAG frameworks (LangChain, LlamaIndex).
Implements: chunking, embedding, retrieval, generation.
"""
import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


# --- Document loading ---
def load_email(path: Path) -> dict:
    """Parse a single email file into structured fields."""
    text = path.read_text(encoding="utf-8")
    lines = text.strip().split("\n")
    subject = ""
    from_line = ""
    to_line = ""
    body_lines = []
    in_body = False
    for line in lines:
        if line.startswith("Subject:"):
            subject = line.replace("Subject:", "").strip()
        elif line.startswith("From:"):
            from_line = line.replace("From:", "").strip()
        elif line.startswith("To:"):
            to_line = line.replace("To:", "").strip()
        elif line.strip() == "" and (subject or from_line):
            in_body = True
        elif in_body:
            body_lines.append(line)
    body = "\n".join(body_lines).strip()
    return {
        "path": str(path),
        "subject": subject,
        "from": from_line,
        "to": to_line,
        "body": body,
        "full_text": f"Subject: {subject}\nFrom: {from_line}\nTo: {to_line}\n\n{body}",
    }


def load_emails(emails_dir: str) -> List[dict]:
    """Load all email documents from directory."""
    path = Path(emails_dir)
    emails = []
    for f in sorted(path.glob("email_*.txt")):
        emails.append(load_email(f))
    return emails


# --- Chunking strategy ---
def chunk_document(doc: dict, max_chunk_chars: int = 400, overlap_chars: int = 80) -> List[dict]:
    """
    Chunk a single email into overlapping segments.
    Strategy: Preserve subject/from/to in every chunk for context. Split body by paragraphs
    first; if a paragraph exceeds max_chunk_chars, split by sentences with overlap.
    """
    chunks = []
    header = f"Subject: {doc['subject']}\nFrom: {doc['from']}\nTo: {doc['to']}\n\n"
    body = doc["body"]
    if not body:
        chunks.append({"text": header.strip(), "source": doc["path"], "doc_id": doc.get("path", "")})
        return chunks

    # Split body into paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    current = []
    current_len = 0
    max_body_chars = max_chunk_chars - len(header)

    for para in paragraphs:
        if current_len + len(para) + 2 <= max_body_chars:
            current.append(para)
            current_len += len(para) + 2
        else:
            if current:
                chunk_body = "\n\n".join(current)
                chunks.append({
                    "text": header + chunk_body,
                    "source": doc["path"],
                    "doc_id": doc.get("path", ""),
                })
            # If single paragraph is too long, split by sentences with overlap
            if len(para) > max_body_chars:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current = []
                current_len = 0
                for sent in sentences:
                    if current_len + len(sent) + 1 <= max_body_chars:
                        current.append(sent)
                        current_len += len(sent) + 1
                    else:
                        if current:
                            chunk_body = " ".join(current)
                            chunks.append({
                                "text": header + chunk_body,
                                "source": doc["path"],
                                "doc_id": doc.get("path", ""),
                            })
                        # Overlap: keep last few words/sentences
                        overlap_tokens = overlap_chars // 5
                        current = sent.split()[-overlap_tokens:] if overlap_tokens else []
                        current = [" ".join(current)] if current else [sent]
                        current_len = len(current[0])
                if current:
                    chunk_body = " ".join(current)
                    chunks.append({
                        "text": header + chunk_body,
                        "source": doc["path"],
                        "doc_id": doc.get("path", ""),
                    })
                current = []
                current_len = 0
                continue
            current = [para]
            current_len = len(para) + 2

    if current:
        chunk_body = "\n\n".join(current)
        chunks.append({
            "text": header + chunk_body,
            "source": doc["path"],
            "doc_id": doc.get("path", ""),
        })
    return chunks


def chunk_documents(docs: List[dict], **kwargs) -> List[dict]:
    """Chunk all documents into a flat list of chunks."""
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, **kwargs))
    return all_chunks


# --- Embedding ---
def embed_chunks(chunks: List[dict], model_name: str = "all-MiniLM-L6-v2") -> Tuple[List[dict], np.ndarray]:
    """Generate embeddings for all chunks. Returns chunks and embedding matrix."""
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return chunks, np.array(embeddings, dtype=np.float32)


# --- Retrieval ---
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vectors a (1D or 2D) and b (1D)."""
    if a.ndim == 1:
        a = a.reshape(1, -1)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return np.dot(a_norm, b_norm).flatten()


def retrieve(query_embedding: np.ndarray, chunk_embeddings: np.ndarray, chunks: List[dict], top_k: int = 5) -> List[dict]:
    """Return top-k chunks by cosine similarity."""
    sims = cosine_similarity(chunk_embeddings, query_embedding)
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [{"chunk": chunks[i], "score": float(sims[i])} for i in top_indices]


# --- Generation ---
def build_prompt(query: str, retrieved: List[dict], max_context_chars: int = 2000) -> str:
    """Build a prompt with retrieved context for the LLM."""
    context_parts = []
    total = 0
    for r in retrieved:
        chunk_text = r["chunk"]["text"]
        if total + len(chunk_text) + 2 > max_context_chars:
            break
        context_parts.append(chunk_text)
        total += len(chunk_text) + 2
    context = "\n\n---\n\n".join(context_parts)
    return f"""Answer the question based only on the following email excerpts. If the answer is not in the context, say "I don't have that information in the provided emails."

Context:
{context}

Question: {query}

Answer:"""


def generate_answer(prompt: str, model_name: str = "google/flan-t5-small", max_new_tokens: int = 150) -> str:
    """Generate answer using a HuggingFace seq2seq model (e.g. FLAN-T5)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# --- RAG pipeline ---
class RAGPipeline:
    """End-to-end RAG: load -> chunk -> embed -> retrieve -> generate."""

    def __init__(
        self,
        emails_dir: str = "emails",
        embedding_model: str = "all-MiniLM-L6-v2",
        gen_model: str = "google/flan-t5-small",
        top_k: int = 5,
    ):
        self.emails_dir = emails_dir
        self.embedding_model_name = embedding_model
        self.gen_model_name = gen_model
        self.top_k = top_k
        self.docs = []
        self.chunks = []
        self.embeddings = None
        self.embedding_model = None
        self._gen_model = None
        self._gen_tokenizer = None

    def index(self) -> None:
        """Load documents, chunk, and embed."""
        self.docs = load_emails(self.emails_dir)
        self.chunks = chunk_documents(self.docs)
        print(f"Loaded {len(self.docs)} emails, {len(self.chunks)} chunks.")
        self.chunks, self.embeddings = embed_chunks(self.chunks, self.embedding_model_name)
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        print("Indexing complete.")

    def _get_gen_model(self):
        """Lazy-load the seq2seq model and tokenizer (avoids pipeline task name)."""
        if self._gen_model is None:
            self._gen_tokenizer = AutoTokenizer.from_pretrained(self.gen_model_name)
            self._gen_model = AutoModelForSeq2SeqLM.from_pretrained(self.gen_model_name)
        return self._gen_tokenizer, self._gen_model

    def query(self, question: str) -> Tuple[str, List[dict]]:
        """Retrieve relevant chunks and generate an answer."""
        if self.embeddings is None or self.embedding_model is None:
            raise RuntimeError("Call index() first.")
        q_emb = self.embedding_model.encode([question], show_progress_bar=False)
        q_emb = np.array(q_emb[0], dtype=np.float32)
        retrieved = retrieve(q_emb, self.embeddings, self.chunks, top_k=self.top_k)
        prompt = build_prompt(question, retrieved)
        tokenizer, model = self._get_gen_model()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return answer, retrieved


def main():
    """CLI: index emails and run interactive Q&A."""
    import sys
    rag = RAGPipeline(emails_dir="emails", top_k=5)
    print("Indexing emails (first run may download models)...")
    rag.index()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer, refs = rag.query(question)
        print("\nQuestion:", question)
        print("Answer:", answer)
        print("\nTop retrieved chunks (scores):")
        for r in refs[:3]:
            print(f"  {r['score']:.3f}: {r['chunk']['text'][:120]}...")
        return

    print("\nEnter questions (blank to exit). Example: Who sent emails about training?")
    while True:
        try:
            q = input("\nQ: ").strip()
            if not q:
                break
            answer, refs = rag.query(q)
            print("A:", answer)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
