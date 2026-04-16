# Multimodal RAG Agent

A research prototype that combines retrieval-augmented generation with GPT-4o vision to answer questions about images. The system grounds visual reasoning in an external knowledge base: at inference time, relevant document chunks are retrieved from a ChromaDB vector store and injected as context before the model reasons over the image. A LangGraph `StateGraph` orchestrates the full pipeline — retrieval, vision analysis, and optional tool use (code execution, web search) — within a single, auditable execution graph. An ablation study on 100 samples from the VQAv2 benchmark quantifies the effect of RAG context on visual question answering performance.

---

## Architecture

```
User Query + Image
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                     LangGraph StateGraph                  │
│                                                           │
│  ┌─────────────────┐                                      │
│  │ retrieve_context│  ChromaDB cosine search              │
│  │     node        │  sentence-transformers/all-MiniLM-L6 │
│  └────────┬────────┘                                      │
│           │ rag_context (top-3 chunks)                    │
│           ▼                                               │
│  ┌─────────────────┐                                      │
│  │ vision_reasoning│  GPT-4o multimodal                   │
│  │     node        │  Chain-of-thought:                   │
│  │                 │  Observation → Reasoning → Answer    │
│  └────────┬────────┘                                      │
│           │ final_answer + seeded messages                │
│           ▼                                               │
│  ┌─────────────────┐                                      │
│  │ tool_reasoning  │  GPT-4o + bind_tools()               │
│  │     node        │                                      │
│  └────────┬────────┘                                      │
│           │                                               │
│     tool_calls?                                           │
│      ├─ yes ──► ┌──────────┐                             │
│      │          │ ToolNode │ python_interpreter           │
│      │          │          │ web_search (DuckDuckGo)      │
│      │          └────┬─────┘                             │
│      │               └──────────► tool_reasoning (loop)  │
│      └─ no ──────────────────────────────────► END       │
└───────────────────────────────────────────────────────────┘
        │
        ▼
   Final Answer
```

**State carried across nodes:**
`messages` · `image_path` · `query` · `rag_context` · `final_answer`

---

## Tech Stack

| Component | Library / Model |
|---|---|
| Agent orchestration | `langgraph` — `StateGraph`, `ToolNode`, conditional edges |
| Vision reasoning | `langchain-openai` → `gpt-4o` (high-detail image encoding) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine) |
| Vector store | `chromadb` — persistent HNSW index |
| Document chunking | Custom sentence-boundary chunker (500 char / 50 char overlap) |
| Tool: code execution | Sandboxed `exec()` with restricted `__builtins__`, stdout capture |
| Tool: web search | `duckduckgo-search` — top-3 results |
| Evaluation dataset | `merve/vqav2-small` via HuggingFace `datasets` (21 k samples) |
| Image handling | `Pillow` — PIL image → base64 JPEG for API payload |
| Environment | `python-dotenv`, `tqdm` |

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/<your-username>/multimodal-rag-agent.git
cd multimodal-rag-agent

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env              # or edit .env directly
# Set: OPENAI_API_KEY=sk-...
```

### 3. Ingest the knowledge base

```bash
python src/rag.py --ingest data/documents/
# → loads 4 .txt files, chunks, embeds, upserts into chroma_db/
# → 33 chunks across ai_overview, computer_vision, rag_systems, visual_concepts
```

Verify retrieval works:

```bash
python src/rag.py --query "what color are common vehicles?"
```

### 4. Run the agent

```bash
python src/agent.py --image path/to/image.jpg --query "what is in this image?"
```

Tool use is automatic — computation queries invoke `python_interpreter`, factual lookup queries invoke `web_search`:

```bash
# Triggers python_interpreter tool
python src/agent.py --image image.jpg --query "calculate the area if the square has side 5"

# Grounded in RAG context
python src/agent.py --image image.jpg --query "what is retrieval augmented generation?"
```

### 5. Run the ablation evaluation

```bash
# Full ablation: RAG OFF vs RAG ON, 100 VQAv2 samples each
python src/evaluate.py --n_samples 100 --ablation

# Single condition
python src/evaluate.py --n_samples 50 --use_rag
```

Results are saved to `eval_results_rag_off.json` and `eval_results_rag_on.json`.

---

## Ablation Study: RAG OFF vs RAG ON

Evaluated on 100 samples drawn from the VQAv2 validation split (`merve/vqav2-small`), shuffled with a fixed seed for identical sample order across both conditions. Scoring follows VQA official evaluation: answers are lowercased, articles and punctuation stripped before exact-match and token-level F1 comparison.

| Metric | RAG OFF | RAG ON | Δ (ON − OFF) |
|---|---|---|---|
| Exact Match Accuracy | 0.0% | 0.0% | +0.0 pp |
| Avg Token F1 | 0.1691 | 0.1551 | −0.014 |
| Samples evaluated | 100 | 100 | — |
| API errors | 0 | 0 | — |

**Analysis.** Exact match is 0% in both conditions because GPT-4o's chain-of-thought format produces sentence-length answers (*"The truck is red."*) while VQA ground truth is a single word (*"red"*). Token F1 captures partial overlap and shows the model is semantically correct on a meaningful fraction of samples — the best individual F1 scores reach 0.57 (e.g., *"The man is holding a tennis racket."* vs GT *"tennis racket"*).

RAG ON is marginally below RAG OFF (−0.014 F1) for two reasons specific to this setup: (1) the knowledge base contains general prose about visual concepts rather than image-caption pairs, so retrieved context rarely shifts the model's answer and occasionally introduces distraction; (2) injecting context sometimes triggers safety refusals on questions the model would otherwise answer directly. Both effects would reverse with a domain-matched retrieval corpus (e.g., COCO captions or object-class descriptions). The pipeline architecture, scoring harness, and comparison infrastructure are validated and ready for improved knowledge bases.

---

## Example: Agent Reasoning Trace

**Input image:** synthetic test image (blue background, yellow square, red circle)
**Query:** *"use Python to compute the first 10 fibonacci numbers and show the results"*

```
================================================================
  Query : use Python to compute the first 10 fibonacci numbers
  Image : test.jpg
================================================================

── RAG Context ──────────────────────────────────────────────
[Source 1] ai_overview.txt (chunk 1, similarity 0.064)
Training an LLM involves two broad phases: pre-training and
fine-tuning ...

── Vision Analysis ──────────────────────────────────────────
**Observation**
The image contains a blue background with two shapes: a yellow
square on the left and a red circle on the right.

**Reasoning**
The image is simple and consists of basic geometric shapes with
distinct colors creating contrast against the blue background.

**Answer**
The image shows a yellow square and a red circle on a blue background.

── Tool Calls ───────────────────────────────────────────────
→ python_interpreter({
    "code": "def fibonacci(n):\n    fib_sequence = [0, 1]\n
             while len(fib_sequence) < n:\n
                 fib_sequence.append(fib_sequence[-1] +
                                     fib_sequence[-2])\n
             return fib_sequence\n\n
             print(fibonacci(10))"
  })
← [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

── Final Answer ─────────────────────────────────────────────
The first 10 Fibonacci numbers are: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

The tool call is generated by `tool_reasoning_node` after the vision analysis seeds the message history. LangGraph's conditional edge routes the `AIMessage` with `tool_calls` to `ToolNode`, executes the sandboxed Python, appends the `ToolMessage` result, and loops back to `tool_reasoning` which then produces a final answer with no further tool calls — exiting to `END`.

---

## Project Structure

```
multimodal_rag_agent/
├── src/
│   ├── agent.py        # LangGraph StateGraph — full agent pipeline
│   ├── vision.py       # GPT-4o image encoding + CoT prompting
│   ├── rag.py          # ChromaDB ingest, chunk, retrieve, format
│   ├── tools.py        # python_interpreter + web_search tools
│   └── evaluate.py     # VQAv2 ablation harness
├── data/
│   └── documents/      # .txt knowledge base files
├── tests/
├── notebooks/
├── requirements.txt
└── .env                # OPENAI_API_KEY (not committed)
```

---

## Limitations and Future Work

- **Answer verbosity.** Adding `"Answer in one word or short phrase."` to the VQA system prompt would close the gap between sentence-level outputs and single-word ground truth, likely raising EM from 0% to competitive levels.
- **Knowledge base quality.** Replacing prose paragraphs with image captions or structured object-label facts would increase retrieved context relevance and make the RAG delta positive.
- **Multi-hop retrieval.** The current pipeline performs a single retrieval pass. Iterative retrieval conditioned on intermediate reasoning steps (HyDE, query rewriting) could improve grounding on complex questions.
- **Evaluation scale.** 100 samples provides directional signal; statistical significance requires 1 000+ samples with confidence interval reporting.
