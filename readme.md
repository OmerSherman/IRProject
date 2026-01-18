## Historical Archives RAG Project with Gemma-3

This project implements a Retrieval-Augmented Generation (RAG) pipeline optimized for analyzing 19th-century newspaper archives (OCR text).

### System Requirements

- Environment: Google Colab (Recommended).

- Hardware: GPU T4 is mandatory for monoT5 reranking and Gemma-3 inference.

- Key Libraries: pyterrier, unsloth, torch, pandas.

### Pipeline Architecture

The system uses a three-stage cascade to optimize performance:

- Lexical Retrieval: BM25 (via PyTerrier) to narrow down the search space.

- Neural Reranking: monoT5 for semantic alignment on noisy OCR text.

- Generation (QA): Gemma-3-270M-IT (4-bit quantization via Unsloth).

Note: To ensure stability and stay within hardware memory limits, the generation stage uses the Top-1 document after neural reranking.

### How to Run

- File Import: Upload the required CSV files to the Colab environment before running the pipelines.

- Notebook Structure: Use the dedicated cells for:

- Installing dependencies.

- Loading precomputed files (to save time to not rerunning the retrieval process).

- Running the evaluation pipelines.

### Technical Disclaimer

Results may vary compared to global baselines as the evaluation was performed on a specific subset of queries. This subset accounts for 19th-century OCR noise but is limited by the hardware constraints of the free Colab environment. Furthermore, to prevent Out-Of-Memory (OOM) errors, it is recommended to load the provided intermediate results instead of re-running the heavy computation stages (BM25/Reranking).