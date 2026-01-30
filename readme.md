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

There are two ways to execute this notebook, depending on whether you want to perform the full computation or simply reproduce the evaluation results using pre-computed data.

1. Prerequisites

Environment: Open the notebook in Google Colab.

File Import: Upload the required CSV files (pre-computed candidates) to the local Colab storage before execution.

Setup: Run the initial cells to install dependencies and initialize the PyTerrier environment.

2. Execution Modes

- **Option A**: Full Pipeline (Time-Consuming)

Use this mode if you want to re-run the entire retrieval and re-ranking process from scratch.

Action: Execute all cells in the notebook. The code will generate new result CSV files before proceeding to the evaluation.

Note: This process is lengthy as it involves model inference across the entire dataset.

- **Option B**: Fast Evaluation (Runnable after Option A)

Use this mode to obtain results instantly without re-running the models. A dedicated cell performs these three key operations:

Loading: Importing DataFrames from the CSV files (previously uploaded from this GitHub data/ folder).

Transformation: Converting DataFrames into PyTerrier "Transformers" (via pt.Transformer.from_df) to make them compatible with the experiment module.

Evaluation: Executing pt.Experiment to calculate metrics (P@k, MAP, nDCG) against the ground truth (qrels).
```bash
# 1. Load pre-computed results from CSV files
example_df_1 = pd.read_csv("example_candidates_1.csv")
example_df_2 = pd.read_csv("example_candidates_2.csv")

# 2. Create PyTerrier Transformers from DataFrames
# This allows static data to be treated as retrieval pipelines
example_pipeline_1 = pt.Transformer.from_df(example_df_1)
example_pipeline_2 = pt.Transformer.from_df(example_df_2)

# 3. Prepare queries and ground truth (qrels)
queries_df = pd.DataFrame(queries).rename(columns={"query_id": "qid", "question": "query"})
qrels_df = pd.DataFrame(qrels).rename(columns={"query_id": "qid", "para_id": "docno"})

# 4. Run the Experiment
# Compares all models using standard metrics
experiment_results = pt.Experiment(
    [example_pipeline_1, example_pipeline_2],
    queries_df,
    qrels_df,
    [P@1, P@5, P@10, R@5, R@10, nDCG@5, nDCG@10, MAP],
    names=["Example_Model_Name_1", "Example_Model_Name_2"]
)

display(experiment_results)
```
### Technical Disclaimer

Results may vary compared to global baselines as the evaluation was performed on a specific subset of queries. This subset accounts for 19th-century OCR noise but is limited by the hardware constraints of the free Colab environment. Furthermore, to prevent Out-Of-Memory (OOM) errors, it is recommended to load the provided intermediate results instead of re-running the heavy computation stages (BM25/Reranking).
