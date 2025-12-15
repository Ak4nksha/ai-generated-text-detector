# ai-generated-text-detector

An authorship detection pipeline for distinguishing **human-written text** from **LLM-generated content**, using a progression of classical ML, sequence models, and transformer-based approaches.

The project emphasizes **robust data construction**, **fixed evaluation splits**, and **comparative modeling** to analyze generalization behavior across different model families.

---

## Project Overview

This repository implements and evaluates multiple approaches for AI-vs-human text detection:

- Stylometric + classical ML baselines
- Sequence modeling with LSTM
- Linear probing on pretrained sentence encoders
- Full transformer fine-tuning

The focus is not just performance, but **understanding why models behave differently**, especially under distribution shift.

---

## Repository Structure

```text
.
├── notebooks/
│   ├── 01_data_ingestion_and_profiling.ipynb
│   ├── 02_data_integration.ipynb
│   ├── 03_feature_engineering_and_baseline_models.ipynb
│   ├── 04_lstm_sequence_model.ipynb
│   ├── 05_linear_probe_pretrained_encoder.ipynb
│   └── 06_transformer_finetune.ipynb
│
├── data/
│   ├── conversations.json
│   ├── meta.json
│   ├── train_all.parquet
│   ├── val_all.parquet
│   ├── test_all.parquet
│   └── cache/
│       ├── X_train.npy
│       ├── X_val.npy
│       └── X_test.npy
│
├── reports/
│   ├── 01_data_ingestion_and_profiling.pdf
|   ├── 02_data_integration.pdf
│   ├── 03_feature_engineering_and_baseline_models.pdf
│   ├── 04_lstm_sequence_model.pdf
│   ├── 05_linear_probe_pretrained_encoder.pdf
│   └── 06_transformer_finetune.pdf
│
└── README.md

```

## Data Access

### Public datasets

The project uses a combination of:

- Kaggle AI-vs-Human text datasets  
- Public human-written sources (Wikipedia, news articles)  
- Consented and anonymized LLM conversation data  

**Important:**

- Large Kaggle datasets are **not included** in this repository due to size and licensing restrictions.  
- Users must download these datasets manually using the **Kaggle API**.



## Two Ways to Run This Project

### Option A — Full Pipeline  
*(Recommended for understanding the methodology)*

Run the notebooks **in order**:

1. `01_data_ingestion_and_profiling.ipynb`
2. `02_data_integration.ipynb`
3. `03_feature_engineering_and_baseline_models.ipynb`
4. `04_lstm_sequence_model.ipynb`
5. `05_linear_probe_pretrained_encoder.ipynb`
6. `06_transformer_finetune.ipynb`

This path shows:

- How raw data was cleaned, filtered, and combined  
- How fixed train / validation / test splits were created  
- Why specific modeling and evaluation choices were made  

---

### Option B — Direct Model Execution  
*(Skip data preparation notebooks)*

If you want to **directly run the models without redoing data preparation**:

- Use the files already provided in the `data/` directory  
- Start from **Notebook 03 onward**

Notebooks **03–06** will run using:

- `train_all.parquet`
- `val_all.parquet`
- `test_all.parquet`
- `meta.json`



## Cached Embeddings

The `data/cache/` directory contains cached sentence-transformer embeddings used in:

- `05_linear_probe_pretrained_encoder.ipynb`

These cached files are provided to:

- Avoid recomputing embeddings
- Enable reproducible results
- Reduce GPU usage

If the cache exists, embeddings are loaded automatically.

---

## Hardware Requirements

GPU is required for:

- LSTM training
- Transformer fine-tuning

Recommended environments:

- Google Colab (GPU runtime)
- University JupyterHub with GPU support

CPU-only execution is **not recommended** for Notebooks 04–06.

---

## Reports (No-Run Option)

The `reports/` directory contains PDF exports of all modeling notebooks.

These reports allow readers to:

- Review code structure and outputs
- Inspect figures, tables, and results
- Understand model behavior without running the notebooks

## Modeling Summary

| Model                              | Validation F1 | Test F1 | Key Insight |
|-----------------------------------|---------------|---------|-------------|
| Stylometric + Logistic Regression | Moderate      | Drops   | Captures surface-level stylistic patterns |
| LSTM Sequence Model               | High          | Drops   | Overfits the training distribution |
| Linear Probe (Sentence Encoder)   | Strong        | Stable  | Pretrained representations generalize better |
| Fine-tuned Transformer            | Very High     | Mixed   | Powerful but sensitive to distribution shift |

---

## Key Takeaways

- High validation accuracy does **not** guarantee generalization to unseen data  
- Pretrained representations are more robust than task-trained sequence models  
- Transformer fine-tuning can overfit subtle stylistic cues  
- Qualitative analysis shows human-written text is harder to model consistently than AI-generated text  

---

## Limitations & Future Work

- Incorporate broader human text sources for improved generalization  
- Perform adversarial testing against newer LLMs  
- Add calibration and uncertainty estimation for prediction confidence  
- Explore hybrid models combining stylometry with semantic embeddings  


