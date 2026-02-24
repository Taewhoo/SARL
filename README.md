# SARL

Requirements: CUDA 12.8 and `uv`.

## 1. Build the Wiki Index

Download the corpus and webpage files from HuggingFace, then build the retrieval index:

```bash
# Download corpus files
wget https://huggingface.co/datasets/inclusionAI/ASearcher-Local-Knowledge/resolve/main/wiki_corpus.jsonl
wget https://huggingface.co/datasets/inclusionAI/ASearcher-Local-Knowledge/resolve/main/wiki_webpages.jsonl

# Build the e5 index
bash wiki_server/build_index.sh
```

Edit `wiki_server/build_index.sh` to set `save_dir` and `corpus_file` to the appropriate paths before running. This will produce an `e5.index` directory.

## 2. Launch the Wiki Server

Start the local retrieval server:

```bash
bash wiki_server/launch_local_server.sh <port> <address_save_path>
```

Edit `wiki_server/launch_local_server.sh` to point `WIKI2018_WORK_DIR` to the directory containing your downloaded corpus and built index.

## 3. Training

Run training scripts from the repo root. For example:

```bash
cd train
bash scripts/run_search_self_critique_v4.sh
```

Before running, set the following variables in the script:
- `DATA_DIR`: path to the training data (`.parquet` files)
- `WANDB_API_KEY`: your Weights & Biases API key
- `trainer.ckpt_path` / `trainer.export_path`: paths for saving checkpoints and exported models
- `environment.skyrl_gym.search.search_url`: URL of the running wiki server
