#!/bin/bash

# Download wiki_corpus.jsonl and wiki_webpages.jsonl using:
# wget https://huggingface.co/datasets/inclusionAI/ASearcher-Local-Knowledge/resolve/main/wiki_corpus.jsonl
# wget https://huggingface.co/datasets/inclusionAI/ASearcher-Local-Knowledge/resolve/main/wiki_webpages.jsonl

# === wiki18 (old) ===
# save_dir="path_to_wiki_retriever_data"
# corpus_file="/mnt/weka/post_training/pt2-search-agent/SARL/wiki_corpus.jsonl"
# save_dir="/mnt/weka/post_training_tmp/pt2-search-agent/retriever/wiki18/e5.index"
# retriever_name=e5 # this is for indexing naming
# retriever_model=intfloat/e5-base-v2
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 index_builder.py \
#     --retrieval_method $retriever_name \
#     --model_path $retriever_model \
#     --corpus_path $corpus_file \
#     --save_dir $save_dir \
#     --use_fp16 \
#     --max_length 256 \
#     --batch_size 2048 \
#     --pooling_method mean \
#     --faiss_type Flat \
#     --save_embedding

# === wiki26 (DPR-chunked) ===
corpus_file="/mnt/weka/post_training_tmp/pt2-search-agent/retriever/wiki26/wiki_corpus.jsonl"
save_dir="/mnt/weka/post_training_tmp/pt2-search-agent/retriever/wiki26/e5.index"
retriever_name=e5
retriever_model=intfloat/e5-base-v2

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 2048 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding
