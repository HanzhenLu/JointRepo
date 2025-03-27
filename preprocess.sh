TOKENIZED_PATH="/data/hanzhenlu/LLaMA-Factory/saves/opc-sft-v1"
TRAIN_FILE_PATH="data/github_projects/python/train_03_18.json"
TRAIN_BM25_REtRIEVAL_CODE_NUM=5
TEST_BM25_RETRIEVAL_CODE_NUM=10

python preprocess.py \
    --dataset_path $TRAIN_FILE_PATH \
    --dataset_name train \
    --tokenizer_path $TOKENIZED_PATH \
    --relevant_code_num $TRAIN_BM25_REtRIEVAL_CODE_NUM

python preprocess.py \
    --dataset_path data/repoeval/line_level/test_0.parquet,data/repoeval/line_level/test_1.parquet \
    --dataset_name repoeval \
    --tokenizer_path $TOKENIZED_PATH \
    --relevant_code_num $TEST_BM25_RETRIEVAL_CODE_NUM

python preprocess.py \
    --dataset_path data/cceval/python/test.parquet \
    --dataset_name cceval \
    --tokenizer_path $TOKENIZED_PATH \
    --relevant_code_num $TEST_BM25_RETRIEVAL_CODE_NUM

python preprocess.py \
    --dataset_path data/ours/python/test.parquet \
    --dataset_name ours \
    --tokenizer_path $TOKENIZED_PATH \
    --relevant_code_num $TEST_BM25_RETRIEVAL_CODE_NUM

python preprocess.py \
    --dataset_path data/ours/python/test_suffix.parquet \
    --dataset_name ours-suffix \
    --tokenizer_path $TOKENIZED_PATH \
    --relevant_code_num $TEST_BM25_RETRIEVAL_CODE_NUM