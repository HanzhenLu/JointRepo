TRAIN_FILE_PATH="data/github_projects/python/sampled-huawei-train_4_18.json"
TRAIN_BM25_REtRIEVAL_CODE_NUM=25
TEST_BM25_RETRIEVAL_CODE_NUM=25

python preprocess.py \
    --dataset_path $TRAIN_FILE_PATH \
    --dataset_name train \
    --relevant_code_num $TRAIN_BM25_REtRIEVAL_CODE_NUM \
    --workers 16

# python preprocess.py \
#     --dataset_path data/repoeval/line_level/test_0.parquet,data/repoeval/line_level/test_1.parquet \
#     --dataset_name repoeval \
#     --relevant_code_num $TEST_BM25_RETRIEVAL_CODE_NUM \
#     --workers 16

# python preprocess.py \
#     --dataset_path data/cceval/python/test.parquet \
#     --dataset_name cceval \
#     --relevant_code_num $TEST_BM25_RETRIEVAL_CODE_NUM \
#     --workers 16

# python preprocess.py \
#     --dataset_path data/ours/python/test.parquet \
#     --dataset_name ours \
#     --relevant_code_num $TEST_BM25_RETRIEVAL_CODE_NUM \
#     --workers 16

# python preprocess.py \
#     --dataset_path data/ours/python/test_suffix.parquet \
#     --dataset_name ours-suffix \
#     --relevant_code_num $TEST_BM25_RETRIEVAL_CODE_NUM \
#     --workers 16