TRAIN_FILE_PATH="data/github_projects/python/sampled-huawei-train_4_18.json"
TRAIN_BM25_REtRIEVAL_CODE_NUM=25
TEST_BM25_RETRIEVAL_CODE_NUM=25

# python preprocess.py \
#     --dataset_path $TRAIN_FILE_PATH \
#     --dataset_name train \
#     --relevant_code_num $TRAIN_BM25_REtRIEVAL_CODE_NUM \
#     --workers 16

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

python preprocess.py \
    --dataset_path data/repoeval/line_level/test_0_only_prefix.parquet,data/repoeval/line_level/test_1_only_prefix.parquet \
    --dataset_name repoeval_only_prefix \
    --relevant_code_num $TEST_BM25_RETRIEVAL_CODE_NUM \
    --workers 16

python preprocess.py \
    --dataset_path data/cceval/python/test_only_prefix.parquet \
    --dataset_name cceval_only_prefix \
    --relevant_code_num $TEST_BM25_RETRIEVAL_CODE_NUM \
    --workers 16

python preprocess.py \
    --dataset_path data/ours/python/test_only_prefix.parquet \
    --dataset_name ours_only_prefix \
    --relevant_code_num $TEST_BM25_RETRIEVAL_CODE_NUM \
    --workers 16

python preprocess.py \
    --dataset_path data/ours/python/test_suffix_only_prefix.parquet \
    --dataset_name ours-suffix_only_prefix \
    --relevant_code_num $TEST_BM25_RETRIEVAL_CODE_NUM \
    --workers 16

# python small_relevant.py \
#     --dataset_path data/repoeval/line_level/test_0_only_prefix.parquet,data/repoeval/line_level/test_1_only_prefix.parquet \
#     --dataset_name repoeval_only_prefix \
#     --relevant_code_num $TEST_BM25_RETRIEVAL_CODE_NUM \
#     --generated_file /data/hanzhenlu/temp/JointRepo-bm25-variant/outputs/opc-sft-v1-modified-only_prefix/result_0/repoeval_line_only_prefix/prediction_truncated.jsonl \
#     --workers 16

# python small_relevant.py \
#     --dataset_path data/cceval/python/test_only_prefix.parquet \
#     --dataset_name cceval_only_prefix \
#     --relevant_code_num $TEST_BM25_RETRIEVAL_CODE_NUM \
#     --generated_file /data/hanzhenlu/temp/JointRepo-bm25-variant/outputs/opc-sft-v1-modified-only_prefix/result_0/cceval_python_only_prefix/prediction_truncated.jsonl \
#     --workers 16

# python small_relevant.py \
#     --dataset_path data/ours/python/test_only_prefix.parquet \
#     --dataset_name ours_only_prefix \
#     --relevant_code_num $TEST_BM25_RETRIEVAL_CODE_NUM \
#     --generated_file /data/hanzhenlu/temp/JointRepo-bm25-variant/outputs/opc-sft-v1-modified-only_prefix/result_0/ours_only_prefix/prediction_truncated.jsonl \
#     --workers 16

# python small_relevant.py \
#     --dataset_path data/ours/python/test_suffix_only_prefix.parquet \
#     --dataset_name ours-suffix_only_prefix \
#     --relevant_code_num $TEST_BM25_RETRIEVAL_CODE_NUM \
#     --generated_file /data/hanzhenlu/temp/JointRepo-bm25-variant/outputs/opc-sft-v1-modified-only_prefix/result_0/ours_suffix_only_prefix/prediction_truncated.jsonl \
#     --workers 16