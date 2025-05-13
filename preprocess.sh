RELEVANT_CODE_NUM=5

# python preprocess.py \
#     --dataset_path $TRAIN_FILE_PATH \
#     --dataset_name train \
#     --tokenizer_path $TOKENIZED_PATH \
#     --relevant_code_num 1 \
#     --workers 16

python preprocess.py \
    --dataset_path data/repoeval/line_level/test_0.parquet,data/repoeval/line_level/test_1.parquet \
    --dataset_name repoeval_line \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --step 8 \
    --output_dir preprocessed/step8/ \
&& \
python preprocess.py \
    --dataset_path data/repoeval/api_level/test_0.parquet,data/repoeval/api_level/test_1.parquet \
    --dataset_name repoeval_api \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --step 8 \
    --output_dir preprocessed/step8/ \
&& \
python preprocess.py \
    --dataset_path data/cceval/python/test.parquet \
    --dataset_name cceval \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --step 8 \
    --output_dir preprocessed/step8/ \
&& \
python preprocess.py \
    --dataset_path data/ours/python/test_suffix.parquet \
    --dataset_name ours-suffix \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --step 8 \
    --output_dir preprocessed/step8/ \
&& \
python preprocess.py \
    --dataset_path data/ours/python/test.parquet \
    --dataset_name ours \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --step 8 \
    --output_dir preprocessed/step8/

python preprocess.py \
    --dataset_path data/repoeval/line_level/test_0_only_prefix.parquet,data/repoeval/line_level/test_1_only_prefix.parquet \
    --dataset_name repoeval_line_only_prefix \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --only_prefix \
    --workers 16 \
    --step 8 \
    --output_dir preprocessed/step8/ \
&& \
python preprocess.py \
    --dataset_path data/repoeval/api_level/test_0_only_prefix.parquet,data/repoeval/api_level/test_1_only_prefix.parquet \
    --dataset_name repoeval_api_only_prefix \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --only_prefix \
    --workers 16 \
    --step 8 \
    --output_dir preprocessed/step8/ \
&& \
python preprocess.py \
    --dataset_path data/cceval/python/test_only_prefix.parquet \
    --dataset_name cceval_only_prefix \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --only_prefix \
    --workers 16 \
    --step 8 \
    --output_dir preprocessed/step8/ \
&& \
python preprocess.py \
    --dataset_path data/ours/python/test_only_prefix.parquet \
    --dataset_name ours_only_prefix \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --only_prefix \
    --workers 16 \
    --step 8 \
    --output_dir preprocessed/step8/ \
&& \
python preprocess.py \
    --dataset_path data/ours/python/test_suffix_only_prefix.parquet \
    --dataset_name ours-suffix_only_prefix \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --only_prefix \
    --workers 16 \
    --step 8 \
    --output_dir preprocessed/step8/

python small_relevant.py \
    --dataset_path data/repoeval/line_level/test_0.parquet,data/repoeval/line_level/test_1.parquet \
    --dataset_name repoeval_line \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --generated_file outputs/retrieval/opc-sft-v1/result_0/repoeval_line/prediction_truncated.jsonl \
    --output_dir preprocessed_retrieval_twice/step7-more \
&& \
python small_relevant.py \
    --dataset_path data/repoeval/api_level/test_0.parquet,data/repoeval/api_level/test_1.parquet \
    --dataset_name repoeval_api \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --generated_file outputs/retrieval/opc-sft-v1/result_0/repoeval_api/prediction_truncated.jsonl \
    --output_dir preprocessed_retrieval_twice/step7-more \
&& \
python small_relevant.py \
    --dataset_path data/cceval/python/test.parquet \
    --dataset_name cceval \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --generated_file outputs/retrieval/opc-sft-v1/result_0/cceval_python/prediction_truncated.jsonl \
    --output_dir preprocessed_retrieval_twice/step7-more \
&& \
python small_relevant.py \
    --dataset_path data/ours/python/test.parquet \
    --dataset_name ours \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --generated_file outputs/retrieval/opc-sft-v1/result_0/ours/prediction_truncated.jsonl \
    --output_dir preprocessed_retrieval_twice/step7-more \
&& \
python small_relevant.py \
    --dataset_path data/ours/python/test_suffix.parquet \
    --dataset_name ours-suffix \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --generated_file outputs/retrieval/opc-sft-v1/result_0/ours_suffix/prediction_truncated.jsonl \
    --output_dir preprocessed_retrieval_twice/step7-more \
&& \
python small_relevant.py \
    --dataset_path data/repoeval/line_level/test_0_only_prefix.parquet,data/repoeval/line_level/test_1_only_prefix.parquet \
    --dataset_name repoeval_line_only_prefix \
    --only_prefix \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --generated_file outputs/only-prefix/with-retrieval/opc-sft-v1/result_0/repoeval_line_only_prefix/prediction_truncated.jsonl \
    --output_dir preprocessed_retrieval_twice/step7-more \
&& \
python small_relevant.py \
    --dataset_path data/repoeval/api_level/test_0_only_prefix.parquet,data/repoeval/api_level/test_1_only_prefix.parquet \
    --dataset_name repoeval_api_only_prefix \
    --only_prefix \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --generated_file outputs/only-prefix/with-retrieval/opc-sft-v1/result_0/repoeval_api_only_prefix/prediction_truncated.jsonl \
    --output_dir preprocessed_retrieval_twice/step7-more \
&& \
python small_relevant.py \
    --dataset_path data/cceval/python/test_only_prefix.parquet \
    --dataset_name cceval_only_prefix \
    --only_prefix \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --generated_file outputs/only-prefix/with-retrieval/opc-sft-v1/result_0/cceval_python_only_prefix/prediction_truncated.jsonl \
    --output_dir preprocessed_retrieval_twice/step7-more \
&& \
python small_relevant.py \
    --dataset_path data/ours/python/test_only_prefix.parquet \
    --dataset_name ours_only_prefix \
    --only_prefix \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --generated_file outputs/only-prefix/with-retrieval/opc-sft-v1/result_0/ours_only_prefix/prediction_truncated.jsonl \
    --output_dir preprocessed_retrieval_twice/step7-more \
&& \
python small_relevant.py \
    --dataset_path data/ours/python/test_suffix_only_prefix.parquet \
    --dataset_name ours_suffix_only_prefix \
    --only_prefix \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --generated_file outputs/only-prefix/with-retrieval/opc-sft-v1/result_0/ours_suffix_only_prefix/prediction_truncated.jsonl \
    --output_dir preprocessed_retrieval_twice/step7-more

python small_relevant.py \
    --dataset_path data/repoeval/line_level/test_0.parquet,data/repoeval/line_level/test_1.parquet \
    --dataset_name repoeval_line_only_prefix \
    --only_prefix \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --generated_file outputs/only-prefix/with-retrieval/opc-sft-v1/result_0/repoeval_line_only_prefix/prediction_truncated.jsonl \
    --output_dir preprocessed_retrieval_twice/step7-more \
&& \
python small_relevant.py \
    --dataset_path data/repoeval/api_level/test_0.parquet,data/repoeval/api_level/test_1.parquet \
    --dataset_name repoeval_api_only_prefix \
    --only_prefix \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --generated_file outputs/only-prefix/with-retrieval/opc-sft-v1/result_0/repoeval_api_only_prefix/prediction_truncated.jsonl \
    --output_dir preprocessed_retrieval_twice/step7-more \
&& \
python small_relevant.py \
    --dataset_path data/cceval/python/test.parquet \
    --dataset_name cceval_only_prefix \
    --only_prefix \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --generated_file outputs/only-prefix/with-retrieval/opc-sft-v1/result_0/cceval_python_only_prefix/prediction_truncated.jsonl \
    --output_dir preprocessed_retrieval_twice/step7-more \
&& \
python small_relevant.py \
    --dataset_path data/ours/python/test.parquet \
    --dataset_name ours_only_prefix \
    --only_prefix \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --generated_file outputs/only-prefix/with-retrieval/opc-sft-v1/result_0/ours_only_prefix/prediction_truncated.jsonl \
    --output_dir preprocessed_retrieval_twice/step7-more \
&& \
python small_relevant.py \
    --dataset_path data/ours/python/test_suffix.parquet \
    --dataset_name ours_suffix_only_prefix \
    --only_prefix \
    --relevant_code_num $RELEVANT_CODE_NUM \
    --workers 16 \
    --generated_file outputs/only-prefix/with-retrieval/opc-sft-v1/result_0/ours_suffix_only_prefix/prediction_truncated.jsonl \
    --output_dir preprocessed_retrieval_twice/step7-more