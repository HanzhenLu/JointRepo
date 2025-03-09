
```bash
python run.py \
    --generator_name_or_path /data/hanzhenlu/LLaMA-Factory/saves/opc-sft-v1 \
    --retriever_name_or_path /data/hanzhenlu/model/snowflake-arctic-embed-xs \
    --output_dir outputs/ \
    --train_filename github_projects \
    --do_train \
    --do_valid \
    --do_test \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --GPU_ids 2 \
    --max_input_length 2048 \
    --max_crossfile_length 1536 

python run.py \
    --generator_name_or_path /data/hanzhenlu/LLaMA-Factory/saves/opc-sft-v1 \
    --retriever_name_or_path /data/hanzhenlu/model/microsoft/unixcoder-base \
    --output_dir outputs/unixcoder-debug \
    --train_filename github_projects \
    --do_train \
    --do_valid \
    --do_test \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --GPU_ids 2 \
    --max_input_length 2048 \
    --max_crossfile_length 1536 \
    --debug

python run.py \
    --generator_name_or_path /data/hanzhenlu/LLaMA-Factory/saves/opc-sft-v1 \
    --retriever_name_or_path /data/hanzhenlu/model/CodeRankEmbed \
    --output_dir outputs/CodeRankEmbed \
    --train_filename github_projects \
    --do_test \
    --eval_batch_size 16 \
    --GPU_ids 6 \
    --max_input_length 2048 \
    --max_crossfile_length 1536

# 用于测试deepseek
python deepseek.py \
    --generator_name_or_path /nasdata/Model/deepseek-coder-1.3b-base \
    --retriever_name_or_path /data/hanzhenlu/model/microsoft/unixcoder-base \
    --retrieval_method unixcoder-base \
    --output_dir outputs/deepseek-unixcoder \
    --eval_batch_size 16 \
    --GPU_ids 5 \
    --max_input_length 2048 \
    --max_crossfile_length 1536
```