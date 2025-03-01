
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
    --output_dir outputs/unixcoder \
    --train_filename github_projects \
    --do_train \
    --do_valid \
    --do_test \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --GPU_ids 2 \
    --max_input_length 2048 \
    --max_crossfile_length 1536 
```

* 顺序对性能有影响吗？如果有，是怎么样的影响？
    * 使用bm25按相似度从小到大、从大到小以及完全随机都测试一遍就能知道结果了
* 如果一个样例没有相应的related file该如何处理
    * 