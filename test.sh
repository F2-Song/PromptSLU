dataset=MixATIS_clean # or MixSNIPS_clean
map=v1
pretrain_model=t5-base
seed=0 # in {0, 1, 2, 3}, used for just checkpoint picking
file=test
batch=32

nohup python -u main.py \
    --dataset_name $dataset \
    --mode test \
    --map $map \
    --validation_file ${file}.json \
    --model_name_or_path $pretrain_model \
    --output_dir results/$dataset/$pretrain_model/$map/$seed/best_checkpoint \
    --per_device_eval_batch_size 128 \
    --seed $seed \
    > log/${file}_${dataset}_${batch}_${pretrain_model}_${map}_${seed}.log 2>&1 &