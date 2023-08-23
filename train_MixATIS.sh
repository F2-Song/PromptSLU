 dataset=MixATIS_clean
map=v1
pretrain_model=t5-base
seed=0 # in {0, 1, 2, 3}
batch=32 # real batch_size

nohup accelerate launch --config_file two_gpu.yaml main.py \
    --dataset_name $dataset \
    --mode train \
    --map $map \
    --dropout_rate 0.1 \
    --train_file train.json \
    --validation_file dev.json \
    --model_name_or_path $pretrain_model \
    --num_train_epochs 10 \
    --output_dir results/$dataset/$pretrain_model/$map/$seed \
    --per_device_eval_batch_size 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 1e-4 \
    --seed $seed \
    > log/train_${dataset}_${batch}_${pretrain_model}_${map}_${seed}.log 2>&1 &