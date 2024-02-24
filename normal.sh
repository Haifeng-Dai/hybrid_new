n_public_data=1000
seed=10000
local_epochs=600
batch_size=160
alpha=1
dataset='cifar10'
model_structure=('resnet18')
device=1

for model_structure_ in ${model_structure[@]}; do
  python normal.py \
    --n_public_data $n_public_data \
    --seed $seed \
    --local_epochs $local_epochs \
    --batch_size $batch_size \
    --alpha $alpha \
    --dataset $dataset \
    --model_structure $model_structure_ \
    --device $device
done
