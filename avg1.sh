n_public_data=100
seed=10000
local_epochs=5
batch_size=160
server_epochs=10
alpha=1
dataset='cifar10'
model_structure=('cnn3')
device=0

for model_structure_ in ${model_structure[@]}; do
  python fedavg1.py \
    --n_public_data $n_public_data \
    --seed $seed \
    --local_epochs $local_epochs \
    --batch_size $batch_size \
    --server_epochs $server_epochs \
    --alpha $alpha \
    --dataset $dataset \
    --model_structure $model_structure_ \
    --device $device
done
