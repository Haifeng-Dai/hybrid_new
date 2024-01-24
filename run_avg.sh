n_client=9
n_train_data=1000
n_test_data=200
seed=10000
local_epochs=1
batch_size=10
server_epochs=100
alpha=1
dataset='cifar10'
model_structure=('cnn3')
device=2

for model_structure_ in ${model_structure[@]}; do
  python fedavg.py \
    --n_client $n_client \
    --n_train_data $n_train_data \
    --n_test_data $n_test_data \
    --seed $seed \
    --local_epochs $local_epochs \
    --batch_size $batch_size \
    --server_epochs $server_epochs \
    --alpha $alpha \
    --dataset $dataset \
    --model_structure $model_structure_ \
    --device $device
done
