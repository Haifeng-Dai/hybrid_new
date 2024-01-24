n_client=9
n_train_data=2000
n_test_data=200
seed=10000
local_epochs=20
batch_size=160
server_epochs=5
alpha=1
dataset='cifar10'
model_structure=('cnn2')
device=2

for model_structure_ in ${model_structure[@]}; do
  python test1.py \
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
