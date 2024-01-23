n_client=9
n_train_data=1100
n_public_data=0
n_test_data=200
seed=10
local_epochs=5
distill_epochs=5
batch_size=160
server_epochs=100
alpha=0.1
dataset='cifar100'
model_structure=('resnet18')
device=2

for model_structure_ in ${model_structure[@]}; do
  python avg_test.py \
    --n_client $n_client \
    --n_train_data $n_train_data \
    --n_public_data $n_public_data \
    --n_test_data $n_test_data \
    --seed $seed \
    --local_epochs $local_epochs \
    --distill_epochs $distill_epochs \
    --batch_size $batch_size \
    --server_epochs $server_epochs \
    --alpha $alpha \
    --dataset $dataset \
    --model_structure $model_structure_ \
    --device $device
done
