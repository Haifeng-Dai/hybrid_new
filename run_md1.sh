a=(1)           # (1.0 0.5)
temperature=(1) # (1 5 10)
n_client=9
n_train_data=1000
n_public_data=100
n_test_data=200
seed=10000
local_epochs=5
distill_epochs=5
server_epochs=20
batch_size=160
alpha=1
dataset='cifar10'
model_structure='cnn2'
device=0

for a_ in ${a[@]}; do
  for temperature_ in ${temperature[@]}; do
    # for model_structure_ in ${model_structure[@]}; do
    python md_test.py \
      --a $a_ \
      --temperature $temperature_ \
      --n_client $n_client \
      --n_train_data $n_train_data \
      --n_public_data $n_public_data \
      --n_test_data $n_test_data \
      --seed $seed \
      --local_epochs $local_epochs \
      --distill_epochs $distill_epochs \
      --server_epochs $server_epochs \
      --batch_size $batch_size \
      --alpha $alpha \
      --dataset $dataset \
      --model_structure $model_structure \
      --device $device
    # done
  done
done
