a=(1.0 0.8 0.6 0.4 0.2 0.0)
temperature=(0.2 0.4 0.6 0.8 1 2 4 6 8 10)
n_client=9
n_train_data=1000
n_public_data=100
n_test_data=200
seed=0
local_epochs=5
distill_epochs=5
server_epochs=100
batch_size=160
alpha=0.1
dataset='mnist'
model_structure='resnet18'
device=1

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
