a=(0.5)         # (1.0 0.5)
temperature=(6) # (1 5 10)
n_public_data=50
seed=10000
local_epochs=5
distill_epochs=5
server_epochs=20
batch_size=160
alpha=1
dataset='cifar10'
model_structure='cnn3'
device=2

for a_ in ${a[@]}; do
  for temperature_ in ${temperature[@]}; do
    # for model_structure_ in ${model_structure[@]}; do
    python hyper_fed.py \
      --a $a_ \
      --temperature $temperature_ \
      --n_public_data $n_public_data \
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
