a=(1.0)   # (1.0 0.5)
temperature1=5 # (1 5 10)
temperature2=10
temperature3=15
n_public_data=100
seed=10000
local_epochs=1
distill_epochs=1
server_epochs=2
batch_size=160
alpha=1
dataset='cifar10'
model_structure='cnn3'
device=2

for a_ in ${a[@]}; do
  # for temperature_ in ${temperature[@]}; do
  # for model_structure_ in ${model_structure[@]}; do
  mpiexec -np 3 python hyper_fed1.py \
    --a $a_ \
    --temperature $temperature1 $temperature2 $temperature3 \
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
  # done
done
