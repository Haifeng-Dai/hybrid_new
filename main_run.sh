echo start run

# for i in run_avg run_md run_md1 run_md2 run_md3 run_md4; do
for i in test; do
    echo $i.sh
    bash $i.sh
done
