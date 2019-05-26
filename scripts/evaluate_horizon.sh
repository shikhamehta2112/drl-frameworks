source ~/anaconda3/etc/profile.d/conda.sh

echo "--- STARTING HORIZON EXPERIMENTS ---"
conda activate horizon-env
echo
echo "--- STARTING HORIZON CARTPOLE EXPERIMENTS ---"
mkdir -p results/cartpole/runtime
echo
for fullfile in src/horizon/experiments/cartpole/cpu/horizon_dqn_cpu_cp100.json; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    mkdir -p results/cartpole/horizon_dqn_cpu_cp100.json
    python src/horizon/run_evaluation.py -p src/horizon/experiments/cartpole/cpu/horizon_dqn_cpu_cp100.json -f results/cartpole/horizon_dqn_cpu_cp100.json/checkpoints.json -v results/cartpole/
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
#done
#for fullfile in src/horizon/experiments/cartpole/gpu/*.json; do 
 #   filename=$(basename -- "$fullfile")
  #  experiment="${filename%.*}"
   # echo "--- STARTING EXPERIMENT ${experiment} --- "
    #mkdir -p results/cartpole/${experiment}
    #python src/horizon/run_evaluation.py -g 0 -p src/horizon/experiments/cartpole/gpu/${experiment}.json -f results/cartpole/${experiment}/checkpoints.json -v results/cartpole/
    #echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    #echo
done
echo "--- HORIZON CARTPOLE EXPERIMENTS COMPLETED ---"
echo
echo "--- HORIZON EXPERIMENTS COMPLETED ---"
echo
