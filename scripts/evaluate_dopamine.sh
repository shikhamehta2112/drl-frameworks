source ~/anaconda3/etc/profile.d/conda.sh

echo "--- STARTING DOPAMINE EXPERIMENTS ---"
conda activate dopamine-env
echo
echo "--- STARTING DOPAMINE CARTPOLE EXPERIMENTS ---"
mkdir -p results/cartpole/runtime
echo
for fullfile in src/dopamine/experiments/cartpole/rainbow/dopamine_rainbow_cpu_cp100.gin; do
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    python src/dopamine/run_evaluation.py --base_dir="results/cartpole/" --gin_files="src/dopamine/experiments/cartpole/rainbow/dopamine_rainbow_cpu_cp100.gin"
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done
echo "--- DOPAMINE CARTPOLE EXPERIMENTS COMPLETED ---"
echo
echo "--- DOPAMINE EXPERIMENTS COMPLETED ---"
echo
