source ~/anaconda3/etc/profile.d/conda.sh

echo "--- STARTING RAY EXPERIMENTS ---"
conda activate ray-env
echo
echo "--- STARTING RAY CARTPOLE EXPERIMENTS ---"
mkdir -p src/results/ray/cartpole/A2C/runtime
echo
for fullfile in src/ray/experiments/cartpole/A2C/ray_a2c_cpu_cp110.yml; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    mkdir ${filename%.*}
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    python src/ray/run_evaluation.py -f="src/ray/experiments/cartpole/A2C/ray_a2c_cpu_cp110.yml"
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done

echo "--- RAY CARTPOLE EXPERIMENTS COMPLETED ---"
echo
echo "--- RAY EXPERIMENTS COMPLETED ---"
echo

