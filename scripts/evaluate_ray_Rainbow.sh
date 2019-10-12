
source ~/anaconda3/etc/profile.d/conda.sh

echo "--- STARTING RAY EXPERIMENTS ---"
conda activate ray-env
echo
echo "--- STARTING RAY CARTPOLE EXPERIMENTS ---"
mkdir -p src/results/ray/cartpole/rainbow/runtime
echo
for fullfile in src/ray/experiments/cartpole/rainbow2/ray_rainbow_cpu_cp103.yml; do 
    filename=$(basename -- "$fullfile")
    experiment="${filename%.*}"
    mkdir ${filename%.*}
    echo "--- STARTING EXPERIMENT ${experiment} --- "
    python src/ray/run_evaluation.py -f="src/ray/experiments/cartpole/rainbow2/ray_rainbow_cpu_cp103.yml"
    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
    echo
done

#for fullfile in src/ray/experiments/cartpole/a2c/*; do 
#    filename=$(basename -- "$fullfile")
#    experiment="${filename%.*}"
#    echo "--- STARTING EXPERIMENT ${experiment} --- "
#    python src/ray/run_evaluation.py -f="src/ray/experiments/cartpole/a2c/$filename/"
#    echo "--- EXPERIMENT ${experiment} COMPLETED --- "
#    echo
#done

echo "--- RAY CARTPOLE EXPERIMENTS COMPLETED ---"
echo
#rm -rf ~/ray_results
echo "--- RAY EXPERIMENTS COMPLETED ---"
echo

