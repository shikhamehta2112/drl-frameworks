
echo "--- STARTING EVALUATION ---"
echo

#echo "--- REMOVING PREVIOUS RESULTS ---"
#bash ./scripts/clean.sh
#echo

echo "--- CONFIGURING ANACONDA ---"
source ~/anaconda3/etc/profile.d/conda.sh
echo

#bash ./scripts/evaluate_dopamine_all.sh

#conda deactivate

bash ./scripts/evaluate_ray.sh

conda deactivate

echo "--- EVALUATION COMPLETED ---"
