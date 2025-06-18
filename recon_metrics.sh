# export CUDA_VISIBLE_DEVICES="0"
export CUDA_VISIBLE_DEVICES="0"

REAL=""
FAKE=""

python -m evaluation.recons_metrics.recons --mode msssim --data_path $FAKE --gt_path $REAL
python -m evaluation.recons_metrics.recons --mode lpips --data_path $FAKE --gt_path $REAL
python -m evaluation.recons_metrics.recons --mode l2 --data_path $FAKE --gt_path $REAL


