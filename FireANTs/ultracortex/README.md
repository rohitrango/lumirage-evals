

## Steps

1. Clean and standardize data: `python preprocess.py`
2. Register the data:
	- python register.py
	- python register.py --no-greedy --deformable_lr 0.25
python register.py --json_path /home/rohitrango/code/vfa/vfa/configs/ultracortex.json --output_dir ultracortex_results

# baseline

python register.py --json_path /home/rohitrango/code/vfa/vfa/configs/ultracortex.json --output_dir ultracortex_results --baseline
