## Preprocessing steps

1. Run nBEST on the brains
2. Run `preprocess.py` to crop the cerebrum / subcortical / tissue images to the mask bbox and pad to the target shape
3. Run `affinealign.py` to align the cerebrum / subcortical / tissue images to the first image
4. Run `register.py` to register the imagespython register.py --output_dir outputs_subcortical_v2 --json_path /data/rohitrango/code/vfa/vfa/configs/primede_subcortical.json

For baseline:
python register.py --output_dir outputs_tissues_v2/ --json_path /data/rohitrango/code/vfa/vfa/configs/primede_tissue.json --baseline
