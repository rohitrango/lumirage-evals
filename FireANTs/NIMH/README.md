python register.py --output_dir outputs_flair_synthseg/ --json_path /home/rohitrango/code/vfa/vfa/configs/nimh_FLAIR_synthseg.json --num_samples 5000
python register.py --output_dir outputs_t1_synthseg --json_path /data/rohitrango/code/vfa/vfa/configs/nimh_t1_synthseg.json
python register.py --output_dir outputs_t1_synthseg/ --json_path /home/rohitrango/code/vfa/vfa/configs/nimh_t1_synthseg.json --baseline
python register.py --output_dir outputs_t2_synthseg/ --json_path /home/rohitrango/code/vfa/vfa/configs/nimh_T2_synthseg.json --baseline --num_samples 5000
