from glob import glob
import json

configs = glob("*.json")
for config in configs:
    with open(config, "r") as f:
        data = json.load(f)
    try:
        print(f"{config} has labels: {len(data['labels'])}")
    except:
        print(config, data)
