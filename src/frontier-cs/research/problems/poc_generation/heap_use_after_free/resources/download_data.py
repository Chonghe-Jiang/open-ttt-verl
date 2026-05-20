import json
import os
import sys
import requests

data = json.loads(sys.argv[1])
print(f"[download_data] Current directory: {os.getcwd()}")
print(f"[download_data] Datasets to download: {data}")

for id in data:
    path = id.split(':')
    dataset_dir = f"datasets/{path[0]}/{path[1]}"
    print(f"[download_data] Creating directory: {dataset_dir}")
    os.makedirs(dataset_dir, exist_ok=True)

    # Download description
    desc_url = f"https://huggingface.co/datasets/sunblaze-ucb/cybergym/resolve/main/data/{path[0]}/{path[1]}/description.txt"
    print(f"[download_data] Downloading {desc_url} ...")
    r = requests.get(desc_url)
    r.raise_for_status()  # Raise exception if download failed
    with open(f"{dataset_dir}/description.txt", "wb") as f:
        f.write(r.content)
    print(f"[download_data] Saved description.txt ({len(r.content)} bytes)")

    # Download source tarball
    src_url = f"https://huggingface.co/datasets/sunblaze-ucb/cybergym/resolve/main/data/{path[0]}/{path[1]}/repo-vul.tar.gz"
    print(f"[download_data] Downloading {src_url} ...")
    r = requests.get(src_url, stream=True)
    r.raise_for_status()
    total_size = 0
    with open(f"{dataset_dir}/repo-vul.tar.gz", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            total_size += len(chunk)
    print(f"[download_data] Saved repo-vul.tar.gz ({total_size} bytes)")

    # Try to download run scripts (optional - may not exist in all datasets)
    for script in ["run_vul.sh", "run_fix.sh"]:
        script_url = f"https://huggingface.co/datasets/sunblaze-ucb/cybergym/resolve/main/data/{path[0]}/{path[1]}/{script}"
        print(f"[download_data] Downloading {script_url} ...")
        r = requests.get(script_url)
        if r.status_code == 200:
            with open(f"{dataset_dir}/{script}", "wb") as f:
                f.write(r.content)
            print(f"[download_data] Saved {script} ({len(r.content)} bytes)")
        else:
            print(f"[download_data] WARNING: {script} not found (HTTP {r.status_code}), skipping")

    print(f"[download_data] Completed download for {id}")
    # Verify required files exist
    for fname in ["description.txt", "repo-vul.tar.gz"]:
        fpath = f"{dataset_dir}/{fname}"
        if os.path.exists(fpath):
            print(f"[download_data] Verified: {fpath} exists ({os.path.getsize(fpath)} bytes)")
        else:
            print(f"[download_data] ERROR: Required file {fpath} NOT FOUND!")
    # Check optional files
    for fname in ["run_vul.sh", "run_fix.sh"]:
        fpath = f"{dataset_dir}/{fname}"
        if os.path.exists(fpath):
            print(f"[download_data] Optional: {fpath} exists ({os.path.getsize(fpath)} bytes)")
        else:
            print(f"[download_data] Optional: {fpath} not downloaded (may need to be generated)")
