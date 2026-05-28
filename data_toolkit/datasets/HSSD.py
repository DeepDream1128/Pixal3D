"""
HSSD auto-download module.

Downloads from the gated HuggingFace dataset ``hssd/hssd-models`` (license must
be accepted on HF first). TRELLIS-500K's HSSD.csv lists 6670 sha256 +
``file_identifier`` entries, where ``file_identifier`` is exactly the in-repo
path (e.g. ``objects/3/3e4790548c158671fec162757053201da04b6259.glb``), so we
download those files one by one through ``huggingface_hub.hf_hub_download``.

Requirements:
  - ``huggingface_hub`` installed (already a dependency)
  - ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``) environment variable set with
    a token that has access to ``hssd/hssd-models``
"""
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd


HSSD_REPO_ID = "hssd/hssd-models"
HSSD_REPO_TYPE = "dataset"


def add_args(parser: argparse.ArgumentParser):
    pass


def get_metadata(**kwargs):
    metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/HSSD.csv")
    return metadata


def _download_one(file_identifier: str, raw_dir: str, token: str):
    """Download a single GLB into raw_dir, preserving the file_identifier path."""
    from huggingface_hub import hf_hub_download
    target = os.path.join(raw_dir, file_identifier)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target
    os.makedirs(os.path.dirname(target), exist_ok=True)
    cached = hf_hub_download(
        repo_id=HSSD_REPO_ID,
        filename=file_identifier,
        repo_type=HSSD_REPO_TYPE,
        token=token,
        local_dir=raw_dir,
        local_dir_use_symlinks=False,
    )
    return cached


def download(metadata, output_dir=None, root=None, **kwargs):
    """Parallel download of GLBs listed in HSSD.csv into <output_dir>/raw/."""
    output_dir = output_dir or root
    raw_dir = os.path.join(output_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)

    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if not token:
        print('\n\033[93m[HSSD] WARN: HF_TOKEN not set — gated repo access will fail.\033[0m')

    workers = int(os.environ.get('HSSD_DOWNLOAD_WORKERS', '8'))
    print(f'  [HSSD] downloading {len(metadata)} files from {HSSD_REPO_ID} '
          f'with {workers} parallel workers')

    downloaded = {}
    fails = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_download_one, str(row['file_identifier']), raw_dir, token):
                (row['sha256'], str(row['file_identifier']))
            for _, row in metadata.iterrows()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc='HSSD'):
            sha256, fid = futures[fut]
            try:
                path = fut.result()
                downloaded[sha256] = os.path.relpath(path, output_dir)
            except Exception as e:
                fails.append((sha256, fid, str(e)))

    if fails:
        print(f'  [HSSD] {len(fails)} downloads failed (showing first 5):')
        for sha256, fid, msg in fails[:5]:
            print(f'    {sha256[:8]} {fid}: {msg}')

    print(f'  [HSSD] done: {len(downloaded)}/{len(metadata)} succeeded')
    return pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])


def _process_instance(args):
    metadatum, output_dir, func = args
    try:
        local_path = metadatum['local_path']
        sha256 = metadatum['sha256']
        file = os.path.join(output_dir, local_path)
        return func(file, sha256)
    except Exception as e:
        print(f"Error processing object {metadatum.get('sha256', '?')}: {e}")
        return None


def foreach_instance(metadata, output_dir, func, max_workers=None,
                     desc='Processing objects', timeout=None) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
    from tqdm import tqdm

    metadata = metadata.to_dict('records')
    max_workers = max_workers or os.cpu_count()
    records = []
    timeout_count = 0

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_instance, (m, output_dir, func)): m['sha256']
                for m in metadata
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                sha256 = futures[future]
                try:
                    r = future.result(timeout=timeout)
                    if r is not None:
                        records.append(r)
                except TimeoutError:
                    timeout_count += 1
                    print(f"Timeout processing object {sha256} (>{timeout}s)")
                    records.append({'sha256': sha256, 'error': f'Timeout (>{timeout}s)'})
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
    except Exception as e:
        print(f"Error happened during processing: {e}")

    if timeout_count > 0:
        print(f"Total timeout: {timeout_count} objects")
    return pd.DataFrame.from_records(records)
