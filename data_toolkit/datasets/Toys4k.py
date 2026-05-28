"""
Toys4k auto-download module (OBJ version).

Dropbox 链接（来自 Stojanov et al., CVPR 2021）：
  blend files: https://www.dropbox.com/s/8hi76lvl0x5o9si/toys4k_blend_files.zip
  obj files:   https://www.dropbox.com/s/k1joosmnnf8304o/toys4k_obj_files.zip
  point clouds: https://www.dropbox.com/s/gbsf9xkoaeevo6o/toys4k_point_clouds.zip

我们走 OBJ 版本：data_toolkit 下游的 dump_mesh / SDF 都吃 OBJ。

TRELLIS-500K 的 Toys4k.csv 给出 3229 条筛选后的 sha256 + ``file_identifier``
（形如 ``hammer/hammer_075/hammer_075.blend``）。这里把 .blend 替换成 .obj 来
对到 toys4k_obj_files.zip 解压后的实际路径。
"""
import os
import sys
import argparse
import zipfile
from urllib.request import urlretrieve
import pandas as pd


TOYS4K_OBJ_URL = "https://www.dropbox.com/s/k1joosmnnf8304o/toys4k_obj_files.zip?dl=1"
TOYS4K_OBJ_ZIPNAME = "toys4k_obj_files.zip"


def add_args(parser: argparse.ArgumentParser):
    pass


def get_metadata(**kwargs):
    metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/Toys4k.csv")
    return metadata


def _download_zip(zip_path: str, url: str = TOYS4K_OBJ_URL):
    """Download via curl (respects http(s)_proxy env vars). Skips if zip exists."""
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1024 * 1024:
        print(f'  [Toys4k] zip already present: {zip_path} '
              f'({os.path.getsize(zip_path) / 1e9:.2f} GB)')
        return
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    proxy_args = ''
    if os.environ.get('https_proxy'):
        proxy_args += f' -x {os.environ["https_proxy"]}'
    print(f'  [Toys4k] downloading {url} -> {zip_path}')
    cmd = (
        f'curl -L --fail --retry 5 --retry-delay 3 --continue-at - '
        f'{proxy_args} -o "{zip_path}" "{url}"'
    )
    rc = os.system(cmd)
    if rc != 0 or not os.path.exists(zip_path):
        raise RuntimeError(f'Failed to download {url} (curl rc={rc})')


def _extract_zip(zip_path: str, extract_to: str):
    """Extract toys4k_obj_files.zip into raw/. Skips already-extracted files."""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(zip_path)
    print(f'  [Toys4k] extracting {zip_path} -> {extract_to}')
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        for i, m in enumerate(members):
            target = os.path.join(extract_to, m)
            if os.path.exists(target) and not m.endswith('/'):
                continue
            zf.extract(m, extract_to)
            if (i + 1) % 1000 == 0:
                print(f'    extracted {i+1}/{len(members)}')
    print(f'  [Toys4k] extraction done ({len(members)} entries)')


def _find_obj_root(raw_dir: str) -> str:
    """Locate the directory that directly contains category folders (like hammer/, guitar/).

    Dropbox 上的 ``toys4k_obj_files.zip`` 解压后通常会在 ``raw_dir`` 下产生
    一层 ``toys4k_obj_files/`` 目录，再往里才是 ``hammer/``、``guitar/`` 这些
    类别。这里挑出"含有最多 ``<category>/<model>/`` 这种二级目录"的候选作为
    obj_root，避免误把 ``raw_dir`` 当根。
    """
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(raw_dir)

    def _category_score(d: str) -> int:
        """Count subdirs in ``d`` that themselves contain at least one subdir."""
        try:
            entries = os.listdir(d)
        except OSError:
            return 0
        score = 0
        for cat in entries:
            cat_path = os.path.join(d, cat)
            if not os.path.isdir(cat_path):
                continue
            try:
                for model in os.listdir(cat_path):
                    if os.path.isdir(os.path.join(cat_path, model)):
                        score += 1
                        break
            except OSError:
                continue
        return score

    candidates = [raw_dir]
    for entry in os.listdir(raw_dir):
        sub = os.path.join(raw_dir, entry)
        if os.path.isdir(sub):
            candidates.append(sub)

    best = raw_dir
    best_score = -1
    for c in candidates:
        s = _category_score(c)
        if s > best_score:
            best_score = s
            best = c
    return best


def download(metadata, output_dir=None, root=None, **kwargs):
    """Download Toys4k OBJ zip, extract, and build sha256 -> local_path map."""
    output_dir = output_dir or root
    raw_dir = os.path.join(output_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)

    zip_path = os.path.join(raw_dir, TOYS4K_OBJ_ZIPNAME)
    _download_zip(zip_path)
    _extract_zip(zip_path, raw_dir)

    obj_root = _find_obj_root(raw_dir)
    print(f'  [Toys4k] OBJ root: {obj_root}')

    downloaded = {}
    miss_count = 0
    for _, row in metadata.iterrows():
        sha256 = row['sha256']
        fid = str(row['file_identifier'])
        # OBJ zip 中每个模型目录下统一是 ``mesh.obj``，
        # 不像 BLEND zip 用 ``<category>_<id>.blend``。
        # 所以这里取 file_identifier 的目录部分再拼 ``mesh.obj``：
        # ``hammer/hammer_075/hammer_075.blend`` -> ``hammer/hammer_075/mesh.obj``
        rel_dir = os.path.dirname(fid)
        cand = os.path.join(obj_root, rel_dir, 'mesh.obj')
        if not os.path.exists(cand):
            # 兜底：万一某些版本仍然是 ``<id>.obj``
            alt = os.path.join(obj_root, fid[:-6] + '.obj') if fid.endswith('.blend') else None
            if alt and os.path.exists(alt):
                cand = alt
        if os.path.exists(cand):
            downloaded[sha256] = os.path.relpath(cand, output_dir)
        else:
            miss_count += 1

    print(f'  [Toys4k] matched {len(downloaded)}/{len(metadata)} '
          f'({miss_count} missing — likely categories the OBJ zip skipped)')
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
