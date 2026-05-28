import os
import argparse
import zipfile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import objaverse.xl as oxl
from utils import get_file_hash


# ---- Patch: tolerate non-UTF8 (surrogate-escape) filenames in zipfile ----
# objaverse.xl.github._process_repo calls shutil.make_archive(...,'zip',...) on
# cloned repos; if any file inside has bytes that round-tripped through
# fs-decoder via 'surrogateescape' (e.g. emoji on a non-UTF8 fs, '\udcXX'),
# zipfile.ZipInfo._encodeFilenameFlags raises UnicodeEncodeError inside a
# multiprocessing worker and kills the entire download pool.
# We replace the offending characters so the zip can still be produced.
_orig_encode_filename_flags = zipfile.ZipInfo._encodeFilenameFlags


def _safe_encode_filename_flags(self):
    try:
        return _orig_encode_filename_flags(self)
    except UnicodeEncodeError:
        # surrogate-escape or other invalid chars: lossy fallback
        safe = self.filename.encode('utf-8', 'replace').decode('utf-8')
        self.filename = safe
        return safe.encode('utf-8'), self.flag_bits | 0x800


zipfile.ZipInfo._encodeFilenameFlags = _safe_encode_filename_flags
# ------------------------------------------------------------------------


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--source', type=str, default='sketchfab',
                        help='Data source to download annotations from (github, sketchfab)')


def get_metadata(source, **kwargs):
    if source == 'sketchfab':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_sketchfab.csv")
    elif source == 'github':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_github.csv")
    else:
        raise ValueError(f"Invalid source: {source}")
    return metadata
        

def download(metadata, output_dir, **kwargs):
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)

    # download annotations
    annotations = oxl.get_annotations()
    annotations = annotations[annotations['sha256'].isin(metadata['sha256'].values)]

    # 控制并发：默认 = min(4, cpu_count)。kwargs 里允许 download_processes 覆盖。
    # 主因：本地 HTTP 代理 (例如 17897) 在 12 路 git clone 并发时会被打爆，
    # 触发大量 'GnuTLS recv error (-110)' 导致仓库被记成 Could not clone。
    src = kwargs.get('source', '')
    default_proc = 4 if src == 'github' else min(8, os.cpu_count() or 1)
    processes = int(kwargs.get('download_processes') or default_proc)

    # 用 handle_missing_object 把 *彻底* clone 不到的 sha256 标到 metadata，
    # 防止下一轮 download 又去重试已删/已转私的仓库。
    missing_records = []
    seen_missing = set()

    def _on_missing(file_identifier, sha256, metadata=None):
        if sha256 in seen_missing:
            return
        seen_missing.add(sha256)
        missing_records.append({'sha256': sha256, 'local_path': '__MISSING__'})

    # download and render objects
    file_paths = oxl.download_objects(
        annotations,
        download_dir=os.path.join(output_dir, "raw"),
        save_repo_format="zip",
        processes=processes,
        handle_missing_object=_on_missing,
    )

    downloaded = {}
    metadata = metadata.set_index("file_identifier")
    for k, v in file_paths.items():
        sha256 = metadata.loc[k, "sha256"]
        downloaded[sha256] = os.path.relpath(v, output_dir)

    df = pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])
    if missing_records:
        # 排掉已经成功下载的（同一个 repo 里某些 mesh 落盘成功、另一些 fileIdentifier
        # 没匹配到也会触发 missing；那些 sha256 会同时出现在 downloaded 里，以下载为准）
        downloaded_sha = set(df['sha256'].tolist())
        missing_records = [r for r in missing_records if r['sha256'] not in downloaded_sha]
        if missing_records:
            df = pd.concat([df, pd.DataFrame(missing_records)], ignore_index=True)
    return df


def _process_instance(args):
    """Worker function for ProcessPoolExecutor (must be top-level for pickling)"""
    import os, tempfile, zipfile
    metadatum, output_dir, func = args
    try:
        local_path = metadatum['local_path']
        sha256 = metadatum['sha256']
        
        direct_file_path = os.path.join(output_dir, local_path)
        if os.path.exists(direct_file_path):
            file = direct_file_path
            record = func(file, sha256)
        elif local_path.startswith('raw/github/repos/'):
            path_parts = local_path.split('/')
            file_name = os.path.join(*path_parts[5:])
            zip_file = os.path.join(output_dir, *path_parts[:5])
            if os.path.exists(zip_file):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(tmp_dir)
                    file = os.path.join(tmp_dir, file_name)
                    record = func(file, sha256)
            else:
                # zip file not found, pass local_path directly (for tasks like dual_grid_view that don't need the original file)
                file = local_path
                record = func(file, sha256)
        else:
            file = os.path.join(output_dir, local_path)
            record = func(file, sha256)
        return record
    except Exception as e:
        print(f"Error processing object {metadatum.get('sha256', '?')}: {e}")
        return None


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects', log_interval=500, timeout=None) -> pd.DataFrame:
    print("================")
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
    from tqdm import tqdm
    
    # load metadata
    metadata = metadata.to_dict('records')

    max_workers = max_workers or os.cpu_count()
    records = []
    
    # Track processed/skipped counts
    total_processed = 0
    total_skipped = 0
    timeout_count = 0
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_instance, (m, output_dir, func)): m['sha256']
                for m in metadata
            }
            pbar = tqdm(as_completed(futures), total=len(futures), desc=desc)
            for future in pbar:
                sha256 = futures[future]
                try:
                    r = future.result(timeout=timeout)
                    if r is not None:
                        records.append(r)
                        # Update stats
                        if '_processed_count' in r:
                            total_processed += r['_processed_count']
                        if '_skipped_count' in r:
                            total_skipped += r['_skipped_count']
                        # Update progress bar display
                        pbar.set_postfix(processed=total_processed, skipped=total_skipped, timeout=timeout_count, refresh=False)
                except TimeoutError:
                    timeout_count += 1
                    print(f"Timeout processing object {sha256} (>{timeout}s)")
                    records.append({'sha256': sha256, 'error': f'Timeout (>{timeout}s)'})
                    pbar.set_postfix(processed=total_processed, skipped=total_skipped, timeout=timeout_count, refresh=False)
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
    except Exception as e:
        print(f"Error happened during processing: {e}")
    
    if timeout_count > 0:
        print(f"Total timeout: {timeout_count} objects")
        
    return pd.DataFrame.from_records(records)
