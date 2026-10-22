"""Download the complete Bridge LeRobot dataset and retain the resolved revision."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ID = 'IPEC-COMMUNITY/bridge_orig_lerobot'


def main():
    parser = argparse.ArgumentParser(description='下载完整 Bridge LeRobot 数据集')
    parser.add_argument('--dest', type=Path, required=True, help='数据保存目录')
    args = parser.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    dest = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / 'download_manifest.json'

    if manifest_path.exists():
        record = json.loads(manifest_path.read_text(encoding='utf-8'))
        if record.get('repo_id') != REPO_ID or not record.get('revision'):
            parser.error('已有下载记录与本数据集不匹配，请选择新的空目录。')
        revision = record['revision']
    else:
        if any(dest.iterdir()):
            parser.error('首次下载请使用空目录，避免混入其他数据或版本。')
        revision = HfApi().dataset_info(REPO_ID, revision='main').sha
        if not revision:
            raise RuntimeError('未能解析数据版本，未开始下载。')

    record = dict(repo_id=REPO_ID, revision=revision, scope='entire_repository',
                  status='incomplete', local_dir=str(dest),
                  started_utc=datetime.now(timezone.utc).isoformat())

    def save_record():
        temporary = dest / 'download_manifest.json.tmp'
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(manifest_path)

    save_record()
    print(f'数据集：{REPO_ID}\n版本：{revision}\n保存位置：{dest}', flush=True)
    snapshot_download(repo_id=REPO_ID, repo_type='dataset', revision=revision,
                      local_dir=str(dest))
    record.update(status='download_completed', completed_utc=datetime.now(timezone.utc).isoformat())
    save_record()
    print(f'数据集下载完成。\n数据位置：{dest}\n版本记录：{manifest_path}')


if __name__ == '__main__':
    main()
