"""Streaming helpers for Hugging Face datasets stored as Parquet shards with
embedded audio (alania-synthetic-speech-tr, turkish-tts-audiobooks).

The pattern is: list shards -> download one shard into a local dir (resumable,
no HF cache copy) -> process its row groups in a process pool -> write lhotse
manifests -> delete the shard. The next shard is downloaded in a background
thread while the current one is processed, so at most two shards are on disk.
"""

from __future__ import annotations

import concurrent.futures as cf
import fnmatch
import hashlib
import json
import logging
import random
import threading
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

log = logging.getLogger("hf_parquet")


def list_shards(repo_id: str, pattern: str, revision: str, token: Optional[str] = None) -> List[str]:
    from huggingface_hub import HfApi

    files = HfApi(token=token).list_repo_files(repo_id, repo_type="dataset", revision=revision)
    return sorted(f for f in files if fnmatch.fnmatch(f, pattern))


def order_shards(shards: List[str], order: str, seed: int) -> List[str]:
    shards = list(shards)
    if order == "shuffled":
        random.Random(seed).shuffle(shards)
    elif order != "sorted":
        raise ValueError(order)
    return shards


def download_shard(repo_id: str, filename: str, revision: str, local_dir: Path,
                   token: Optional[str] = None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id,
            filename,
            repo_type="dataset",
            revision=revision,
            local_dir=str(local_dir),
            token=token,
        )
    )


class Prefetcher:
    """Downloads shard i+1 in a thread while shard i is processed."""

    def __init__(self, fetch: Callable[[str], Path], names: Iterable[str]):
        self._fetch = fetch
        self._names = list(names)
        self._futures: Dict[str, cf.Future] = {}
        self._pool = cf.ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()

    def get(self, idx: int, prefetch_next: bool = True) -> Path:
        name = self._names[idx]
        fut = self._submit(name)
        if prefetch_next and idx + 1 < len(self._names):
            self._submit(self._names[idx + 1])
        return fut.result()

    def _submit(self, name: str) -> cf.Future:
        with self._lock:
            if name not in self._futures:
                self._futures[name] = self._pool.submit(self._fetch, name)
            return self._futures[name]

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


def num_row_groups(path: Path) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(str(path)).num_row_groups


def read_row_group(path: Path, idx: int, columns: List[str]) -> List[Dict]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    cols = [c for c in columns if c in pf.schema_arrow.names]
    return pf.read_row_group(idx, columns=cols).to_pylist()


def stable_percent(key: str) -> float:
    """Deterministic value in [0, 100) used for held-out voice/speaker selection."""
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 10000) / 100.0


class ShardState:
    """JSON file recording which shards are finished (for resuming)."""

    def __init__(self, path: Path):
        self.path = path
        self.data = {"done": {}}
        if path.exists():
            self.data = json.loads(path.read_text())

    def is_done(self, shard: str) -> bool:
        """Finished shards are skipped; partial (smoke-test) ones are redone by a full run."""
        info = self.data["done"].get(shard)
        return info is not None and not info.get("partial", False)

    def mark_done(self, shard: str, info: Dict) -> None:
        self.data["done"][shard] = info
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))
        tmp.replace(self.path)

    def total(self, key: str) -> float:
        return sum(float(v.get(key, 0.0)) for v in self.data["done"].values())
