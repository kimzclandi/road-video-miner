"""Download a frozen sequence sample using bounded public ZIP ranges."""

import concurrent.futures
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import requests
from PIL import Image
from remote_zip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip"
LABEL_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Immutable mismatch {path}")
    else:
        path.write_bytes(data)


def main():
    started = time.perf_counter()
    remote = RemoteZip(URL)
    z = zipfile.ZipFile(remote)
    infos = {
        i.filename: i
        for i in z.infolist()
        if i.filename.startswith("training/image_02/") and i.filename.endswith(".png")
    }
    byseq = {}
    for name in infos:
        byseq.setdefault(name.split("/")[2], []).append(name)
    # Every sequence except pilot 0009, whose first probe was viewed for cost only.
    seqs = sorted(set(byseq) - {"0009"})
    splits = {s: ("pool" if i < 4 else "dev" if i < 8 else "eval") for i, s in enumerate(seqs)}
    frames = []
    clips = []
    for seq in seqs:
        names = sorted(byseq[seq])
        n = len(names)
        if [int(Path(p).stem) for p in names] != list(range(n)):
            raise ValueError("Missing source frames")
        # Six complete 1-second clips spread over sequence; infer at 5Hz within them.
        starts = [round(i * (n - 10) / 5) for i in range(6)]
        if any(b < a + 10 for a, b in zip(starts, starts[1:])):
            raise ValueError("Overlapping clips")
        for start in starts:
            cid = f"{seq}-{start:06d}"
            ids = []
            for f in range(start, start + 10, 2):
                name = f"training/image_02/{seq}/{f:06d}.png"
                ids.append(name)
                frames.append(
                    dict(
                        id=name,
                        sequence=seq,
                        frame=f,
                        split=splits[seq],
                        clip=cid,
                        source_crc=infos[name].CRC,
                    )
                )
            clips.append(
                dict(
                    id=cid,
                    sequence=seq,
                    split=splits[seq],
                    start_frame=start,
                    end_frame_exclusive=start + 10,
                    fps=10,
                    seconds=1,
                    frames=ids,
                )
            )
    plan = {
        "source": URL,
        "etag": remote.etag,
        "size": remote.size,
        "splits": splits,
        "clips": clips,
        "frames": frames,
        "source_sequence_lengths": {s: len(byseq[s]) for s in seqs},
        "sampling": "5Hz observations within six disjoint one-second clips per sequence; unobserved frames are not decode failures",
    }
    save(ROOT / "reports/download_plan.json", (json.dumps(plan, indent=2) + "\n").encode())
    labelpath = ROOT / "data/labels.zip"
    if not labelpath.exists():
        resp = requests.get(LABEL_URL, timeout=60)
        resp.raise_for_status()
        save(labelpath, resp.content)
    with zipfile.ZipFile(labelpath) as lz:
        for seq in seqs:
            save(ROOT / f"data/labels/{seq}.txt", lz.read(f"training/label_02/{seq}.txt"))

    def fetch(r):
        target = ROOT / "data" / r["id"]
        info = infos[r["id"]]
        if target.exists():
            data = target.read_bytes()
        else:
            rr = RemoteZip.__new__(RemoteZip)
            rr.url = remote.url
            rr.etag = remote.etag
            rr.size = remote.size
            rr.pos = 0
            rr.bytes_read = 0
            rr.requests = 0
            data = rr.member(info)
            save(target, data)
        import zlib

        if zlib.crc32(data) != r["source_crc"]:
            raise ValueError("Corrupt cached frame")
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            size = list(im.size)
        return {**r, "sha256": hashlib.sha256(data).hexdigest(), "size": size}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for r in ex.map(fetch, frames):
            results.append(r)
            if len(results) % 50 == 0:
                print(f"downloaded {len(results)}/{len(frames)}", flush=True)
    hashes = [r["sha256"] for r in results]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate frames require explicit review before freezing")
    manifest = {
        **plan,
        "frames": results,
        "label_hashes": {
            s: hashlib.sha256((ROOT / f"data/labels/{s}.txt").read_bytes()).hexdigest() for s in seqs
        },
    }
    save(ROOT / "reports/manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps({"frames": len(results), "clips": len(clips), "seconds": time.perf_counter() - started}))


if __name__ == "__main__":
    main()
