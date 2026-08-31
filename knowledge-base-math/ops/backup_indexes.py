"""
ops/backup_indexes.py - Archive the indexes off the pod.

DEPLOYMENT.md §6 has carried "no backups, and now nothing to rebuild from" as an open
risk since uploaded PDFs stopped being retained. That change was right — the pod holds no
copy of the family's library — but it inverted what is derived data. chroma_db/ and
bm25_indexes/ used to be regenerable from the PDFs beside them; they are now the ONLY
copy that exists anywhere except the family's own laptops (ARCHITECTURE.md §5).

A RunPod network volume is one delete, one unpaid storage charge, or one wrong region
away from gone, and it does not follow you to another datacenter. That last point is why
this belongs to the containerisation work rather than being a separate chore: with the
image reproducible, an archive here is the difference between "recreate the pod anywhere
in two minutes" and "recreate the pod and then ask everyone to re-upload their textbooks
and pay for Marker again".

    python -m ops.backup_indexes            # one archive, now
    python -m ops.backup_indexes --dry-run  # what it would archive and where

Configuration (all no-ops if KBM_BACKUP_URL is unset — a laptop run backs nothing up):

    KBM_BACKUP_URL       s3://bucket/optional/prefix
    KBM_BACKUP_ENDPOINT  S3-compatible endpoint (Backblaze B2, Cloudflare R2, RunPod's
                         own S3 API). Omit for real AWS.
    KBM_BACKUP_KEEP      how many archives to retain (default 7)
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY   standard boto3 credentials

What is archived, and what is deliberately not: the indexes and the telemetry log, which
are irreplaceable, and not the model caches, which are a re-download. That asymmetry is
the whole point — bandwidth is recoverable, the corpus is not.
"""

import argparse
import datetime as _dt
import os
import sys
import tarfile
import tempfile

from kbm.config import BM25_DIR, CHROMA_DIR, DATA_DIR, TELEMETRY_PATH

BACKUP_URL = os.environ.get("KBM_BACKUP_URL", "").strip()
BACKUP_ENDPOINT = os.environ.get("KBM_BACKUP_ENDPOINT", "").strip()
BACKUP_KEEP = int(os.environ.get("KBM_BACKUP_KEEP", "7"))


def enabled() -> bool:
    return bool(BACKUP_URL)


def _targets() -> list[tuple[str, str]]:
    """(path on disk, name inside the archive) for everything worth keeping."""
    out = []
    for path, arcname in (
        (CHROMA_DIR, "chroma_db"),
        (BM25_DIR, "bm25_indexes"),
        (os.path.dirname(TELEMETRY_PATH), "telemetry"),
    ):
        if os.path.exists(path):
            out.append((path, arcname))
    return out


def _parse_url(url: str) -> tuple[str, str]:
    """s3://bucket/prefix -> (bucket, prefix). Prefix may be empty."""
    if not url.startswith("s3://"):
        raise ValueError(f"KBM_BACKUP_URL must start with s3:// — got {url!r}")
    rest = url[len("s3://"):].strip("/")
    bucket, _, prefix = rest.partition("/")
    if not bucket:
        raise ValueError(f"KBM_BACKUP_URL has no bucket: {url!r}")
    return bucket, prefix


def create_archive(dest_dir: str) -> tuple[str, int]:
    """Tar the index directories into dest_dir. Returns (path, bytes)."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(dest_dir, f"kbm-indexes-{stamp}.tar.gz")
    with tarfile.open(path, "w:gz") as tar:
        for src, arcname in _targets():
            tar.add(src, arcname=arcname)
    return path, os.path.getsize(path)


def _client():
    # Imported here, not at module scope. boto3 is only needed by a deployment that
    # actually configured a backup target, and a missing optional dependency must not
    # make this module unimportable — api/routes.py imports it on every ingest.
    import boto3

    return boto3.client("s3", endpoint_url=BACKUP_ENDPOINT or None)


def _prune(client, bucket: str, prefix: str) -> int:
    """Keep the newest BACKUP_KEEP archives; delete the rest. Returns how many went."""
    if BACKUP_KEEP <= 0:
        return 0
    listing = client.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}kbm-indexes-")
    keys = sorted((o["Key"] for o in listing.get("Contents", [])), reverse=True)
    stale = keys[BACKUP_KEEP:]
    for key in stale:
        client.delete_object(Bucket=bucket, Key=key)
    return len(stale)


def backup_indexes() -> str:
    """Archive and upload. Returns a one-line status. **Never raises.**

    Same contract as kbm/telemetry.py, and for the same reason: this is called from the
    ingest path, and a backup that fails an upload the family just waited ten minutes for
    would be a worse outcome than no backup at all. Failures go to the log, where an
    operator can act on them; the job carries on.
    """
    if not enabled():
        return "backup: disabled (KBM_BACKUP_URL unset)"
    try:
        bucket, prefix = _parse_url(BACKUP_URL)
        prefix = f"{prefix}/" if prefix else ""
        targets = _targets()
        if not targets:
            return f"backup: nothing to archive under {DATA_DIR}"

        with tempfile.TemporaryDirectory(prefix="kbm_backup_") as tmp:
            path, size = create_archive(tmp)
            key = f"{prefix}{os.path.basename(path)}"
            client = _client()
            client.upload_file(path, bucket, key)
            pruned = _prune(client, bucket, prefix)

        mb = size / 1_048_576
        tail = f", pruned {pruned} old" if pruned else ""
        return f"backup: uploaded {key} ({mb:.1f} MB){tail}"
    except Exception as e:  # noqa: BLE001 - a backup must never break the caller
        return f"backup FAILED: {type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be archived and where, upload nothing")
    args = ap.parse_args()

    print(f"DATA_DIR : {DATA_DIR}")
    for src, arcname in _targets():
        print(f"  archive: {src}  ->  {arcname}/")
    if not _targets():
        print("  (nothing exists to archive)")
    print(f"target   : {BACKUP_URL or '(unset — backups disabled)'}")
    if BACKUP_ENDPOINT:
        print(f"endpoint : {BACKUP_ENDPOINT}")
    print(f"retain   : {BACKUP_KEEP} archives")

    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="kbm_backup_") as tmp:
            path, size = create_archive(tmp)
            print(f"dry-run  : would upload {os.path.basename(path)} "
                  f"({size / 1_048_576:.1f} MB)")
        return 0

    status = backup_indexes()
    print(status)
    return 1 if "FAILED" in status else 0


if __name__ == "__main__":
    sys.exit(main())
