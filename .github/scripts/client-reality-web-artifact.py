#!/usr/bin/env python3
"""Create and verify the exact production Web artifact used by Client Reality CI."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = 1
ARCHIVE_NAME = "client-reality-web.tar.gz"
MANIFEST_NAME = "client-reality-web-manifest.json"
NEXT_ROOT = PurePosixPath("apps/web/.next")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_paths(next_dir: Path):
    for path in sorted(next_dir.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(next_dir)
        if relative.parts and relative.parts[0] == "cache":
            continue
        yield path, relative


def tree_digest(next_dir: Path) -> str:
    digest = hashlib.sha256()
    for path, relative in included_paths(next_dir):
        rel = relative.as_posix().encode()
        if path.is_symlink():
            digest.update(b"L\0" + rel + b"\0" + os.readlink(path).encode() + b"\n")
        elif path.is_dir():
            digest.update(b"D\0" + rel + b"\n")
        elif path.is_file():
            digest.update(
                b"F\0"
                + rel
                + b"\0"
                + str(path.stat().st_size).encode()
                + b"\0"
                + sha256_file(path).encode()
                + b"\n"
            )
    return digest.hexdigest()


def normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    if info.isdir():
        info.mode = 0o755
    elif info.isfile():
        info.mode = 0o644
    return info


def add_path(tar: tarfile.TarFile, source: Path, arcname: PurePosixPath) -> None:
    info = normalized(tar.gettarinfo(str(source), arcname.as_posix()))
    if info.isfile():
        with source.open("rb") as handle:
            tar.addfile(info, handle)
    else:
        tar.addfile(info)


def create_bundle(root: Path, output_dir: Path, manifest: dict[str, object]) -> tuple[Path, Path]:
    next_dir = root / "apps/web/.next"
    if not (next_dir / "BUILD_ID").is_file():
        raise SystemExit("apps/web/.next/BUILD_ID is missing; a production Web build is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / ARCHIVE_NAME
    checksum = output_dir / f"{ARCHIVE_NAME}.sha256"
    manifest = {**manifest, "schemaVersion": SCHEMA_VERSION, "buildTreeSha256": tree_digest(next_dir)}
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()

    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as tar:
                manifest_info = tarfile.TarInfo(MANIFEST_NAME)
                manifest_info.size = len(manifest_bytes)
                manifest_info.mode = 0o644
                manifest_info = normalized(manifest_info)
                tar.addfile(manifest_info, io.BytesIO(manifest_bytes))

                root_info = tarfile.TarInfo(NEXT_ROOT.as_posix())
                root_info.type = tarfile.DIRTYPE
                root_info.mode = 0o755
                tar.addfile(normalized(root_info))
                for source, relative in included_paths(next_dir):
                    add_path(tar, source, NEXT_ROOT / PurePosixPath(relative.as_posix()))

    checksum.write_text(f"{sha256_file(archive)}  {ARCHIVE_NAME}\n")
    return archive, checksum


def read_checksum(checksum: Path) -> str:
    parts = checksum.read_text().strip().split()
    if len(parts) != 2 or parts[1] != ARCHIVE_NAME or len(parts[0]) != 64:
        raise SystemExit("Client Reality artifact checksum sidecar is malformed")
    return parts[0].lower()


def read_manifest(archive: Path) -> dict[str, object]:
    with tarfile.open(archive, mode="r:gz") as tar:
        member = tar.getmember(MANIFEST_NAME)
        handle = tar.extractfile(member)
        if handle is None:
            raise SystemExit("Client Reality artifact manifest is unreadable")
        return json.loads(handle.read())


def safe_members(tar: tarfile.TarFile):
    for member in tar.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"Unsafe Client Reality artifact member: {member.name}")
        if member.name == MANIFEST_NAME:
            continue
        if path != NEXT_ROOT and NEXT_ROOT not in path.parents:
            raise SystemExit(f"Unexpected Client Reality artifact member: {member.name}")
        yield member


def verify_bundle(
    root: Path,
    archive: Path,
    checksum: Path,
    *,
    expected_checkout_sha: str,
    expected_head_sha: str,
    expected_base_sha: str,
    expected_api_url: str,
    extract: bool,
) -> dict[str, object]:
    expected_checksum = read_checksum(checksum)
    actual_checksum = sha256_file(archive)
    if actual_checksum != expected_checksum:
        raise SystemExit(
            f"Client Reality artifact checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
        )

    manifest = read_manifest(archive)
    expected = {
        "schemaVersion": SCHEMA_VERSION,
        "checkoutSha": expected_checkout_sha,
        "eventHeadSha": expected_head_sha,
        "eventBaseSha": expected_base_sha,
        "apiUrl": expected_api_url,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise SystemExit(
                f"Client Reality artifact manifest mismatch for {key}: expected {value!r}, got {manifest.get(key)!r}"
            )

    if extract:
        target = root / "apps/web/.next"
        shutil.rmtree(target, ignore_errors=True)
        with tarfile.open(archive, mode="r:gz") as tar:
            members = list(safe_members(tar))
            tar.extractall(root, members=members, filter="data")
        actual_tree = tree_digest(target)
        if manifest.get("buildTreeSha256") != actual_tree:
            raise SystemExit(
                "Client Reality artifact tree digest does not match its manifest after extraction"
            )
        if not (target / "BUILD_ID").is_file():
            raise SystemExit("Verified Client Reality artifact did not contain a Next BUILD_ID")

    return manifest


def git_sha(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def create_command(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    checkout_sha = git_sha(root)
    manifest = {
        "checkoutSha": checkout_sha,
        "eventHeadSha": args.event_head_sha,
        "eventBaseSha": args.event_base_sha,
        "apiUrl": args.api_url,
        "nodeVersion": args.node_version,
        "pnpmVersion": args.pnpm_version,
        "buildCommand": "pnpm --filter @woof/web build",
        "excludedBuildPaths": ["apps/web/.next/cache"],
    }
    archive, checksum = create_bundle(root, Path(args.output_dir), manifest)
    print(f"Created {archive} with sha256={sha256_file(archive)}")
    print(checksum)


def verify_command(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    checkout_sha = git_sha(root)
    manifest = verify_bundle(
        root,
        Path(args.archive),
        Path(args.checksum),
        expected_checkout_sha=checkout_sha,
        expected_head_sha=args.event_head_sha,
        expected_base_sha=args.event_base_sha,
        expected_api_url=args.api_url,
        extract=True,
    )
    print(
        "Verified Client Reality Web artifact "
        f"checkout={manifest['checkoutSha']} head={manifest['eventHeadSha']} tree={manifest['buildTreeSha256']}"
    )


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        next_dir = root / "apps/web/.next"
        (next_dir / "static/chunks").mkdir(parents=True)
        (next_dir / "cache").mkdir()
        (next_dir / "BUILD_ID").write_text("build-1\n")
        (next_dir / "static/chunks/app.js").write_text("console.log('woof')\n")
        (next_dir / "cache/ignored").write_text("nondeterministic cache")
        checkout = "a" * 40
        head = "b" * 40
        base = "c" * 40
        api = "http://127.0.0.1:59999/api/v1"
        out = root / "out"
        archive, checksum = create_bundle(
            root,
            out,
            {
                "checkoutSha": checkout,
                "eventHeadSha": head,
                "eventBaseSha": base,
                "apiUrl": api,
            },
        )
        shutil.rmtree(next_dir)
        verify_bundle(
            root,
            archive,
            checksum,
            expected_checkout_sha=checkout,
            expected_head_sha=head,
            expected_base_sha=base,
            expected_api_url=api,
            extract=True,
        )
        if (next_dir / "cache").exists():
            raise SystemExit("Self-test restored excluded Next build cache")

        rejected = False
        try:
            verify_bundle(
                root,
                archive,
                checksum,
                expected_checkout_sha=checkout,
                expected_head_sha="d" * 40,
                expected_base_sha=base,
                expected_api_url=api,
                extract=False,
            )
        except SystemExit:
            rejected = True
        if not rejected:
            raise SystemExit("Self-test failed to reject a mismatched PR head SHA")

        data = bytearray(archive.read_bytes())
        data[-1] ^= 1
        archive.write_bytes(data)
        rejected = False
        try:
            verify_bundle(
                root,
                archive,
                checksum,
                expected_checkout_sha=checkout,
                expected_head_sha=head,
                expected_base_sha=base,
                expected_api_url=api,
                extract=False,
            )
        except SystemExit:
            rejected = True
        if not rejected:
            raise SystemExit("Self-test failed to reject a tampered artifact")

    print("Client Reality shared Web artifact self-test passed.")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--root", default=".")
    create.add_argument("--output-dir", required=True)
    create.add_argument("--event-head-sha", required=True)
    create.add_argument("--event-base-sha", required=True)
    create.add_argument("--api-url", required=True)
    create.add_argument("--node-version", required=True)
    create.add_argument("--pnpm-version", required=True)
    create.set_defaults(func=create_command)

    verify = sub.add_parser("verify")
    verify.add_argument("--root", default=".")
    verify.add_argument("--archive", required=True)
    verify.add_argument("--checksum", required=True)
    verify.add_argument("--event-head-sha", required=True)
    verify.add_argument("--event-base-sha", required=True)
    verify.add_argument("--api-url", required=True)
    verify.set_defaults(func=verify_command)

    sub.add_parser("self-test").set_defaults(func=lambda _args: self_test())
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
