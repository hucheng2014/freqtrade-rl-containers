#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

OPENCLAW_WORKDIR = Path("/home/jianglei/.openclaw/workspace")
NODE_BIN = Path("/home/jianglei/.nvm/versions/node/v22.22.0/bin/node")
OPENCLAW_MJS = Path("/home/jianglei/.nvm/versions/node/v22.22.0/lib/node_modules/openclaw/openclaw.mjs")
DEFAULT_TARGET = "194652099"
DEFAULT_MAX_CHARS = 3500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a report to Telegram via OpenClaw.")
    parser.add_argument("--title", default="", help="Optional title prefix for multipart messages.")
    parser.add_argument("--target", default=os.environ.get("TELEGRAM_TARGET", DEFAULT_TARGET))
    parser.add_argument("--max-chars", type=int, default=int(os.environ.get("REPORT_MAX_CHARS", DEFAULT_MAX_CHARS)))
    parser.add_argument("--dry-run", action="store_true", help="Print chunk metadata instead of sending.")
    return parser.parse_args()


def chunk_text(text: str, max_chars: int) -> list[str]:
    if not text:
        return [""]

    chunks: list[str] = []
    current = ""

    for raw_line in text.splitlines(keepends=True):
        line = raw_line
        while line:
            available = max_chars - len(current)
            if available <= 0:
                chunks.append(current)
                current = ""
                available = max_chars

            if len(line) <= available:
                current += line
                line = ""
                continue

            if current:
                chunks.append(current)
                current = ""
                continue

            chunks.append(line[:max_chars])
            line = line[max_chars:]

    if current:
        chunks.append(current)

    return chunks or [text[:max_chars]]


def prefix_chunks(chunks: list[str], title: str) -> list[str]:
    if len(chunks) <= 1:
        if title:
            return [f"{title}\n{chunks[0]}".strip() + "\n"]
        return chunks

    prefixed: list[str] = []
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        header = f"{title} ({index}/{total})" if title else f"({index}/{total})"
        prefixed.append(f"{header}\n{chunk}".strip() + "\n")
    return prefixed


def send_message(target: str, message: str) -> None:
    subprocess.run(
        [
            str(NODE_BIN),
            str(OPENCLAW_MJS),
            "message",
            "send",
            "--channel",
            "telegram",
            "--target",
            target,
            "--message",
            message,
        ],
        cwd=str(OPENCLAW_WORKDIR),
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )


def main() -> int:
    args = parse_args()
    report = sys.stdin.read()
    chunks = prefix_chunks(chunk_text(report, max(256, args.max_chars)), args.title.strip())

    if args.dry_run:
        print(f"dry_run target={args.target} chunks={len(chunks)}")
        for index, chunk in enumerate(chunks, start=1):
            print(f"chunk[{index}] chars={len(chunk)}")
        return 0

    for chunk in chunks:
        send_message(args.target, chunk)

    print(f"sent target={args.target} chunks={len(chunks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
