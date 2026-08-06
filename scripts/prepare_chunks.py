#!/usr/bin/env python3
"""
Job 'prepare': pecah folder frame hasil extract ffmpeg jadi N chunk yang
bisa diproses PARALEL di job matrix, tanpa merusak window temporal
(TEMPORAL_WINDOW=2) di sambungan antar-chunk.

Caranya: tiap chunk dapat "core range" (frame yang benar-benar milik
chunk itu, akan ditulis outputnya) + overlap TEMPORAL_WINDOW frame di
kiri & kanan (dipakai HANYA sebagai konteks window, tidak ditulis
output-nya). Overlap ini persis meniru bagaimana original script
mengambil window +/-2 frame dari SELURUH video -- bedanya sekarang
window itu diambil dari overlap antar-chunk, bukan dari file yang sama.
"""
import os
import glob
import json
import argparse

TEMPORAL_WINDOW = 2  # HARUS sama dengan detect_and_mask.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="frames_in")
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--out-prefix", default="chunk")
    args = parser.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    total = len(frame_paths)
    if total == 0:
        raise SystemExit(f"Error: tidak ada frame ditemukan di {args.frames_dir}")

    # Tidak mungkin bikin chunk lebih banyak dari jumlah frame yang ada.
    n = max(1, min(args.num_chunks, total))
    base = total // n
    rem = total % n

    # Bagi total frame jadi n rentang core yang hampir sama besar
    # (rentang pertama 'rem' dapat +1 frame supaya semua frame terpakai).
    boundaries = []
    start = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        end = start + size
        boundaries.append((start, end))
        start = end

    manifest = []
    for i, (core_start, core_end) in enumerate(boundaries):
        ov_start = max(0, core_start - TEMPORAL_WINDOW)
        ov_end = min(total, core_end + TEMPORAL_WINDOW)

        chunk_dir = f"{args.out_prefix}_{i}"
        os.makedirs(chunk_dir, exist_ok=True)
        for idx in range(ov_start, ov_end):
            src = frame_paths[idx]
            dst = os.path.join(chunk_dir, os.path.basename(src))
            if not os.path.exists(dst):
                # hardlink (bukan copy) supaya hemat waktu & disk
                try:
                    os.link(src, dst)
                except OSError:
                    import shutil
                    shutil.copy2(src, dst)

        meta = {
            "chunk_index": i,
            # index RELATIF ke folder chunk_i (dipakai langsung oleh
            # detect_and_mask.py sebagai --core-start-index/--core-end-index)
            "core_start_index": core_start - ov_start,
            "core_end_index": core_end - ov_start,
            # index GLOBAL (dipakai kalau perlu debug urutan asli)
            "global_core_start_frame": core_start,
            "global_core_end_frame": core_end,
        }
        with open(os.path.join(chunk_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        manifest.append(meta)
        print(f"Chunk {i}: frame global core [{core_start},{core_end}) "
              f"| overlap konteks [{ov_start},{ov_end}) -> {chunk_dir}/ "
              f"({ov_end - ov_start} frame)")

    with open("chunks_manifest.json", "w") as f:
        json.dump({"total_frames": total, "num_chunks": n, "chunks": manifest}, f, indent=2)

    print(f"\nTotal {total} frame dipecah jadi {n} chunk "
          f"(overlap={TEMPORAL_WINDOW} frame tiap sisi, sama seperti TEMPORAL_WINDOW original).")


if __name__ == "__main__":
    main()
