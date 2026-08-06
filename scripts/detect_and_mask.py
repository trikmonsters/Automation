#!/usr/bin/env python3
"""
"""
import os
import sys
import glob
import time
import argparse
import numpy as np
import cv2
import easyocr
from ultralytics import SAM

# Window temporal: ambil deteksi dari frame ini +/- N frame tetangga
# sebelum digabung. TIDAK DIUBAH dari original.
TEMPORAL_WINDOW = 2


def detect_boxes(frame_path, reader):
    """Tahap 1: deteksi teks mentah di 1 frame, return list bbox
    [x1,y1,x2,y2] (sudah dikasih padding kecil). SAMA PERSIS original."""
    image_cv = cv2.imread(frame_path)
    h, w, _ = image_cv.shape
    # Parameter tambahan di bawah ini mengatur SENSITIVITAS text-detector
    # (CRAFT) di dalam EasyOCR -- bukan threshold prob (yang tetap 0.15,
    # tidak diubah). Nilai default EasyOCR (text_threshold=0.7,
    # low_text=0.4, link_threshold=0.4, mag_ratio=1, contrast_ths=0.1,
    # adjust_contrast=0.5) dirancang untuk teks dokumen/scan biasa, dan
    # terbukti (dari pengecekan visual hasil) GAGAL mendeteksi caption
    # bold-stroke-outline (mis. "mengalami") -- bukan gagal di tahap
    # confidence filter, tapi gagal di tahap detector-nya SAMA SEKALI
    # (box tidak pernah diajukan). Diturunkan/dinaikkan di sini supaya
    # detector lebih sensitif terhadap teks kontras-rendah/stroke-tebal:
    #   - text_threshold & low_text diturunkan -> makin longgar menandai
    #     suatu region sebagai kandidat teks.
    #   - mag_ratio dinaikkan -> frame di-upscale internal dulu sebelum
    #     dideteksi, membantu teks yang secara relatif kecil di frame.
    #   - contrast_ths dinaikkan & adjust_contrast dinaikkan -> lebih
    #     banyak region kontras-rendah (mis. watermark semi-transparan)
    #     kena penyesuaian kontras sebelum dideteksi.
    results_ocr = reader.readtext(
        frame_path,
        text_threshold=0.4,
        low_text=0.25,
        link_threshold=0.3,
        mag_ratio=1.5,
        contrast_ths=0.2,
        adjust_contrast=0.7,
    )
    boxes = []
    for (bbox, text, prob) in results_ocr:
        if prob > 0.15:
            pts = np.array(bbox, dtype=np.int32)
            x_min = int(np.min(pts[:, 0]))
            y_min = int(np.min(pts[:, 1]))
            x_max = int(np.max(pts[:, 0]))
            y_max = int(np.max(pts[:, 1]))
            h_pad = int((y_max - y_min) * 0.30)
            w_pad = int((x_max - x_min) * 0.15)
            boxes.append([
                max(0, x_min - w_pad),
                max(0, y_min - h_pad),
                min(w, x_max + w_pad),
                min(h, y_max + h_pad),
            ])
    return boxes, (h, w)


def merge_boxes(boxes, x_pad=5, y_pad=5):
    """Gabungkan box yang tumpang tindih/berdekatan jadi 1 box (union
    rectangle). SAMA PERSIS original."""
    if not boxes:
        return []
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ax1, ay1, ax2, ay2 = a[0] - x_pad, a[1] - y_pad, a[2] + x_pad, a[3] + y_pad
                bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
                if not (bx2 < ax1 or bx1 > ax2 or by2 < ay1 or by1 > ay2):
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(j)
                    merged = True
                    break
            if merged:
                break
    return boxes


def make_mask(frame_path, boxes, sam_model, h, w):
    """Tahap 2: dari box (hasil gabungan window temporal), jalankan SAM2
    di frame ini untuk dapat mask presisi. SAMA PERSIS original."""
    if not boxes:
        return np.zeros((h, w), dtype=np.uint8)

    combined_sam2_mask = np.zeros((h, w), dtype=np.uint8)
    results = sam_model(frame_path, bboxes=boxes, device="cpu", verbose=False)
    for result in results:
        if result.masks is not None:
            for mask_tensor in result.masks.data:
                mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                if mask_np.shape != (h, w):
                    mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                combined_sam2_mask |= mask_np

    combined_sam2_mask = (combined_sam2_mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.dilate(combined_sam2_mask, kernel, iterations=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="frames_in",
                         help="Folder frame chunk ini (core + overlap konteks kiri/kanan).")
    parser.add_argument("--outdir", default="lama_input",
                         help="Folder output pasangan image/mask (LaMa input).")
    parser.add_argument("--core-start-index", type=int, default=None,
                         help="Index 0-based (di frames-dir yang sudah di-sort) mulai frame "
                              "yang HARUS diproses & ditulis output-nya. Frame sebelum index "
                              "ini cuma dipakai sebagai konteks window temporal (overlap kiri).")
    parser.add_argument("--core-end-index", type=int, default=None,
                         help="Index 0-based, exclusive, akhir frame core. Frame mulai index "
                              "ini sampai akhir cuma dipakai sebagai konteks window temporal "
                              "(overlap kanan).")
    parser.add_argument("--sam2-weights", default=os.environ.get("SAM2_WEIGHTS_PATH", "sam2_l.pt"))
    parser.add_argument("--passthrough-dir", default=None,
                         help="[Optimisasi A] Kalau diisi: frame dengan mask KOSONG (tidak ada "
                              "subtitle terdeteksi di window) tidak ditulis ke --outdir (jadi "
                              "tidak diproses LaMa sama sekali), melainkan langsung dicopy ke "
                              "folder ini sebagai '{name}_mask.png' -- nama file yang sama "
                              "seperti output LaMa. Ini AMAN karena LaMa dengan mask kosong "
                              "secara matematis cuma menghasilkan balikan gambar aslinya "
                              "(output = original*(1-mask) + generated*mask, mask=0 di semua "
                              "pixel -> output = original). Kalau tidak diisi, perilaku SAMA "
                              "PERSIS seperti sebelumnya (semua frame ditulis ke --outdir).")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.passthrough_dir:
        os.makedirs(args.passthrough_dir, exist_ok=True)

    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    if not frame_paths:
        print(f"Error: tidak ada frame ditemukan di {args.frames_dir}")
        sys.exit(1)

    total = len(frame_paths)
    core_start = args.core_start_index if args.core_start_index is not None else 0
    core_end = args.core_end_index if args.core_end_index is not None else total
    print(f"Total frame di chunk ini (termasuk overlap konteks): {total}")
    print(f"Core range yang akan diproses & ditulis output: [{core_start}, {core_end})")

    print("Memuat model EasyOCR & SAM2 (sekali saja untuk chunk ini)...")
    reader = easyocr.Reader(['en', 'id'], gpu=False)
    sam_model = SAM(args.sam2_weights)

    # --- Tahap 1: OCR mentah di SEMUA frame chunk (termasuk overlap) ---
    # Overlap tetap di-OCR karena dipakai sebagai konteks window untuk
    # frame core di ujung chunk -- persis seperti original yang mengambil
    # window dari frame tetangga di seluruh video.
    print("\n[Tahap 1/2] Deteksi teks mentah di semua frame (termasuk overlap)...")
    t1 = time.time()
    all_boxes = []
    frame_shape = None
    for i, frame_path in enumerate(frame_paths, 1):
        boxes, shape = detect_boxes(frame_path, reader)
        all_boxes.append(boxes)
        frame_shape = shape
        if i % 25 == 0 or i == total:
            print(f"  OCR [{i}/{total}] selesai")
    print(f"Tahap 1 selesai dalam {time.time() - t1:.1f}s")

    # --- Tahap 2: gabungkan window temporal + SAM2, HANYA untuk core range ---
    print(f"\n[Tahap 2/2] SAM2 mask (window +/-{TEMPORAL_WINDOW} frame) untuk core range...")
    t2 = time.time()
    n_with_text = 0
    n_written = 0
    n_skipped_lama = 0
    core_total = max(0, core_end - core_start)
    for i in range(core_start, core_end):
        frame_path = frame_paths[i]
        t0 = time.time()
        lo = max(0, i - TEMPORAL_WINDOW)
        hi = min(total, i + TEMPORAL_WINDOW + 1)
        window_boxes = []
        for j in range(lo, hi):
            window_boxes.extend(all_boxes[j])
        merged = merge_boxes(window_boxes)

        name = os.path.splitext(os.path.basename(frame_path))[0]

        if not merged and args.passthrough_dir:
            # [Optimisasi A] Mask kosong -> skip LaMa sepenuhnya, langsung
            # copy frame asli sebagai hasil akhir. Matematis identik dengan
            # hasil kalau frame ini tetap dikirim ke LaMa dengan mask nol.
            image_cv = cv2.imread(frame_path)
            cv2.imwrite(os.path.join(args.passthrough_dir, f"{name}_mask.png"), image_cv)
            n_skipped_lama += 1
            n_written += 1
            elapsed = time.time() - t0
            if n_written % 10 == 0 or n_written == core_total:
                print(f"  [{n_written}/{core_total}] {os.path.basename(frame_path)} "
                      f"- 0 box - SKIP LaMa (passthrough) - {elapsed:.2f}s")
            continue

        h, w = frame_shape
        refined_mask = make_mask(frame_path, merged, sam_model, h, w)
        if merged:
            n_with_text += 1

        image_cv = cv2.imread(frame_path)
        cv2.imwrite(os.path.join(args.outdir, f"{name}.png"), image_cv)
        cv2.imwrite(os.path.join(args.outdir, f"{name}_mask.png"), refined_mask)
        n_written += 1

        elapsed = time.time() - t0
        if n_written % 10 == 0 or n_written == core_total:
            print(f"  [{n_written}/{core_total}] {os.path.basename(frame_path)} "
                  f"- {len(merged)} box (gabungan window) - {elapsed:.2f}s")

    total_elapsed = time.time() - t2
    avg = total_elapsed / max(1, core_total)
    print(f"\nTahap 2 selesai dalam {total_elapsed:.1f}s (rata-rata {avg:.2f}s/frame).")
    print(f"Frame dengan mask (setelah gabungan window): {n_with_text}/{core_total}")
    if args.passthrough_dir:
        print(f"Frame di-skip dari LaMa (mask kosong, passthrough): {n_skipped_lama}/{core_total}")
    print(f"\nTOTAL waktu deteksi+mask (tahap 1+2) chunk ini: {time.time() - t1:.1f}s")


if __name__ == "__main__":
    main()
