#!/usr/bin/env python3
"""PP-OCRv4 RKNN 板端测试: det -> (cls 可选) -> rec, DB 后处理 + CTC 解码.
用法: python3 test_ocr_board.py <image_path> [det.rknn rec.rknn]"""
import os
import sys
import time
import numpy as np
import cv2
from rknnlite.api import RKNNLite

HOME = os.path.expanduser("~/ocr_test")
DET_RKNN = os.path.join(HOME, "ppocrv4_det_960.rknn")
REC_RKNN = os.path.join(HOME, "ppocrv4_rec_320.rknn")
KEYS_TXT = os.path.join(HOME, "ppocr_keys_v1.txt")

# ---------- rec 字典 ----------
def load_charset():
    with open(KEYS_TXT, "r", encoding="utf-8") as f:
        chars = [c.rstrip("\n") for c in f.readlines()]
    # classes = blank(0) + 6623 keys + space(6624) = 6625
    return ["blank"] + chars + [" "]

# ---------- det: DB 后处理 ----------
def boxes_from_bitmap(bitmap, dest_width, dest_height, max_side=960):
    bitmap = np.squeeze(np.asarray(bitmap, dtype=np.float32)) > 0.3
    height, width = bitmap.shape
    outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)
    contours = outs[-2]
    boxes = []
    for contour in contours[:1000]:
        points = cv2.boxPoints(cv2.minAreaRect(contour))
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        # unclip: 简化 Clipper, 按周长扩 1.5
        area = cv2.contourArea(points)
        length = cv2.arcLength(points, True)
        if area < 10 or length == 0:
            continue
        offset = area * 1.5 / length
        pts = []
        for p in points:
            v0 = p - points[(np.where(points == p)[0][0] + 1) % len(points)]
            v1 = p - points[(np.where(points == p)[0][0] - 1) % len(points)]
            v0 = v0 / (np.linalg.norm(v0) + 1e-6)
            v1 = v1 / (np.linalg.norm(v1) + 1e-6)
            q = p + v0 * offset + v1 * offset
            pts.append(q)
        pts = np.array(pts)
        box = cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float32)
        # 缩放回原图坐标
        box[:, 0] = np.clip(box[:, 0] * dest_width / width, 0, dest_width - 1)
        box[:, 1] = np.clip(box[:, 1] * dest_height / height, 0, dest_height - 1)
        boxes.append(box.astype(np.int32))
    boxes = sorted(boxes, key=lambda b: (b[:, 1].min(), b[:, 0].min()))
    return boxes

def det_preprocess(img, max_side=960):
    h, w = img.shape[:2]
    ratio = max_side / max(h, w)  # 缩放到最长边 960
    nh, nw = int(round(h * ratio)), int(round(w * ratio))
    resized = cv2.resize(img, (nw, nh))
    # letterbox 补齐到固定 960x960 (模型 static_shape)
    canvas = np.zeros((max_side, max_side, 3), dtype=np.float32)
    canvas[:nh, :nw] = resized
    x = canvas.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = x.transpose(2, 0, 1)[np.newaxis]
    return x, ratio

# ---------- rec ----------
def rec_preprocess(crop, img_w=320, img_h=48):
    h, w = crop.shape[:2]
    ratio = img_w / w
    resized = cv2.resize(crop, (img_w, img_h)) if int(h * ratio) > img_h \
        else cv2.resize(crop, (img_w, int(h * ratio)))
    resized = cv2.copyMakeBorder(resized, 0, img_h - resized.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
    x = resized.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = x.transpose(2, 0, 1)[np.newaxis]
    return x

def ctc_decode(pred, charset):
    pred_idx = pred.argmax(axis=-1)[0]  # [T]
    text, last = "", -1
    for idx in pred_idx:
        if idx != last and idx != 0:
            text += charset[idx]
        last = idx
    return text

def main():
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print("ERROR: cannot read image", img_path)
        sys.exit(1)
    charset = load_charset()

    # --- det ---
    det = RKNNLite()
    det.load_rknn(DET_RKNN)
    det.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    x, ratio = det_preprocess(img)
    t0 = time.time()
    prob = det.inference(inputs=[x], data_format=["nchw"])
    det_ms = (time.time() - t0) * 1000
    prob = prob[0] if isinstance(prob, list) else prob
    boxes = boxes_from_bitmap(prob, 960, 960)
    # 检测框坐标: padded 空间 -> 原图空间
    ih, iw = img.shape[:2]
    adj = []
    for b in boxes:
        b = b.astype(np.float32)
        b[:, 0] = np.clip(b[:, 0] / ratio, 0, iw - 1)
        b[:, 1] = np.clip(b[:, 1] / ratio, 0, ih - 1)
        adj.append(b.astype(np.int32))
    boxes = adj
    print(f"det: {det_ms:.1f} ms, {len(boxes)} boxes (inference {x.shape})")

    # --- rec ---
    rec = RKNNLite()
    rec.load_rknn(REC_RKNN)
    rec.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    results = []
    for i, box in enumerate(boxes):
        pts = box.astype(np.float32)
        w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
        h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
        if w < 3 or h < 3:
            continue
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(pts, dst)
        crop = cv2.warpPerspective(img, M, (w, h)).astype(np.uint8)
        if h > w * 1.2:  # 竖字旋转
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        x_rec = rec_preprocess(crop)
        t1 = time.time()
        out = rec.inference(inputs=[x_rec], data_format=["nchw"])
        rec_ms = (time.time() - t1) * 1000
        out = out[0] if isinstance(out, list) else out
        text = ctc_decode(out, charset)
        results.append(text)
        print(f"  box{i:02d} rec {rec_ms:.1f} ms: {text}")

    print("\n===== OCR RESULT =====")
    for t in results:
        print(t)
    det.release(); rec.release()

if __name__ == "__main__":
    main()
