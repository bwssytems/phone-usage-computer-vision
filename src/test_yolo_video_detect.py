#!/usr/bin/env python3
import argparse, os, time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---- defaults (you can override via CLI) ----
DEFAULT_MODEL = "runs/detect/train/weights/best.pt"
DEFAULT_VIDEO = "0"      # webcam
DEFAULT_THRESH = 0.25
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = "auto"  # "0" for first GPU, "cpu" to force CPU

# 3 fixed colors (if exactly 3 classes provided), else random palette
COLORS3 = [(0,255,0),(255,0,0),(0,0,255)]
PALETTE = np.random.randint(0, 255, size=(64, 3), dtype=np.uint8)

def load_labels_from_txt(path):
    # One label per line
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def load_filters(filter_path):
    if not filter_path:
        return None
    with open(filter_path) as f:
        allow = [int(x.strip()) for x in f if x.strip()]
    return set(allow)  # whitelist of class IDs

def open_video_capture(video_path):
    if str(video_path).isdigit():
        cap = cv2.VideoCapture(int(video_path))
    else:
        cap = cv2.VideoCapture(video_path)
    # Set a decent preview size; the model will letterbox/resize internally
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

def pick_color(class_id, n_names):
    if n_names == 3:
        return COLORS3[class_id % 3]
    return tuple(int(c) for c in PALETTE[class_id % len(PALETTE), :])

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("video", help="Path to video or webcam index (e.g. 0).", nargs="?", default=DEFAULT_VIDEO)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Path to YOLOv11 .pt weights")
    ap.add_argument("--labels", default=None, help="Optional TXT labels file to override model.names (one per line)")
    ap.add_argument("--filter", default=None, help="Optional whitelist of class IDs (one per line)")
    ap.add_argument("--videoout", default=None, help="Optional output MP4 path")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESH, help="Confidence threshold")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Inference image size")
    ap.add_argument("--device", default=DEFAULT_DEVICE, help='CUDA device like "0", "0,1", "cpu", or "auto"')
    ap.add_argument("--half", action="store_true", help="Use half-precision (FP16) if CUDA")

    ap.add_argument("--sync", choices=["video", "off"], default="video",
                    help="Synchronize to video FPS (files only). Use 'off' to run as fast as possible.")
    ap.add_argument("--max_fps", type=float, default=None,
                    help="Optional cap on playback FPS (files only).")
    ap.add_argument("--display-size", type=str, default=None,
                    help="Resize output display window, e.g. 1280x720. Inference still runs on original frames.")

    args = ap.parse_args()

    disp_wh = None
    if args.display_size:
        try:
            w, h = args.display_size.lower().split("x")
            disp_wh = (int(w), int(h))
        except Exception:
            print("Invalid --display-size format. Use WxH, e.g. 1280x720")
            disp_wh = None

    # Load model
    model = YOLO(args.model)

    # Device resolve
    if args.device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Half only valid on CUDA
    half = bool(args.half and torch.cuda.is_available() and ("cpu" not in str(device)))
    if half:
        try:
            model.to("cuda")
        except Exception:
            half = False  # fallback silently if something odd

    # Class names
    if args.labels:
        names = load_labels_from_txt(args.labels)
    else:
        # model.names is a dict {id: name} or a list depending on version
        names = [None] * len(model.names)
        for k, v in (model.names.items() if isinstance(model.names, dict) else enumerate(model.names)):
            if isinstance(model.names, dict):
                names[k] = v
            else:
                names[k] = v

    allow_ids = load_filters(args.filter)  # set or None

    # Video I/O
    cap = open_video_capture(args.video)
    if not cap.isOpened():
        print("ERROR: cannot open video:", args.video)
        return

    is_cam = str(args.video).isdigit()  # webcams run realtime; no throttling needed
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or np.isnan(src_fps) or src_fps <= 1:
        src_fps = 30.0  # sensible default when container doesn't report FPS

    target_fps = src_fps
    if args.max_fps:
        target_fps = min(target_fps, float(args.max_fps))

    frame_period = 1.0 / target_fps
    next_frame_time = time.perf_counter()  # scheduler baseline

    # Prepare writer if requested
    writer = None
    if args.videoout:
        out_path = os.path.abspath(args.videoout)
        print("Video output path:", out_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 1:
            fps = 30.0
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # FPS overlay settings
    row_size, text_x = 40, 760
    text_color, font_size, font_thickness = (0, 0, 255), 2, 2
    fps_avg_frame_count = 15
    counter, fps = 0, 0.0
    start_time = time.time()

    print("Starting inference… (press 'q' to quit)")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("End of stream.")
            break

        counter += 1

        # Inference on the current frame
        # stream=False returns a list; [0] is the only result for a single frame
        results = model.predict(
            frame,
            device=device,
            imgsz=args.imgsz,
            conf=args.threshold,
            half=half,
            verbose=False
        )
        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy  # (N, 4) on device
            conf = r.boxes.conf  # (N,)
            cls  = r.boxes.cls   # (N,)

            # Move to CPU numpy for drawing
            xyxy = xyxy.detach().cpu().numpy().astype(int)
            conf = conf.detach().cpu().numpy()
            cls  = cls.detach().cpu().numpy().astype(int)

            H, W = frame.shape[:2]
            for i, (x1,y1,x2,y2) in enumerate(xyxy):
                cid = int(cls[i])
                if allow_ids is not None and cid not in allow_ids:
                    continue
                score = float(conf[i])
                color = pick_color(cid, len(names))
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

                name = names[cid] if 0 <= cid < len(names) and names[cid] is not None else f"id{cid}"
                label = f"{name}: {score*100:.0f}%"
                y = max(y1 - 10, 10)
                cv2.putText(frame, label, (x1, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # FPS calculation & overlay
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()
        cv2.putText(frame, f"FPS = {fps:.1f}", (text_x, row_size),
                    cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)

        frame_to_show = frame
        if disp_wh is not None:
            frame_to_show = cv2.resize(frame, disp_wh, interpolation=cv2.INTER_LINEAR)

        # Show & optionally write
        cv2.imshow("YOLOv11 Detection", frame_to_show)
        if writer is not None:
            writer.write(frame_to_show)

        # --- Throttle to source FPS for files ---
        if args.sync == "video" and not is_cam:
            now = time.perf_counter()
            # schedule next frame relative to previous (smooth pacing)
            next_frame_time += frame_period
            sleep_time = next_frame_time - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If we're behind schedule (inference slower than target FPS),
                # snap next_frame_time to now to avoid growing lag.
                next_frame_time = now

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
