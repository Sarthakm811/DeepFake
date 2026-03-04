from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


IMAGE_SIZE = 224
VIDEO_SIZE = (224, 224)
VIDEO_FRAMES = 48
VIDEO_FPS = 12


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_image(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), image)


def _real_image(generator: np.random.Generator) -> np.ndarray:
    h, w = IMAGE_SIZE, IMAGE_SIZE
    base = np.zeros((h, w, 3), dtype=np.uint8)

    color1 = generator.integers(20, 120, size=3)
    color2 = generator.integers(130, 240, size=3)

    grad_x = np.linspace(0, 1, w, dtype=np.float32)
    grad = np.tile(grad_x, (h, 1))
    for ch in range(3):
        base[:, :, ch] = (color1[ch] * (1 - grad) + color2[ch] * grad).astype(np.uint8)

    for _ in range(4):
        center = (int(generator.integers(20, w - 20)), int(generator.integers(20, h - 20)))
        radius = int(generator.integers(10, 50))
        color = tuple(int(x) for x in generator.integers(40, 220, size=3))
        cv2.circle(base, center, radius, color, thickness=-1)

    blur_k = int(generator.choice([3, 5]))
    base = cv2.GaussianBlur(base, (blur_k, blur_k), 0)
    noise = generator.normal(0, 5, size=base.shape).astype(np.float32)
    out = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def _fake_image(generator: np.random.Generator) -> np.ndarray:
    h, w = IMAGE_SIZE, IMAGE_SIZE
    tile = int(generator.choice([8, 12, 16]))
    yy, xx = np.indices((h, w))
    checker = (((xx // tile) + (yy // tile)) % 2) * 255
    image = np.stack([checker, np.roll(checker, tile // 2, axis=0), np.roll(checker, tile // 2, axis=1)], axis=-1).astype(np.uint8)

    strong_noise = generator.normal(0, 35, size=image.shape).astype(np.float32)
    image = np.clip(image.astype(np.float32) + strong_noise, 0, 255).astype(np.uint8)

    if generator.random() < 0.7:
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)

    if generator.random() < 0.7:
        q = int(generator.integers(10, 35))
        ok, enc = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            image = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return image


def _real_video_frames(generator: np.random.Generator) -> list[np.ndarray]:
    w, h = VIDEO_SIZE
    frames: list[np.ndarray] = []
    bg = _real_image(generator)

    x = int(generator.integers(20, w - 20))
    y = int(generator.integers(20, h - 20))
    vx = int(generator.choice([-2, -1, 1, 2]))
    vy = int(generator.choice([-2, -1, 1, 2]))

    for _ in range(VIDEO_FRAMES):
        frame = bg.copy()
        cv2.circle(frame, (x, y), 18, (240, 240, 240), thickness=-1)
        cv2.circle(frame, (x, y), 10, (80, 120, 255), thickness=-1)

        x += vx
        y += vy
        if x < 20 or x > w - 20:
            vx *= -1
            x = max(20, min(w - 20, x))
        if y < 20 or y > h - 20:
            vy *= -1
            y = max(20, min(h - 20, y))

        frames.append(frame)

    return frames


def _fake_video_frames(generator: np.random.Generator) -> list[np.ndarray]:
    w, h = VIDEO_SIZE
    frames: list[np.ndarray] = []

    for i in range(VIDEO_FRAMES):
        frame = _fake_image(generator)

        if i % 6 == 0:
            shift = int(generator.integers(-25, 26))
            frame = np.roll(frame, shift, axis=1)

        if i % 8 == 0:
            band_h = int(generator.integers(10, 30))
            y0 = int(generator.integers(0, h - band_h))
            frame[y0:y0 + band_h, :, :] = generator.integers(0, 255, size=(band_h, w, 3), dtype=np.uint8)

        frames.append(frame)

    return frames


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, VIDEO_FPS, VIDEO_SIZE)
    for frame in frames:
        if frame.shape[:2] != (VIDEO_SIZE[1], VIDEO_SIZE[0]):
            frame = cv2.resize(frame, VIDEO_SIZE)
        writer.write(frame)
    writer.release()


def synthesize_images(root: Path, count_per_class: int, seed: int) -> None:
    real_dir = root / "Real"
    fake_dir = root / "Fake"
    _ensure_dir(real_dir)
    _ensure_dir(fake_dir)

    generator = _rng(seed)
    for idx in range(count_per_class):
        _save_image(real_dir / f"real_{idx:04d}.jpg", _real_image(generator))
        _save_image(fake_dir / f"fake_{idx:04d}.jpg", _fake_image(generator))


def synthesize_videos(root: Path, count_per_class: int, seed: int) -> None:
    real_dir = root / "Real"
    fake_dir = root / "Fake"
    _ensure_dir(real_dir)
    _ensure_dir(fake_dir)

    random.seed(seed)
    for idx in range(count_per_class):
        real_gen = _rng(seed + idx * 3 + 1)
        fake_gen = _rng(seed + idx * 3 + 2)

        _write_video(real_dir / f"real_{idx:04d}.mp4", _real_video_frames(real_gen))
        _write_video(fake_dir / f"fake_{idx:04d}.mp4", _fake_video_frames(fake_gen))


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize labeled image/video datasets for bias evaluation")
    parser.add_argument("--image-root", type=str, default="Test_Image_Synth", help="Relative folder for synthesized images")
    parser.add_argument("--video-root", type=str, default="Test_Video_Synth", help="Relative folder for synthesized videos")
    parser.add_argument("--image-count", type=int, default=120, help="Samples per class for images")
    parser.add_argument("--video-count", type=int, default=32, help="Samples per class for videos")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    image_root = PROJECT_ROOT / args.image_root
    video_root = PROJECT_ROOT / args.video_root

    synthesize_images(image_root, args.image_count, args.seed)
    synthesize_videos(video_root, args.video_count, args.seed)

    print("=== SYNTHETIC DATA GENERATION COMPLETE ===")
    print(f"Image dataset: {image_root} (per-class={args.image_count})")
    print(f"Video dataset: {video_root} (per-class={args.video_count})")


if __name__ == "__main__":
    main()
