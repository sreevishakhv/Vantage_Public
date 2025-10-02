import argparse, json, re
from pathlib import Path
import numpy as np
import h5py
import imageio.v3 as iio
import cv2  # for resizing

def natural_key(s):
    # natural sort: "demo_2" < "demo_10"
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def find_frames(demo_dir: Path):
    """Prefer 1.png, 2.png, ... pattern; fall back to any *.png (excluding *_depth.*) with natural sort."""
    frames = []
    i = 1
    while (demo_dir / f"{i}.png").exists():
        frames.append(demo_dir / f"{i}.png")
        i += 1
    if frames:
        return frames
    frames = sorted(
        [p for p in demo_dir.glob("*.png") if not p.stem.endswith("_depth")],
        key=lambda p: natural_key(p.name)
    )
    return frames

def find_depth_for_frames(frames):
    """Require a 1:1 *_depth.png for every RGB frame (keeps order aligned)."""
    depth_paths = []
    for p in frames:
        dp = p.with_name(f"{p.stem}_depth.png")  # e.g., 1.png -> 1_depth.png
        if not dp.exists():
            return []  # need all depth frames present
        depth_paths.append(dp)
    return depth_paths

def pad_actions(arr: np.ndarray, pad_to: int) -> np.ndarray:
    T, A = arr.shape
    if A == pad_to:
        return arr.astype(np.float32, copy=False)
    if A > pad_to:
        return arr[:, :pad_to].astype(np.float32, copy=False)
    out = np.zeros((T, pad_to), dtype=np.float32)
    out[:, :A] = arr.astype(np.float32, copy=False)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", type=str, required=True, help="Root with subfolders per demo (e.g., imgs/demo_0/1.png)")
    ap.add_argument("--actions_json", type=str, required=True, help='JSON like {"demo_0":[[dx,dy,dz,grip],...], ...}')
    ap.add_argument("--out_dir", type=str, required=True, help="Where to write episode_*.hdf5")
    ap.add_argument("--camera_name", type=str, default="top", help="Name under observations/images/<camera_name>")
    ap.add_argument("--pad_action_to", type=int, default=14, help="Final action dimension (to match sample format)")
    ap.add_argument("--sim", type=str, default="False", help="Root attr 'sim': 'True' or 'False'")
    ap.add_argument("--max_episodes", type=int, default=None, help="Optional limit")
    args = ap.parse_args()

    images_root = Path(args.images_root)
    actions_json = Path(args.actions_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_flag = args.sim.lower() in ("1", "true", "yes", "y")

    with open(actions_json, "r") as f:
        actions_dict = json.load(f)

    demo_ids = sorted(list(actions_dict.keys()), key=natural_key)
    written = 0

    for epi_idx, demo_id in enumerate(demo_ids):
        if args.max_episodes is not None and written >= args.max_episodes:
            break

        demo_dir = images_root / demo_id
        if not demo_dir.exists():
            print(f"[WARN] {demo_dir} missing; skipping")
            continue

        frames = find_frames(demo_dir)
        if not frames:
            print(f"[WARN] no frames in {demo_dir}; skipping")
            continue

        acts = np.asarray(actions_dict[demo_id], dtype=np.float32)
        T = min(len(frames), len(acts))
        if T == 0:
            print(f"[WARN] zero-length episode {demo_id}; skipping")
            continue
        frames = frames[:T]
        acts = acts[:T]

        # Depth
        depth_paths = find_depth_for_frames(frames)
        has_depth = len(depth_paths) == T

        # Read RGB
        imgs = []
        H, W = None, None
        for p in frames:
            im = iio.imread(p)
            if im.ndim == 2:
                im = np.repeat(im[..., None], 3, axis=-1)
            if im.shape[-1] == 4:
                im = im[..., :3]
            if H is None:
                H, W = im.shape[0], im.shape[1]
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)  # ensure consistent size
            imgs.append(im.astype(np.uint8, copy=False))
        imgs = np.stack(imgs, axis=0)

        # Actions
        action = pad_actions(acts, args.pad_action_to)
        qpos = np.zeros_like(action, dtype=np.float32)
        # qpos = np.vstack([np.zeros((1, action.shape[1]), dtype=np.float32), action[:-1]])
        qvel = np.zeros_like(action, dtype=np.float32)

        out_path = out_dir / f"episode_{written}.hdf5"
        with h5py.File(out_path, "w") as h5:
            h5.attrs["sim"] = np.bool_(sim_flag)
            h5.attrs["orig_action_dim"] = acts.shape[1]

            obs = h5.create_group("observations")
            img_grp = obs.create_group("images")

            h5.create_dataset("action", data=action, compression="gzip", dtype="float32")
            obs.create_dataset("qpos", data=qpos, compression="gzip", dtype="float32")
            obs.create_dataset("qvel", data=qvel, compression="gzip", dtype="float32")
            img_grp.create_dataset(args.camera_name, data=imgs, compression="gzip", dtype="uint8")

        print(f"[OK] wrote {out_path})")
        written += 1

    print(f"Done. Episodes written: {written} -> {out_dir}")

if __name__ == "__main__":
    main()


#python convert_to_act_hdf5.py   --images_root "D1_data/clean/imgs"   --actions_json "D1_data/clean/actions.json"   --out_dir "D1_data/act/data"   --camera_name top   --pad_action_to 14  