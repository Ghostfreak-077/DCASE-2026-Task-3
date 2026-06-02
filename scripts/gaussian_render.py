import json
import numpy as np
from pathlib import Path
from tqdm import tqdm


import resource

def set_memory_limit(gigabytes):
    # Convert gigabytes to bytes
    max_bytes = int(gigabytes * 1024 * 1024 * 1024)
    # Set a soft limit on Virtual Memory (RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, resource.RLIM_INFINITY))

# Limit Python to 2.8 GB so it errors out BEFORE the server's 3.1 GB limit
set_memory_limit(0.7)

def _render_gaussian(h, w, points_vals, sigma_az=12, sigma_el=8):
    out = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for xi, yi, v in points_vals:
        g = v * np.exp(-((xx - xi)**2 / (2*sigma_az**2) + (yy - yi)**2 / (2*sigma_el**2)))
        np.maximum(out, g, out=out)
    return out

def preprocess_dataset(input_json_path, output_json_path, img_h=180, img_w=360, threshold=0.01):
    """Convert triplet segmentations to Gaussian-rendered segmentations."""
    with open(input_json_path) as f:
        data = json.load(f)
    
    for ann in tqdm(data["annotations"], desc=f"  {input_json_path.name}", leave=False):
        _pts = []
        for sub in ann["segmentation"]:
            for triplet in sub:
                x, y, v = float(triplet[0]), float(triplet[1]), float(triplet[2])
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < img_w and 0 <= yi < img_h:
                    _pts.append((xi, yi, v))
        
        gaussian = _render_gaussian(img_h, img_w, _pts) if _pts else np.zeros((img_h, img_w), dtype=np.float32)
        
        # Convert back to [x, y, intensity] triplets where gaussian > threshold
        new_seg = []
        for yi in range(img_h):
            for xi in range(img_w):
                val = float(gaussian[yi, xi])
                if val > threshold:
                    new_seg.append([float(xi), float(yi), val])
        
        ann["segmentation"] = [new_seg]
    
    with open(output_json_path, 'w') as f:
        json.dump(data, f)
    print(f"✓ {output_json_path.name}")

# Usage
input_dir = Path("/teamspace/studios/this_studio/data/labels_dev")
output_dir = Path("/teamspace/studios/this_studio/labels_dev_gaussian")
output_dir.mkdir(exist_ok=True)

old_files = list(output_dir.glob("**/*.json"))

json_files = list(input_dir.glob("**/*.json"))
print(f"[Preprocessing] Found {len(json_files)} JSON files\n")

for json_file in tqdm(json_files, desc="Processing sequences"):
    if json_file in old_files:
        continue
    rel_path = json_file.relative_to(input_dir)
    out_file = output_dir / rel_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    preprocess_dataset(json_file, out_file)

print(f"\n[Done] All sequences saved to {output_dir}")