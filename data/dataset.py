from collections import defaultdict
import glob
import hashlib
import json
import pickle
import sqlite3
from typing import OrderedDict

from torch.utils.data import Dataset
from utils.acoustic_features import wav_path_from_seq_dir
import os
import re
import torch
import numpy as np

def json_to_seq_name(json_path):
    stem = os.path.splitext(os.path.basename(json_path))[0]
    return re.sub(r"_std$", "", stem, flags=re.IGNORECASE)

def frame_path(seq_dir, seq_name, idx):
    return os.path.join(seq_dir, f"{seq_name}_{idx:04d}.png")

def scan_available_frames(seq_dir, seq_name):
    out = []
    for p in sorted(glob.glob(os.path.join(seq_dir, f"{seq_name}_*.png"))):
        m = re.search(r"_(\d{4})\.png$", os.path.basename(p))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)

def get_sequence_infos(split_keyword, labels_base, frames_base):
    infos = []
    if not os.path.isdir(labels_base) or not os.path.isdir(frames_base):
        return infos
    for split_dir in os.listdir(labels_base):
        # if split_keyword not in split_dir:
        #     continue
        split_path = os.path.join(labels_base, split_dir)
        if not os.path.isdir(split_path):
            continue
        for json_file in sorted(glob.glob(os.path.join(split_path, "*.json"))):
            seq_name = json_to_seq_name(json_file)
            seq_dir  = os.path.join(frames_base, split_dir, seq_name)
            infos.append((json_file, seq_dir, seq_name))
    return infos


_DB_SCHEMA_VERSION = "v5_unet_fixed"

def _dataset_cache_fingerprint(sequence_infos):
    h = hashlib.md5()
    h.update(_DB_SCHEMA_VERSION.encode())
    for json_path, seq_dir, seq_name in sorted(sequence_infos):
        mtime = str(os.path.getmtime(json_path)) if os.path.exists(json_path) else "missing"
        h.update(f"{json_path}:{mtime}:{seq_dir}:{seq_name}".encode())
    return h.hexdigest()[:16]


class EnergySegDataset(Dataset):
    CACHE_DIR = ".dataset_cache"

    def __init__(self, sequence_infos, frames_base, mic_base, acoustic_extractor,
                 frames_per_epoch=None, img_w=360, img_h=180, dist_norm=500.0,
                 cache_max_size=200, augmentor=None):
        self.frames_base        = frames_base
        self.mic_base           = mic_base
        self.acoustic_extractor = acoustic_extractor
        self.frames_per_epoch   = frames_per_epoch
        self.img_w              = img_w
        self.img_h              = img_h
        self.dist_norm          = dist_norm
        self.cache_max_size     = cache_max_size
        self.augmentor          = augmentor
        self.frame_cache        = OrderedDict()
        self.seq_map            = {}
        self.all_samples        = []
        self.current_indices    = []
        self.class_to_sample_indices = defaultdict(list)
        self.db_conn            = None

        os.makedirs(self.CACHE_DIR, exist_ok=True)
        fingerprint  = _dataset_cache_fingerprint(sequence_infos)
        self.db_path = os.path.join(self.CACHE_DIR, f"annotations_{fingerprint}.db")

        if os.path.exists(self.db_path):
            print(f"[Dataset] Cache hit  — loading index from {self.db_path}")
            self._load_index_from_db()
        else:
            print(f"[Dataset] Cache miss — building SQLite database from JSONs...")
            self._build_db(sequence_infos)

        print(f"[Dataset] frames_per_epoch={frames_per_epoch} → "
              f"{'subsampling' if frames_per_epoch else 'using all ' + str(len(self.all_samples))}")
        self.reset_epoch()

    def _get_db_conn(self):
        if self.db_conn is None:
            import pathlib
            db_uri       = pathlib.Path(self.db_path).absolute().as_uri()
            self.db_conn = sqlite3.connect(f"{db_uri}?mode=ro", uri=True)
        return self.db_conn

    def _build_db(self, sequence_infos):
        from tqdm import tqdm
        conn = sqlite3.connect(self.db_path)
        c    = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS annots (seq_id INTEGER, frame_idx INTEGER, data BLOB)")
        c.execute("CREATE TABLE IF NOT EXISTS seqs   (seq_id INTEGER, seq_dir TEXT, seq_name TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS frame_classes (sample_idx INTEGER, category_id INTEGER)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_lookup ON annots (seq_id, frame_idx)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_fc     ON frame_classes (category_id)")
        seq_id = 0
        for json_path, seq_dir, seq_name in tqdm(sequence_infos, desc="[Dataset] Building DB", leave=False):
            c.execute("INSERT INTO seqs VALUES (?, ?, ?)", (seq_id, seq_dir, seq_name))
            self.seq_map[seq_id] = (seq_dir, seq_name)
            with open(json_path) as f:
                data = json.load(f)
            by_frame = defaultdict(list)
            for ann in data.get("annotations", []):
                by_frame[int(ann["metadata_frame_index"])].append(ann)
            all_ids = sorted(set(by_frame.keys()) | set(scan_available_frames(seq_dir, seq_name)))
            for fi in all_ids:
                anns       = by_frame.get(fi, [])
                sample_idx = len(self.all_samples)
                self.all_samples.append((seq_id, fi))
                c.execute("INSERT INTO annots VALUES (?, ?, ?)", (seq_id, fi, pickle.dumps(anns)))
                for ann in anns:
                    cat_id = int(ann["category_id"])
                    c.execute("INSERT INTO frame_classes VALUES (?, ?)", (sample_idx, cat_id))
                    self.class_to_sample_indices[cat_id].append(sample_idx)
            seq_id += 1
        conn.commit(); conn.close()

    def _load_index_from_db(self):
        from tqdm import tqdm
        conn = sqlite3.connect(self.db_path)
        c    = conn.cursor()
        for row in c.execute("SELECT seq_id, seq_dir, seq_name FROM seqs"):
            self.seq_map[row[0]] = (row[1], row[2])
        c.execute("SELECT COUNT(*) FROM annots")
        total = c.fetchone()[0]
        c.execute("SELECT seq_id, frame_idx FROM annots")
        for row in tqdm(c, total=total, desc="[Dataset] Loading Index", leave=False):
            self.all_samples.append((row[0], row[1]))
        for row in c.execute("SELECT sample_idx, category_id FROM frame_classes"):
            self.class_to_sample_indices[row[1]].append(row[0])
        if self.class_to_sample_indices:
            n_pairs = sum(len(v) for v in self.class_to_sample_indices.values())
            print(f"[Dataset] {len(self.class_to_sample_indices)} classes, {n_pairs} pairs")
        conn.close()

    def clear_caches(self):
        self.frame_cache.clear()
        if self.acoustic_extractor:
            self.acoustic_extractor.clear_cache()

    def __del__(self):
        if self.db_conn:
            self.db_conn.close()

    def load_frame_tensor(self, seq_dir, seq_name, frame_idx):
        key = (seq_name, frame_idx)
        if key in self.frame_cache:
            t = self.frame_cache.pop(key)
            self.frame_cache[key] = t
            return t.clone()
        wav_path = wav_path_from_seq_dir(seq_dir, self.frames_base, self.mic_base)
        tensor   = self.acoustic_extractor.get_frame_bands(wav_path, frame_idx)
        if self.cache_max_size > 0:
            self.frame_cache[key] = tensor
            if len(self.frame_cache) > self.cache_max_size:
                self.frame_cache.popitem(last=False)
        return tensor.clone()

    def build_annotation_target(self, frame_annots):
        boxes, labels, bin_masks = [], [], []
        energy_maps_list, energy_masks_list = [], []
        distances, iids = [], []
        combined_energy = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        combined_mask   = np.zeros((self.img_h, self.img_w), dtype=bool)

        for ann in frame_annots:
            cat  = int(ann["category_id"]) + 1
            dist = float(ann["distance"])
            iid  = int(ann["instance_id"])
            energy_map  = np.zeros((self.img_h, self.img_w), dtype=np.float32)
            energy_mask = np.zeros((self.img_h, self.img_w), dtype=bool)
            xs, ys = [], []
            for sub in ann["segmentation"]:
                for triplet in sub:
                    x, y, v = float(triplet[0]), float(triplet[1]), float(triplet[2])
                    xi, yi  = int(round(x)), int(round(y))
                    if 0 <= xi < self.img_w and 0 <= yi < self.img_h:
                        energy_map[yi, xi]      = float(v)
                        energy_mask[yi, xi]     = True
                        combined_energy[yi, xi] = max(combined_energy[yi, xi], float(v))
                        # combined_mask[yi, xi]   = True
                        combined_mask = combined_energy > 0.01
                        xs.append(xi); ys.append(yi)
            if not xs:
                continue
            boxes.append([float(min(xs)), float(min(ys)), float(max(xs)+1), float(max(ys)+1)])
            labels.append(cat)
            bin_masks.append(energy_mask.copy())
            energy_maps_list.append(energy_map.copy())
            energy_masks_list.append(energy_mask.copy())
            distances.append(dist / self.dist_norm)
            iids.append(iid)

        if not boxes:
            N = 0
            return dict(
                boxes        = torch.zeros(N, 4,           dtype=torch.float32),
                labels       = torch.zeros(N,              dtype=torch.int64),
                masks        = torch.zeros(N, self.img_h, self.img_w, dtype=torch.bool),
                energy_maps  = torch.zeros(N, self.img_h, self.img_w, dtype=torch.float32),
                energy_masks = torch.zeros(N, self.img_h, self.img_w, dtype=torch.bool),
                vmap         = torch.zeros(self.img_h, self.img_w,    dtype=torch.float32),
                vmask        = torch.zeros(self.img_h, self.img_w,    dtype=torch.bool),
                distances    = torch.zeros(N,              dtype=torch.float32),
                instance_ids = torch.zeros(N,              dtype=torch.int64),
            )
        return dict(
            boxes        = torch.tensor(boxes,                       dtype=torch.float32),
            labels       = torch.tensor(labels,                      dtype=torch.int64),
            masks        = torch.tensor(np.stack(bin_masks),         dtype=torch.bool),
            energy_maps  = torch.tensor(np.stack(energy_maps_list),  dtype=torch.float32),
            energy_masks = torch.tensor(np.stack(energy_masks_list), dtype=torch.bool),
            vmap         = torch.tensor(combined_energy,             dtype=torch.float32),
            vmask        = torch.tensor(combined_mask,               dtype=torch.bool),
            distances    = torch.tensor(distances,                   dtype=torch.float32),
            instance_ids = torch.tensor(iids,                        dtype=torch.int64),
        )

    def reset_epoch(self, balanced=True):
        total = len(self.all_samples)
        n     = min(self.frames_per_epoch, total) if self.frames_per_epoch else total
        if not (balanced and self.class_to_sample_indices):
            idx = np.arange(total); np.random.shuffle(idx)
            self.current_indices = idx[:n]; return

        classes   = list(self.class_to_sample_indices.keys())
        min_quota = max(1, n // len(classes) // 2)
        balanced_part = []
        for cls in classes:
            pool  = self.class_to_sample_indices[cls]
            quota = max(int(len(pool) / total * n), min_quota)
            balanced_part.extend(np.random.choice(pool, quota, replace=(len(pool) < quota)))

        balanced_part   = np.array(balanced_part, dtype=np.int64)
        unique_balanced = np.unique(balanced_part)
        if len(balanced_part) >= n:
            np.random.shuffle(balanced_part)
            self.current_indices = balanced_part[:n]
        else:
            remaining = np.setdiff1d(np.arange(total), unique_balanced)
            n_rem     = n - len(balanced_part)
            if n_rem > 0 and len(remaining) > 0:
                extra = np.random.choice(remaining, n_rem, replace=(len(remaining) < n_rem))
                self.current_indices = np.random.permutation(np.concatenate([balanced_part, extra]))
            else:
                self.current_indices = np.random.permutation(balanced_part)

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx):
        real_idx          = self.current_indices[idx]
        seq_id, fi        = self.all_samples[real_idx]
        seq_dir, seq_name = self.seq_map[seq_id]
        c = self._get_db_conn().cursor()
        c.execute("SELECT data FROM annots WHERE seq_id=? AND frame_idx=?", (seq_id, fi))
        row  = c.fetchone()
        anns = pickle.loads(row[0]) if row and row[0] else []
        image  = self.load_frame_tensor(seq_dir, seq_name, fi)
        target = self.build_annotation_target(anns)
        if self.augmentor:
            image, target = self.augmentor(image, target)
        return image, target


def collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch]

def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    if info:
        ds = info.dataset
        if hasattr(ds, "frame_cache"):
            ds.frame_cache.clear()
