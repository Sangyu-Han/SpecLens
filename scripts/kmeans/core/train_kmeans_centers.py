#!/usr/bin/env python3
"""
Train K-means clustering on collected activation data using Faiss.

Standard approach (per Faiss maintainers):
  1. Load a random subset of samples (up to --max-samples, default 10M)
  2. Run Faiss GPU K-means on the subset
  3. Save centroids

No mini-batch logic needed — Faiss FAQ confirms there is no consistent
improvement beyond 20 iterations and ~k*256 training points.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import faiss
import numpy as np
import torch
import psutil

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FaissKMeansTrainer:
    """Train K-means clustering on activation data using Faiss."""

    def __init__(
        self,
        data_dir: str,
        n_clusters: int,
        n_init: int = 2,
        max_iter: int = 20,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.seed = seed

        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        self.file_format = self._detect_format()

    def _detect_format(self) -> str:
        pt_files = list(self.data_dir.glob("chunk_*.pt"))
        parquet_files = list(self.data_dir.glob("chunk_*.parquet"))

        if pt_files and parquet_files:
            raise ValueError(
                f"Found both .pt and .parquet files in {self.data_dir}. "
                "Please use a directory with only one format."
            )
        elif pt_files:
            logger.info(f"Detected PyTorch .pt format ({len(pt_files)} files)")
            return "pt"
        elif parquet_files:
            if pq is None:
                raise ImportError("PyArrow is required for Parquet format.")
            logger.info(f"Detected Parquet format ({len(parquet_files)} files)")
            return "parquet"
        else:
            raise ValueError(
                f"No chunk_*.pt or chunk_*.parquet files found in {self.data_dir}"
            )

    def _validate_checkpoint(self) -> Dict:
        checkpoint_path = self.data_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            checkpoint_path = self.data_dir.parent / "checkpoint.json"

        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint file not found in {self.data_dir} or parent")

        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)

        if not checkpoint.get("completed", False):
            raise ValueError(
                f"Data collection not completed in {self.data_dir}. "
                "Please complete data collection before training K-means."
            )

        return checkpoint

    def _load_pt(self, max_samples: int) -> Tuple[torch.Tensor, int]:
        """
        Load PyTorch .pt chunk files with pre-allocated tensor.

        Randomly shuffles chunks and loads up to max_samples.
        Uses pre-allocation to avoid 2x peak memory from torch.cat.
        """
        chunk_files = sorted(self.data_dir.glob("chunk_*.pt"))
        if not chunk_files:
            raise ValueError(f"No chunk files found in {self.data_dir}")

        logger.info(f"Found {len(chunk_files)} chunk files")

        # Randomly shuffle chunks
        rng = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(len(chunk_files), generator=rng).tolist()
        chunk_files = [chunk_files[i] for i in indices]

        # Load first chunk to get dimensions
        first_chunk = torch.load(chunk_files[0], map_location="cpu")
        act_size = first_chunk.shape[-1]
        avg_chunk_samples = first_chunk.shape[0]
        logger.info(f"Activation size: {act_size}, avg chunk: {avg_chunk_samples:,} samples")

        # Pre-allocate output tensor
        est_total = avg_chunk_samples * len(chunk_files)
        alloc_size = min(est_total, max_samples)
        logger.info(f"Pre-allocating tensor: [{alloc_size:,}, {act_size}] "
                     f"({alloc_size * act_size * 4 / 1e9:.1f} GB)")
        data = torch.empty(alloc_size, act_size)

        # Copy first chunk
        n = first_chunk.shape[0]
        data[:n] = first_chunk
        offset = n
        del first_chunk

        # Load remaining chunks
        for i, chunk_file in enumerate(chunk_files[1:], 1):
            if offset >= max_samples:
                logger.info(f"Reached sample limit of {max_samples:,}")
                break

            chunk_data = torch.load(chunk_file, map_location="cpu")

            if chunk_data.shape[-1] != act_size:
                raise ValueError(
                    f"Dimension mismatch in {chunk_file.name}: "
                    f"expected {act_size}, got {chunk_data.shape[-1]}"
                )

            n = chunk_data.shape[0]

            if offset + n > data.shape[0]:
                remaining = data.shape[0] - offset
                if remaining > 0:
                    data[offset:offset + remaining] = chunk_data[:remaining]
                    offset += remaining
                del chunk_data
                break

            data[offset:offset + n] = chunk_data
            offset += n
            del chunk_data

            if (i + 1) % 100 == 0:
                logger.info(f"  Loaded {i + 1}/{len(chunk_files)} chunks ({offset:,} samples)")

        data = data[:offset]
        logger.info(f"Loaded total: {data.shape[0]:,} samples, {act_size} dimensions")
        return data, act_size

    def _load_parquet(self, max_samples: int) -> Tuple[torch.Tensor, int]:
        """Load Parquet chunk files with random sampling."""
        chunk_files = sorted(self.data_dir.glob("chunk_*.parquet"))
        if not chunk_files:
            raise ValueError(f"No parquet chunk files found in {self.data_dir}")

        logger.info(f"Found {len(chunk_files)} parquet chunk files")

        np.random.seed(self.seed)
        chunk_files = list(np.random.permutation(chunk_files))

        # Detect format
        schema = pq.read_schema(chunk_files[0])
        if 'activations' not in schema.names:
            raise ValueError(f"Invalid schema: expected 'activations' column")

        activations_field = schema.field('activations')
        is_fixed_size_list = pa.types.is_fixed_size_list(activations_field.type)

        if is_fixed_size_list:
            act_size = activations_field.type.list_size
        else:
            first_table = pq.read_table(chunk_files[0]).slice(0, 1)
            act_size = len(first_table.column('activations')[0].as_py())

        # Count total rows
        file_row_counts = [pq.read_metadata(f).num_rows for f in chunk_files]
        total_rows = sum(file_row_counts)
        target = min(total_rows, max_samples)
        logger.info(f"Total rows: {total_rows:,}, loading: {target:,}")

        if target < total_rows:
            # Random sample across files
            all_indices = np.sort(np.random.choice(total_rows, size=target, replace=False))
            chunks = []
            cumsum = 0
            for file_idx, row_count in enumerate(file_row_counts):
                mask = (all_indices >= cumsum) & (all_indices < cumsum + row_count)
                local_indices = all_indices[mask] - cumsum
                if len(local_indices) > 0:
                    table = pq.read_table(chunk_files[file_idx])
                    sampled = table.take(local_indices)
                    col = sampled.column('activations')
                    arr = np.array([row.as_py() for row in col], dtype=np.float32)
                    chunks.append(torch.from_numpy(arr))
                cumsum += row_count
        else:
            chunks = []
            for chunk_file in chunk_files:
                table = pq.read_table(chunk_file)
                col = table.column('activations')
                arr = np.array([row.as_py() for row in col], dtype=np.float32)
                chunks.append(torch.from_numpy(arr))

        data = torch.cat(chunks, dim=0)
        logger.info(f"Loaded total: {data.shape[0]:,} samples, {act_size} dimensions")
        return data, act_size

    def _diversity_subsample(
        self,
        data_np: np.ndarray,
        target_samples: int,
        diversity_k: int = 256,
        diversity_iters: int = 10,
    ) -> np.ndarray:
        """
        Diversity-aware subsampling via rough k-means + balanced cluster sampling.

        1. Run a fast, rough k-means (small k, few iters) on the full data
        2. Sample equally from each cluster to get ``target_samples`` points
        3. Return the balanced subset

        This ensures rare activation patterns are represented proportionally
        in the final k-means training set.
        """
        n = data_np.shape[0]
        d = data_np.shape[1]
        if n <= target_samples:
            logger.info("[diversity] n=%d <= target=%d, skipping subsampling", n, target_samples)
            return data_np

        actual_k = min(diversity_k, n // 10)
        logger.info("[diversity] Running rough k-means: k=%d, niter=%d on %d samples",
                     actual_k, diversity_iters, n)

        t0 = time.time()
        rough_km = faiss.Kmeans(
            d=d, k=actual_k, niter=diversity_iters, nredo=1,
            verbose=False, seed=self.seed, gpu=True,
            max_points_per_centroid=max(256, n // actual_k + 1),
        )
        rough_km.train(data_np)

        # Assign all points to clusters
        _, labels = rough_km.index.search(data_np, 1)
        labels = labels.ravel()
        dt = time.time() - t0
        logger.info("[diversity] Rough k-means done in %.1fs", dt)

        # Balanced sampling: equal quota per cluster
        per_cluster = max(1, target_samples // actual_k)
        selected = []
        rng = np.random.RandomState(self.seed)

        cluster_sizes = []
        for c in range(actual_k):
            members = np.where(labels == c)[0]
            cluster_sizes.append(len(members))
            take = min(per_cluster, len(members))
            if take > 0:
                chosen = rng.choice(members, size=take, replace=False)
                selected.append(chosen)

        selected = np.concatenate(selected)

        # If we still need more points to reach target, fill from remaining
        if len(selected) < target_samples:
            remaining = np.setdiff1d(np.arange(n), selected)
            extra = min(target_samples - len(selected), len(remaining))
            if extra > 0:
                selected = np.concatenate([
                    selected, rng.choice(remaining, size=extra, replace=False)
                ])

        rng.shuffle(selected)
        selected = selected[:target_samples]

        # Log cluster distribution stats
        cluster_sizes = np.array(cluster_sizes)
        logger.info("[diversity] Cluster sizes: min=%d, max=%d, median=%d, std=%.0f",
                     cluster_sizes.min(), cluster_sizes.max(),
                     np.median(cluster_sizes), cluster_sizes.std())
        logger.info("[diversity] Selected %d samples (%.1f%% of %d), per_cluster=%d",
                     len(selected), 100.0 * len(selected) / n, n, per_cluster)

        return data_np[selected]

    def train(
        self,
        max_samples: int = 10_000_000,
        unit_norm: bool = False,
        diversity_sampling: bool = False,
        diversity_k: int = 256,
    ) -> Tuple[torch.Tensor, int]:
        """
        Train K-means on a random subset of the data.

        Steps:
          1. Calculate memory-safe sample limit
          2. Load random subset (shuffled chunks, up to limit)
          3. (optional) Center by global mean and unit-normalize
          4. (optional) Diversity-aware subsampling
          5. Run Faiss GPU K-means
          6. Return centroids

        Parameters
        ----------
        max_samples : int
            Cap on training samples.
        unit_norm : bool
            If True, compute global mean, center, and L2-normalize each
            vector before clustering.  The global mean is stored in
            ``self.global_mean`` after training.
        diversity_sampling : bool
            If True, apply diversity-aware subsampling after preprocessing.
            Uses rough k-means to identify clusters and balanced-samples
            from each cluster.  global_mean is computed from the FULL
            dataset BEFORE diversity subsampling.
        diversity_k : int
            Number of clusters for the rough k-means used in diversity
            subsampling.
        """
        logger.info("=" * 80)
        logger.info("K-means Training (Faiss GPU, subsample approach)")
        logger.info(f"  n_clusters={self.n_clusters}, n_init={self.n_init}, "
                     f"max_iter={self.max_iter}, max_samples={max_samples:,}")
        if diversity_sampling:
            logger.info(f"  diversity_sampling=True, diversity_k={diversity_k}")
        logger.info("=" * 80)

        self._validate_checkpoint()

        # Peek at first chunk to get act_size for memory calculation
        if self.file_format == "pt":
            first_file = next(self.data_dir.glob("chunk_*.pt"))
            first_chunk = torch.load(first_file, map_location="cpu")
            act_size = first_chunk.shape[-1]
            del first_chunk
        else:
            schema = pq.read_schema(next(self.data_dir.glob("chunk_*.parquet")))
            field = schema.field('activations')
            act_size = field.type.list_size if pa.types.is_fixed_size_list(field.type) else 256

        # Memory-aware sample limit (60% of available RAM)
        available_ram = psutil.virtual_memory().available
        bytes_per_sample = act_size * 4  # float32
        mem_limit = int(0.6 * available_ram / bytes_per_sample)
        actual_max = min(max_samples, mem_limit)

        logger.info(f"RAM available: {available_ram / 1e9:.1f} GB")
        logger.info(f"Sample limit: {actual_max:,} "
                     f"({actual_max * bytes_per_sample / 1e9:.1f} GB)")

        # Load random subset
        start_load = time.time()
        if self.file_format == "pt":
            data, act_size = self._load_pt(max_samples=actual_max)
        else:
            data, act_size = self._load_parquet(max_samples=actual_max)
        load_time = time.time() - start_load
        logger.info(f"Data loaded in {load_time:.1f}s")

        # Convert to float32 numpy
        data_np = data.float().cpu().numpy()
        del data
        torch.cuda.empty_cache()

        n_samples = data_np.shape[0]
        logger.info(f"Training data: {n_samples:,} samples × {act_size}D")

        # Optional: center by global mean and unit-normalize
        # IMPORTANT: global_mean is computed from the FULL loaded data
        # (before diversity subsampling) to represent the true population.
        self.global_mean = None
        if unit_norm:
            global_mean = data_np.mean(axis=0)
            self.global_mean = torch.from_numpy(global_mean.copy()).float()
            logger.info(f"Global mean norm: {np.linalg.norm(global_mean):.4f}")
            data_np -= global_mean
            norms = np.linalg.norm(data_np, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            data_np /= norms
            logger.info("Applied unit-norm preprocessing (center + L2 normalize)")

        # Optional: diversity-aware subsampling
        if diversity_sampling:
            # Target: keep enough for good k-means but balanced across clusters
            # Use 80% of loaded data as diversity target (balanced > random)
            diversity_target = min(n_samples, max(self.n_clusters * 256, int(n_samples * 0.8)))
            data_np = self._diversity_subsample(
                data_np, target_samples=diversity_target,
                diversity_k=diversity_k,
            )
            n_samples = data_np.shape[0]
            logger.info(f"After diversity subsampling: {n_samples:,} samples")

        # Prevent Faiss internal subsampling copy (would double memory)
        max_pts = max(256, (n_samples // self.n_clusters) + 1)

        # Train K-means
        logger.info("=" * 80)
        logger.info(f"Running Faiss K-means: k={self.n_clusters}, "
                     f"nredo={self.n_init}, niter={self.max_iter}")
        logger.info(f"max_points_per_centroid={max_pts}")
        logger.info("=" * 80)

        start_train = time.time()
        kmeans = faiss.Kmeans(
            d=act_size,
            k=self.n_clusters,
            niter=self.max_iter,
            nredo=self.n_init,
            verbose=True,
            seed=self.seed,
            gpu=True,
            max_points_per_centroid=max_pts,
        )
        kmeans.train(data_np)
        train_time = time.time() - start_train

        logger.info(f"Training completed in {train_time:.1f}s")

        centroids = torch.from_numpy(kmeans.centroids).float()
        logger.info(f"Centroids shape: {centroids.shape}")

        return centroids, act_size

    def save_centroids(
        self,
        centroids: torch.Tensor,
        act_size: int,
        output_path: str,
        format: str = "pt",
    ) -> None:
        """Save trained centroids to disk."""
        output_path = Path(output_path)

        if format == "parquet":
            if pq is None:
                raise ImportError("PyArrow is required for Parquet format.")

            centroids_np = centroids.cpu().numpy().astype(np.float32)
            list_type = pa.list_(pa.float32(), act_size)
            centroids_list = [row.tolist() for row in centroids_np]
            centroids_array = pa.array(centroids_list, type=list_type)
            table = pa.table({'centroid': centroids_array})
            metadata = {
                b'n_clusters': str(self.n_clusters).encode(),
                b'act_size': str(act_size).encode(),
                b'trained_at': str(time.time()).encode(),
                b'n_init': str(self.n_init).encode(),
                b'max_iter': str(self.max_iter).encode(),
                b'seed': str(self.seed).encode(),
            }
            table = table.replace_schema_metadata(metadata)
            pq.write_table(table, output_path, compression='zstd')
        else:
            save_dict = {
                "centroids": centroids,
                "n_clusters": self.n_clusters,
                "act_size": act_size,
                "trained_at": time.time(),
                "config": {
                    "n_init": self.n_init,
                    "max_iter": self.max_iter,
                    "seed": self.seed,
                },
            }
            # Save global mean if unit-norm preprocessing was used
            if self.global_mean is not None:
                save_dict["global_mean"] = self.global_mean
                logger.info("Saved global_mean in centroid checkpoint")
            torch.save(save_dict, output_path)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Centroids saved to {output_path} ({file_size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Train K-means clustering on activation data using Faiss",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing chunked activation data")
    parser.add_argument("--n-clusters", type=int, required=True,
                        help="Number of clusters")
    parser.add_argument("--n-init", type=int, default=2,
                        help="Number of K-means re-initializations")
    parser.add_argument("--max-iter", type=int, default=20,
                        help="Maximum iterations per run")
    parser.add_argument("--max-samples", type=int, default=10_000_000,
                        help="Maximum training samples to load (randomly sampled)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="centroids.pt",
                        help="Output filename (saved in data-dir)")
    parser.add_argument("--output-format", type=str, choices=["pt", "parquet"],
                        default="pt", help="Output format")
    parser.add_argument("--unit-norm", action="store_true",
                        help="Center by global mean and L2-normalize before clustering. "
                             "Saves global_mean in checkpoint for SAE b_dec init.")
    parser.add_argument("--diversity-sampling", action="store_true",
                        help="Enable diversity-aware subsampling via rough k-means "
                             "before final clustering. Ensures rare patterns are represented.")
    parser.add_argument("--diversity-k", type=int, default=256,
                        help="Number of clusters for the rough k-means in diversity subsampling")

    args = parser.parse_args()

    try:
        trainer = FaissKMeansTrainer(
            data_dir=args.data_dir,
            n_clusters=args.n_clusters,
            n_init=args.n_init,
            max_iter=args.max_iter,
            seed=args.seed,
        )

        centroids, act_size = trainer.train(
            max_samples=args.max_samples,
            unit_norm=args.unit_norm,
            diversity_sampling=args.diversity_sampling,
            diversity_k=args.diversity_k,
        )

        if os.path.isabs(args.output):
            output_path = args.output
        else:
            output_path = os.path.join(args.data_dir, args.output)

        output_format = args.output_format
        if output_format == "pt" and args.output.endswith(".parquet"):
            output_format = "parquet"
        elif output_format == "parquet" and args.output.endswith(".pt"):
            output_format = "pt"

        trainer.save_centroids(centroids, act_size, output_path, format=output_format)

        logger.info("=" * 80)
        logger.info("Done!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
