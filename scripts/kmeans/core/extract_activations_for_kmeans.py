#!/usr/bin/env python3
"""
Inference-based Activation Extraction Script for K-means Clustering Initialization

This script extracts activations from a trained model for use in K-means clustering
initialization of SAE dictionaries. It tracks progress by inferences (not tokens),
supports checkpoint-based resumption, and collects from multiple layers simultaneously.

Usage:
    python scripts/extract_activations_for_kmeans.py \
        --config configs/mask2former_sav_train.yaml \
        --output-dir outputs/kmeans_activations \
        --primary-layer "model.pixel_level_module.decoder.mask_projection" \
        --target-tokens-primary 10000000 \
        --layers "model.pixel_level_module.decoder.mask_projection" \
                 "model.transformer_module.decoder.layers.0@0" \
        --auto-probe \
        --format parquet

Features:
- Inference-based tracking (not token-based)
- Checkpoint-based resumption
- Per-layer buffer management with automatic flushing
- Subsample rate configuration per layer
- Atomic checkpoint writes
- Progress logging with detailed statistics
- Support for both PyTorch (.pt) and Parquet formats
- Parquet format uses FixedSizeList for efficient random access
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import yaml

# PyArrow imports with fallback
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    pa = None
    pq = None

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.sae.train.runner import SAETrainingPipeline

logger = logging.getLogger(__name__)


class InferenceBasedExtractor:
    """
    Extracts activations from a model, tracking progress by inference count.

    Features:
    - Tracks inferences, not tokens
    - Checkpoint-based resumption
    - Per-layer buffer management
    - Atomic checkpoint writes
    """

    def __init__(
        self,
        activation_store,
        output_dir: Path,
        primary_layer: str,
        target_tokens_primary: int,
        layers: Optional[List[str]] = None,
        subsample_rates: Optional[Dict[str, float]] = None,
        flush_every_tokens: int = 16384,
        rank: int = 0,
        world_size: int = 1,
        file_format: str = "pt",
    ):
        """
        Initialize the extractor.

        Args:
            activation_store: UniversalActivationStore instance
            output_dir: Directory to save activation chunks and checkpoints
            primary_layer: Reference layer for calculating target inference count
            target_tokens_primary: Target number of tokens for primary layer
            layers: List of layer names to extract (default: all expanded hook points)
            subsample_rates: Per-layer subsample rates (default: 1.0 for all)
            flush_every_tokens: Flush buffer when it reaches this size
            rank: DDP rank (default: 0)
            world_size: DDP world size (default: 1)
            file_format: Output file format - "pt" (PyTorch) or "parquet" (Arrow)
        """
        self.store = activation_store
        self.output_dir = Path(output_dir)
        self.primary_layer = primary_layer
        self.target_tokens_primary = target_tokens_primary
        self.flush_every_tokens = flush_every_tokens
        self.rank = rank
        self.world_size = world_size
        self.file_format = file_format

        # Validate file format
        if file_format not in ["pt", "parquet"]:
            raise ValueError(f"Invalid file format: {file_format}. Must be 'pt' or 'parquet'")
        if file_format == "parquet" and not PARQUET_AVAILABLE:
            raise ValueError("Parquet format requested but PyArrow is not installed. Install with: pip install pyarrow")

        # Global file I/O owner (for checkpoint saves)
        self.is_owner = (rank == 0)
        # Layer ownership map (for activation extraction - will be set from activation_store.layer_owners)
        self.layer_owners = getattr(self.store, "layer_owners", {})

        # Layer configuration
        self.layers = layers if layers is not None else self.store.expanded_hook_points
        self.subsample_rates = subsample_rates or {}

        # State tracking
        self.inferences_completed = 0
        self.target_inferences = 0  # Will be calculated
        self.completed = False

        # Per-layer state
        self.layer_state: Dict[str, Dict] = {}
        self.layer_buffers: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.layer_buffer_sizes: Dict[str, int] = defaultdict(int)

        # Setup output directories (all ranks need to create directories for their chunks)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for layer_name in self.layers:
            layer_dir = self._layer_dir(layer_name)
            layer_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.output_dir / "checkpoint.json"

        if self.is_owner:
            logger.info(f"InferenceBasedExtractor initialized:")
            logger.info(f"  Rank: {self.rank}/{self.world_size}")
            logger.info(f"  Output directory: {self.output_dir}")
            logger.info(f"  Primary layer: {self.primary_layer}")
            logger.info(f"  Target tokens (primary): {target_tokens_primary:,}")
            logger.info(f"  Layers to extract: {len(self.layers)}")
            logger.info(f"  File format: {self.file_format}")

    def _layer_dir(self, layer_name: str) -> Path:
        """Get directory for a layer's activation chunks."""
        safe_name = layer_name.replace("/", "_").replace(":", "__")
        return self.output_dir / safe_name

    def _chunk_path(self, layer_name: str, chunk_idx: int) -> Path:
        """Get path for a specific chunk file."""
        extension = ".parquet" if self.file_format == "parquet" else ".pt"
        return self._layer_dir(layer_name) / f"chunk_{chunk_idx:06d}{extension}"

    def _write_parquet_chunk(self, data: torch.Tensor, output_path: Path):
        """
        Write chunk to Parquet with FixedSizeList format for efficient random access.

        Args:
            data: Tensor of shape (n_samples, n_dims) to write
            output_path: Path to write the Parquet file
        """
        # Convert to numpy float32 for efficiency
        data_np = data.numpy().astype(np.float32)
        n_samples, n_dims = data_np.shape

        # Use FixedSizeList for efficient single-column format
        # This is much more efficient than creating 768 separate columns
        fixed_size_list_type = pa.list_(pa.float32(), n_dims)

        # Flatten the array to 1D, then create FixedSizeListArray
        # This is more efficient than creating individual arrays for each row
        flat_data = data_np.flatten()
        values_array = pa.array(flat_data, type=pa.float32())

        # Create FixedSizeListArray from flat values
        activations_array = pa.FixedSizeListArray.from_arrays(values_array, n_dims)

        # Create table with single 'activations' column
        table = pa.table({'activations': activations_array})

        # Write with row group size for efficient random sampling
        # Smaller row groups (10000) enable better random access patterns
        pq.write_table(
            table,
            output_path,
            compression='snappy',  # Fast compression with good ratio
            row_group_size=10000,  # Enables efficient random row access
        )

        logger.debug(
            f"[Rank {self.rank}] Wrote Parquet chunk: {output_path.name}, "
            f"{n_samples:,} samples × {n_dims} dims, "
            f"size={output_path.stat().st_size / 1024 / 1024:.2f} MB"
        )

    def initialize_from_checkpoint(self) -> bool:
        """
        Load state from checkpoint if it exists.

        Returns:
            True if checkpoint was loaded, False otherwise
        """
        if not self.checkpoint_path.exists():
            logger.info("No checkpoint found, starting fresh")
            return False

        try:
            with open(self.checkpoint_path, "r") as f:
                ckpt = json.load(f)

            # Validate checkpoint
            if ckpt.get("primary_layer") != self.primary_layer:
                logger.warning(
                    f"Checkpoint primary layer mismatch: "
                    f"{ckpt.get('primary_layer')} != {self.primary_layer}"
                )
                return False

            # Validate file format (allow resumption with same format)
            ckpt_format = ckpt.get("file_format", "pt")  # Default to "pt" for old checkpoints
            if ckpt_format != self.file_format:
                logger.warning(
                    f"Checkpoint file format mismatch: "
                    f"{ckpt_format} != {self.file_format}. "
                    f"Cannot resume with different format."
                )
                return False

            # Restore state
            self.inferences_completed = ckpt.get("inferences_completed", 0)
            self.target_inferences = ckpt.get("target_inferences", 0)
            self.completed = ckpt.get("completed", False)

            # Restore per-layer state
            layers_data = ckpt.get("layers", {})
            for layer_name in self.layers:
                if layer_name in layers_data:
                    self.layer_state[layer_name] = layers_data[layer_name]
                else:
                    logger.warning(f"Layer {layer_name} not found in checkpoint")

            logger.info(f"Checkpoint loaded:")
            logger.info(f"  Inferences completed: {self.inferences_completed:,}")
            logger.info(f"  Target inferences: {self.target_inferences:,}")
            logger.info(f"  Progress: {self.inferences_completed/max(1, self.target_inferences)*100:.1f}%")
            logger.info(f"  Completed: {self.completed}")

            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def save_checkpoint(self):
        """Save current state to checkpoint (atomic write, owner rank only)."""
        if not self.is_owner:
            return  # Only owner rank saves checkpoint

        # Use local counts only (no aggregation to avoid NCCL deadlock)
        # Aggregation is expensive and not needed for checkpointing
        ckpt = {
            "primary_layer": self.primary_layer,
            "target_inferences": self.target_inferences,
            "target_tokens_primary": self.target_tokens_primary,
            "inferences_completed": self.inferences_completed,
            "completed": self.completed,
            "file_format": self.file_format,
            "layers": {},
        }

        # Save per-layer state with local counts (no aggregation)
        for layer_name in self.layers:
            state = self.layer_state.get(layer_name, {})
            ckpt["layers"][layer_name] = {
                "act_size": state.get("act_size", -1),
                "tokens_per_inference": state.get("tokens_per_inference", -1),
                "subsample_rate": self.subsample_rates.get(layer_name, 1.0),
                "tokens_collected": state.get("tokens_collected", 0),
                "chunks_written": state.get("chunks_written", 0),
            }

        # Atomic write: write to temp file, then rename
        tmp_path = self.checkpoint_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(ckpt, f, indent=2)
            tmp_path.rename(self.checkpoint_path)
            logger.debug(f"[Rank {self.rank}] Checkpoint saved: {self.checkpoint_path}")
        except Exception as e:
            logger.error(f"[Rank {self.rank}] Failed to save checkpoint: {e}")
            if tmp_path.exists():
                tmp_path.unlink()

    def _aggregate_token_counts(self, safe_mode: bool = True) -> Dict[str, Dict[str, int]]:
        """
        Aggregate token counts across all ranks for logging.

        Args:
            safe_mode: If True, wrap all_reduce in try-catch to prevent deadlocks

        Returns:
            Dict mapping layer names to aggregated stats:
                {layer_name: {"tokens_collected": X, "chunks_written": Y, "buffered": Z}}
        """
        if self.world_size == 1:
            # Single GPU: no aggregation needed
            result = {}
            for layer_name in self.layers:
                state = self.layer_state.get(layer_name, {})
                buffered = self.layer_buffer_sizes.get(layer_name, 0)
                result[layer_name] = {
                    "tokens_collected": state.get("tokens_collected", 0),
                    "chunks_written": state.get("chunks_written", 0),
                    "buffered": buffered,
                }
            return result

        # Multi-GPU: aggregate across ranks
        import torch.distributed as dist

        if not dist.is_initialized():
            # DDP not initialized, fallback to local counts
            logger.warning("[_aggregate_token_counts] DDP not initialized, using local counts")
            result = {}
            for layer_name in self.layers:
                state = self.layer_state.get(layer_name, {})
                buffered = self.layer_buffer_sizes.get(layer_name, 0)
                result[layer_name] = {
                    "tokens_collected": state.get("tokens_collected", 0),
                    "chunks_written": state.get("chunks_written", 0),
                    "buffered": buffered,
                }
            return result

        # Gather counts from all ranks with error handling
        aggregated = {}
        try:
            for layer_name in self.layers:
                state = self.layer_state.get(layer_name, {})
                buffered = self.layer_buffer_sizes.get(layer_name, 0)

                # Create tensors for all_reduce
                tokens_collected = torch.tensor(
                    state.get("tokens_collected", 0), dtype=torch.long, device="cuda"
                )
                chunks_written = torch.tensor(
                    state.get("chunks_written", 0), dtype=torch.long, device="cuda"
                )
                buffered_tokens = torch.tensor(
                    buffered, dtype=torch.long, device="cuda"
                )

                # All-reduce to sum across all ranks (with timeout handling in safe_mode)
                if safe_mode:
                    try:
                        dist.all_reduce(tokens_collected, op=dist.ReduceOp.SUM)
                        dist.all_reduce(chunks_written, op=dist.ReduceOp.SUM)
                        dist.all_reduce(buffered_tokens, op=dist.ReduceOp.SUM)
                    except Exception as e:
                        logger.warning(f"[Rank {self.rank}] all_reduce failed for {layer_name}: {e}, using local counts")
                        # Fall back to local counts if all_reduce fails
                        aggregated[layer_name] = {
                            "tokens_collected": state.get("tokens_collected", 0),
                            "chunks_written": state.get("chunks_written", 0),
                            "buffered": buffered,
                        }
                        continue
                else:
                    dist.all_reduce(tokens_collected, op=dist.ReduceOp.SUM)
                    dist.all_reduce(chunks_written, op=dist.ReduceOp.SUM)
                    dist.all_reduce(buffered_tokens, op=dist.ReduceOp.SUM)

                aggregated[layer_name] = {
                    "tokens_collected": tokens_collected.item(),
                    "chunks_written": chunks_written.item(),
                    "buffered": buffered_tokens.item(),
                }

        except Exception as e:
            logger.error(f"[Rank {self.rank}] _aggregate_token_counts failed: {e}, falling back to local counts")
            # Return local counts on any error
            result = {}
            for layer_name in self.layers:
                state = self.layer_state.get(layer_name, {})
                buffered = self.layer_buffer_sizes.get(layer_name, 0)
                result[layer_name] = {
                    "tokens_collected": state.get("tokens_collected", 0),
                    "chunks_written": state.get("chunks_written", 0),
                    "buffered": buffered,
                }
            return result

        return aggregated

    def _flush_layer_buffer(self, layer_name: str):
        """Flush buffer for a layer to disk (all ranks save their data)."""
        if not self.layer_buffers[layer_name]:
            return

        # Concatenate all buffered tensors
        buffer = torch.cat(self.layer_buffers[layer_name], dim=0)
        tokens_in_chunk = buffer.shape[0]

        # ALL ranks save their data (no more "if self.is_owner" check)
        # Get local chunk index (how many chunks this rank has written)
        state = self.layer_state.get(layer_name, {})
        local_chunk_idx = state.get("chunks_written", 0)

        # Calculate global chunk index using interleaved numbering:
        # Rank 0 writes: 0, 2, 4, 6, ...
        # Rank 1 writes: 1, 3, 5, 7, ...
        # Formula: global_chunk_idx = local_chunk_idx * world_size + rank
        global_chunk_idx = local_chunk_idx * self.world_size + self.rank
        chunk_path = self._chunk_path(layer_name, global_chunk_idx)

        # Save to disk in the appropriate format
        if self.file_format == "parquet":
            self._write_parquet_chunk(buffer.cpu(), chunk_path)
        else:  # pt format
            torch.save(buffer.cpu(), chunk_path)

        # Update state
        if layer_name not in self.layer_state:
            self.layer_state[layer_name] = {}
        self.layer_state[layer_name]["chunks_written"] = local_chunk_idx + 1
        self.layer_state[layer_name]["tokens_collected"] = (
            state.get("tokens_collected", 0) + tokens_in_chunk
        )

        logger.debug(
            f"[Rank {self.rank}] Flushed {layer_name}: local_chunk={local_chunk_idx}, "
            f"global_chunk={global_chunk_idx}, {tokens_in_chunk:,} tokens, "
            f"format={self.file_format}, "
            f"total: {self.layer_state[layer_name]['tokens_collected']:,}"
        )

        # Clear buffer on all ranks
        self.layer_buffers[layer_name].clear()
        self.layer_buffer_sizes[layer_name] = 0

    def calculate_target_inferences(
        self, tokens_per_inference: Dict[str, float]
    ) -> int:
        """
        Calculate target inference count based on primary layer.

        Args:
            tokens_per_inference: Dict mapping layer names to average tokens per inference

        Returns:
            Target number of inferences needed
        """
        if self.primary_layer not in tokens_per_inference:
            raise ValueError(
                f"Primary layer '{self.primary_layer}' not found in tokens_per_inference"
            )

        tpi = tokens_per_inference[self.primary_layer]
        subsample = self.subsample_rates.get(self.primary_layer, 1.0)

        # Account for subsampling
        effective_tpi = tpi * subsample

        # Prevent division by zero/very small numbers
        if effective_tpi < 0.001:
            logger.error(
                f"Effective tokens per inference is too small: {effective_tpi:.6f}. "
                f"This likely means probe failed to collect tokens for primary layer."
            )
            raise ValueError(f"Invalid effective_tpi: {effective_tpi}")

        target = int(self.target_tokens_primary / effective_tpi)

        logger.info(f"Target inference calculation:")
        logger.info(f"  Primary layer: {self.primary_layer}")
        logger.info(f"  Tokens per inference: {tpi:.1f}")
        logger.info(f"  Subsample rate: {subsample}")
        logger.info(f"  Effective tokens/inference: {effective_tpi:.1f}")
        logger.info(f"  Target inferences: {target:,}")

        return target

    def configure_subsample_rates(self):
        """Configure subsample rates in the activation store."""
        if not self.subsample_rates:
            logger.info("No subsample rates to configure")
            return

        # Temporarily set subsample rates in per_layer config
        for layer_name, rate in self.subsample_rates.items():
            if layer_name not in self.store.per_layer:
                self.store.per_layer[layer_name] = {}
            self.store.per_layer[layer_name]["random_subsample_rate"] = rate

        logger.info(f"Configured subsample rates for {len(self.subsample_rates)} layers")

    def collect_inference(self) -> Dict[str, int]:
        """
        Collect activations from one inference batch.

        Follows the exact pattern from runner.py training loop:
        1. All ranks call collect_round() to trigger inference
        2. All ranks call next_batch() - owner gets data, non-owner gets None
        3. Only owner rank saves to disk

        Returns:
            Dict mapping layer names to number of tokens collected
        """
        # All ranks participate in collection (triggers inference on all GPUs)
        self.store.collect_round(n_batches=1)

        tokens_collected = {}

        # All ranks call next_batch, but only owner rank gets actual data
        for layer_name in self.layers:
            # Check if this rank is the owner of this layer
            layer_owner = self.layer_owners.get(layer_name, 0)
            is_layer_owner = (self.rank == layer_owner)

            # Check queue size and request exactly what's available (only on owner rank)
            batch_size = 4096  # Default
            if is_layer_owner and hasattr(self.store, 'queues') and layer_name in self.store.queues:
                queue = self.store.queues[layer_name]
                ntoks = queue.ntoks if hasattr(queue, 'ntoks') else 0
                logger.info(f"[Rank {self.rank}] Queue {layer_name}: {ntoks} tokens before next_batch()")
                # Request exactly what's available (or default if empty)
                if ntoks > 0:
                    batch_size = ntoks

            # Get available tokens (owner rank gets data, non-owner gets None)
            # pop_batch() returns None if ntoks < batch_size, so request exactly what's available
            batch = self.store.next_batch(layer_name, batch_size=batch_size)

            # Debug: Log what we got (only on owner rank)
            if is_layer_owner:
                if batch is not None:
                    logger.info(f"[Rank {self.rank}] next_batch returned: shape={batch.shape}, numel={batch.numel()}")
                else:
                    logger.info(f"[Rank {self.rank}] next_batch returned: None")

            if batch is not None and batch.numel() > 0:
                # Only owner rank reaches here (non-owner gets None)
                logger.info(f"[Rank {self.rank}] Extracted {batch.shape[0]} tokens from {layer_name}")

                # Add to buffer
                self.layer_buffers[layer_name].append(batch.cpu())
                self.layer_buffer_sizes[layer_name] += batch.shape[0]
                tokens_collected[layer_name] = batch.shape[0]

                # Initialize state if needed
                if layer_name not in self.layer_state:
                    self.layer_state[layer_name] = {
                        "act_size": batch.shape[-1],
                        "tokens_per_inference": -1,  # Will be set during probing
                        "tokens_collected": 0,
                        "chunks_written": 0,
                    }

                # Flush if buffer is full
                if self.layer_buffer_sizes[layer_name] >= self.flush_every_tokens:
                    self._flush_layer_buffer(layer_name)
            else:
                # Non-owner ranks reach here (or owner with empty queue)
                if is_layer_owner:
                    logger.warning(f"[Rank {self.rank}] No tokens extracted from {layer_name}")
                tokens_collected[layer_name] = 0

        return tokens_collected

    def extract(self, save_checkpoint_every: int = 100):
        """
        Main extraction loop.

        In DDP mode, all ranks participate in collect_round() but only owner rank
        extracts and saves tokens.

        Args:
            save_checkpoint_every: Save checkpoint every N inferences
        """
        if self.completed:
            if self.is_owner:
                logger.info("Extraction already completed")
            return

        if self.is_owner:
            logger.info(f"Starting extraction from inference {self.inferences_completed:,}")

        start_time = time.time()
        last_log_time = start_time
        log_interval = 30.0  # Log every 30 seconds (reduced from 10s to minimize NCCL overhead)
        last_gc_time = start_time
        gc_interval = 300.0  # Run garbage collection every 5 minutes

        while self.inferences_completed < self.target_inferences:
            # Collect one inference (all ranks participate)
            tokens_collected = self.collect_inference()
            # One collect_round(n_batches=1) = one model batch
            self.inferences_completed += 1

            # Periodic garbage collection to prevent memory leaks
            current_time = time.time()
            if current_time - last_gc_time >= gc_interval:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                last_gc_time = current_time
                if self.is_owner:
                    logger.debug(f"[Rank {self.rank}] Ran garbage collection at inference {self.inferences_completed}")

            # Periodic logging (less frequent to reduce NCCL overhead)
            if current_time - last_log_time >= log_interval:
                # Aggregate token counts across all ranks (with safe mode to handle errors)
                aggregated_stats = self._aggregate_token_counts(safe_mode=True)

                # Only rank 0 logs, but with aggregated values
                if self.is_owner:
                    elapsed = current_time - start_time
                    progress = self.inferences_completed / max(1, self.target_inferences)
                    inferences_per_sec = self.inferences_completed / max(1.0, elapsed)
                    eta_sec = (self.target_inferences - self.inferences_completed) / max(1e-6, inferences_per_sec)

                    # Calculate samples processed (inferences × batch_size)
                    batch_size = getattr(self.store, "model_batch_size", 1)
                    samples_processed = self.inferences_completed * batch_size
                    target_samples = self.target_inferences * batch_size

                    logger.info(
                        f"Progress: {self.inferences_completed:,}/{self.target_inferences:,} inferences "
                        f"({samples_processed:,}/{target_samples:,} samples) "
                        f"({progress*100:.1f}%) | "
                        f"Speed: {inferences_per_sec:.1f} inf/s | "
                        f"ETA: {eta_sec/60:.1f} min"
                    )

                    # Log per-layer stats with aggregated counts
                    for layer_name in self.layers:
                        stats = aggregated_stats.get(layer_name, {})
                        total = stats.get("tokens_collected", 0) + stats.get("buffered", 0)
                        logger.debug(
                            f"  {layer_name}: {total:,} tokens (aggregated across {self.world_size} ranks) "
                            f"({stats.get('chunks_written', 0)} chunks, {stats.get('buffered', 0):,} buffered)"
                        )

                last_log_time = current_time

            # Periodic checkpoint save
            if self.inferences_completed % save_checkpoint_every == 0:
                self.save_checkpoint()

        # Final flush and checkpoint
        if self.is_owner:
            logger.info("Extraction complete, flushing remaining buffers...")
        for layer_name in self.layers:
            if self.layer_buffers[layer_name]:
                self._flush_layer_buffer(layer_name)

        self.completed = True
        self.save_checkpoint()

        # Final statistics (aggregated across all ranks with safe mode)
        total_time = time.time() - start_time
        aggregated_final_stats = self._aggregate_token_counts(safe_mode=True)

        # Only rank 0 logs final summary
        if self.is_owner:
            logger.info(f"Extraction completed in {total_time/60:.1f} minutes")
            logger.info(f"Average speed: {self.inferences_completed/(total_time+1e-6):.1f} inferences/sec")
            logger.info(f"Per-layer summary (aggregated across {self.world_size} ranks):")
            for layer_name in self.layers:
                state = self.layer_state.get(layer_name, {})
                stats = aggregated_final_stats.get(layer_name, {})
                logger.info(
                    f"  {layer_name}: "
                    f"{stats.get('tokens_collected', 0):,} tokens, "
                    f"{stats.get('chunks_written', 0)} chunks, "
                    f"act_size={state.get('act_size', -1)}"
                )


def probe_tokens_per_inference(
    activation_store, layers: List[str], num_probe_batches: int = 10, rank: int = 0, world_size: int = 1
) -> Dict[str, float]:
    """
    Probe the activation store to measure tokens per inference for each layer.

    Each rank probes its own layers (the ones it owns), then results are gathered.

    Args:
        activation_store: UniversalActivationStore instance
        layers: List of layer names to probe
        num_probe_batches: Number of inference batches to probe
        rank: Current rank
        world_size: Total number of ranks

    Returns:
        Dict mapping layer names to average tokens per inference
    """
    # Get layer owners
    layer_owners = getattr(activation_store, "layer_owners", {})

    # Find which layers this rank owns
    my_layers = [l for l in layers if layer_owners.get(l, 0) == rank]

    if my_layers:
        logger.info(f"[Rank {rank}] Probing tokens per inference for {len(my_layers)} layers with {num_probe_batches} batches...")

    tokens_per_layer = defaultdict(list)

    # All ranks participate in collect_round (for inference)
    for i in range(num_probe_batches):
        # Collect one inference
        activation_store.collect_round(n_batches=1)

        # Each rank measures its own layers
        for layer_name in my_layers:
            # Check how many tokens are available in queue
            queue = activation_store.queues.get(layer_name)
            if queue is None:
                logger.warning(f"[Rank {rank}] Batch {i}: {layer_name} -> queue not found!")
                continue

            avail_tokens = queue.ntoks
            if avail_tokens <= 0:
                logger.debug(f"[Rank {rank}] Batch {i}: {layer_name} -> no tokens in queue yet")
                continue

            # Request all available tokens (or a reasonable limit)
            request_size = min(avail_tokens, 100000)
            batch = activation_store.next_batch(layer_name, batch_size=request_size)

            if batch is not None and batch.numel() > 0:
                tokens_per_layer[layer_name].append(batch.shape[0])
                logger.debug(f"[Rank {rank}] Batch {i}: {layer_name} -> {batch.shape[0]} tokens")

    # Calculate averages for this rank's layers
    local_result = {}
    for layer_name in my_layers:
        if tokens_per_layer[layer_name]:
            avg = sum(tokens_per_layer[layer_name]) / len(tokens_per_layer[layer_name])
            local_result[layer_name] = avg
            logger.info(
                f"[Rank {rank}] {layer_name}: {avg:.1f} tokens/inference "
                f"(min={min(tokens_per_layer[layer_name])}, "
                f"max={max(tokens_per_layer[layer_name])})"
            )
        else:
            logger.error(f"[Rank {rank}] {layer_name}: NO TOKENS COLLECTED!")
            local_result[layer_name] = 0.0

    # Gather results from all ranks
    result = local_result
    if world_size > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            # Convert local_result to list
            local_result_list = [(k, v) for k, v in local_result.items()]

            # All-gather results from all ranks
            gathered_results = [None] * world_size
            dist.all_gather_object(gathered_results, local_result_list)

            # Merge all results
            result = {}
            for rank_results in gathered_results:
                if rank_results:
                    result.update(dict(rank_results))

            if rank == 0:
                logger.info(f"Gathered probe results from {world_size} ranks: {list(result.keys())}")

    return result


def auto_calculate_subsample_rates(
    tokens_per_inference: Dict[str, float],
    primary_layer: str,
    primary_subsample: float = 1.0,
    min_rate: float = 0.00001,  # 0.001% (100x lower than before)
    max_rate: float = 1.0,
) -> Dict[str, float]:
    """
    Auto-calculate subsample rates to balance token collection across layers.

    Goal: Make all layers collect similar amounts of tokens per inference,
    based on primary layer's effective collection rate.

    Args:
        tokens_per_inference: Dict mapping layer names to tokens/inference
        primary_layer: Name of primary layer (reference for target rate)
        primary_subsample: Subsample rate for primary layer (default: 1.0)
        min_rate: Minimum subsample rate to prevent over-aggressive subsampling
        max_rate: Maximum subsample rate

    Returns:
        Dict mapping layer names to subsample rates
    """
    if primary_layer not in tokens_per_inference:
        logger.warning(
            f"Primary layer '{primary_layer}' not found in tokens_per_inference. "
            "Cannot auto-calculate subsample rates."
        )
        return {}

    # Target: primary layer's effective tokens per inference
    primary_tpi = tokens_per_inference[primary_layer]
    target_tokens_per_inf = primary_tpi * primary_subsample

    logger.info("Auto-calculating subsample rates:")
    logger.info(f"  Primary layer: {primary_layer}")
    logger.info(f"  Primary tokens/inference: {primary_tpi:.1f}")
    logger.info(f"  Primary subsample rate: {primary_subsample}")
    logger.info(f"  Target tokens/inference (for all layers): {target_tokens_per_inf:.1f}")
    logger.info(f"  Min subsample rate: {min_rate}, Max: {max_rate}")

    subsample_rates = {}
    for layer_name, tpi in tokens_per_inference.items():
        if tpi <= 0:
            logger.warning(f"  {layer_name}: invalid tpi={tpi}, skipping")
            continue

        # Calculate rate to match target
        rate = target_tokens_per_inf / tpi

        # Clamp to [min_rate, max_rate]
        rate = max(min_rate, min(max_rate, rate))

        subsample_rates[layer_name] = rate

        effective_tokens = tpi * rate
        logger.info(
            f"  {layer_name}: tpi={tpi:.1f} → rate={rate:.4f} → "
            f"effective={effective_tokens:.1f} tokens/inf"
        )

    return subsample_rates


def setup_logging(output_dir: Path, verbose: bool = False):
    """Configure logging."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "extraction.log"

    level = logging.DEBUG if verbose else logging.INFO

    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger.info(f"Logging to {log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract activations for K-means clustering initialization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base output directory for activations and checkpoints",
    )
    parser.add_argument(
        "--primary-layer",
        type=str,
        required=True,
        help="Reference layer name for calculating target inferences",
    )
    parser.add_argument(
        "--target-tokens-primary",
        type=int,
        default=10_000_000,
        help="Target number of tokens for primary layer (default: 10M)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        nargs="+",
        default=None,
        help="List of layer names to extract (default: all from config)",
    )
    parser.add_argument(
        "--auto-probe",
        action="store_true",
        help="Automatically probe tokens per inference before extraction",
    )
    parser.add_argument(
        "--subsample-rates",
        type=str,
        default=None,
        help='JSON dict of per-layer subsample rates, e.g. \'{"layer1": 0.5, "layer2": 0.25}\'',
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=16384,
        help="Flush buffer every N tokens (default: 16384)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save checkpoint every N inferences (default: 100)",
    )
    parser.add_argument(
        "--num-probe-batches",
        type=int,
        default=10,
        help="Number of batches to use for probing (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG level) logging",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pt", "parquet"],
        default="pt",
        help="Output file format: pt (PyTorch) or parquet (Arrow) (default: pt)",
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    setup_logging(output_dir, args.verbose)

    logger.info("=" * 80)
    logger.info("Inference-based Activation Extraction for K-means")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Primary layer: {args.primary_layer}")
    logger.info(f"Target tokens (primary): {args.target_tokens_primary:,}")
    logger.info(f"Output format: {args.format}")

    # Validate Parquet availability if needed
    if args.format == "parquet" and not PARQUET_AVAILABLE:
        logger.error("Parquet format requested but PyArrow is not installed.")
        logger.error("Install PyArrow with: pip install pyarrow")
        sys.exit(1)

    # Parse subsample rates
    subsample_rates = {}
    if args.subsample_rates:
        try:
            subsample_rates = json.loads(args.subsample_rates)
            logger.info(f"Subsample rates: {subsample_rates}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse subsample rates JSON: {e}")
            sys.exit(1)

    # Initialize pipeline (this loads model, dataset, activation store)
    # Support both DDP and single-GPU execution
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    logger.info("Initializing SAE training pipeline...")
    logger.info(f"DDP config: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    # Helper to log GPU memory
    def log_gpu_mem(tag: str):
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            alloc = torch.cuda.memory_allocated(dev) / 1e9
            reserved = torch.cuda.memory_reserved(dev) / 1e9
            logger.info(f"[Rank {rank}] GPU Memory @ {tag}: allocated={alloc:.2f}GB, reserved={reserved:.2f}GB")

    log_gpu_mem("script_start")

    # Initialize DDP if multi-GPU
    if world_size > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
            logger.info(f"[Rank {rank}] DDP initialized with nccl backend")
            log_gpu_mem("after_ddp_init")

    pipeline = SAETrainingPipeline(args.config, rank=rank, world_size=world_size)
    log_gpu_mem("after_pipeline_init")

    # Load model and dataset
    logger.info("Loading model and dataset...")
    pipeline.model = pipeline._load_model()
    log_gpu_mem("after_load_model")

    pipeline._load_dataset()
    log_gpu_mem("after_load_dataset")

    # Get layers to extract (must be specified for extraction)
    if not args.layers:
        raise ValueError("--layers must be specified for extraction (discovery disabled to save memory)")
    layers = args.layers

    logger.info(f"Layers to extract: {len(layers)}")
    for layer in layers:
        logger.info(f"  - {layer}")

    # Disable hook validation for extraction (we manually specify layers)
    pipeline.config["sae"]["validate_hook_points"] = {"enabled": False}

    # Initialize extractor state (before activation store creation)
    extractor_state = {
        "output_dir": output_dir,
        "primary_layer": args.primary_layer,
        "target_tokens_primary": args.target_tokens_primary,
        "layers": layers,
        "subsample_rates": subsample_rates,
        "flush_every_tokens": args.flush_every,
        "rank": rank,
        "world_size": world_size,
        "file_format": args.format,
    }

    # Try to resume from checkpoint
    resumed = False
    checkpoint_data = None
    if args.resume:
        ckpt_path = os.path.join(output_dir, "checkpoint.json")
        if os.path.exists(ckpt_path):
            logger.info(f"[Rank {rank}] Resuming from checkpoint: {ckpt_path}")
            try:
                with open(ckpt_path, "r") as f:
                    checkpoint_data = json.load(f)
                resumed = True
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                resumed = False

    # ====================================================================
    # PHASE 1: Auto-probe and calculate subsample rates (if needed)
    # This happens BEFORE final hook registration
    # ====================================================================

    tokens_per_inference = None
    auto_rates = None

    if resumed and checkpoint_data:
        # Load subsample rates and tokens_per_inference from checkpoint
        logger.info("Loading subsample rates from checkpoint...")
        layers_data = checkpoint_data.get("layers", {})

        # Extract subsample rates and tokens_per_inference
        checkpoint_rates = {}
        tokens_per_inference = {}
        for layer_name, layer_info in layers_data.items():
            rate = layer_info.get("subsample_rate", 1.0)
            tpi = layer_info.get("tokens_per_inference", -1)
            checkpoint_rates[layer_name] = rate
            if tpi > 0:
                tokens_per_inference[layer_name] = tpi

        # Apply checkpoint subsample rates to config
        if "per_layer" not in pipeline.config["sae"]["training"]:
            pipeline.config["sae"]["training"]["per_layer"] = {}
        for layer_name, rate in checkpoint_rates.items():
            if layer_name not in pipeline.config["sae"]["training"]["per_layer"]:
                pipeline.config["sae"]["training"]["per_layer"][layer_name] = {}
            pipeline.config["sae"]["training"]["per_layer"][layer_name]["random_subsample_rate"] = rate

        logger.info(f"Loaded subsample rates for {len(checkpoint_rates)} layers from checkpoint")
        for layer_name, rate in sorted(checkpoint_rates.items()):
            tpi = tokens_per_inference.get(layer_name, 0)
            logger.info(f"  {layer_name}: {rate:.6f} (tokens/inf: {tpi:.0f})")

    elif not resumed and args.auto_probe:
        # Step 1: Create temporary activation store with subsample_rate=1.0 for probing
        logger.info("Creating temporary activation store for probing (subsample_rate=1.0)...")

        # IMPORTANT: Save the original primary layer subsample rate BEFORE overwriting
        original_primary_subsample = 0.0625  # Default: 6.25% sampling (1/16)
        if args.primary_layer in pipeline.config["sae"]["training"].get("per_layer", {}):
            original_primary_subsample = pipeline.config["sae"]["training"]["per_layer"][args.primary_layer].get("random_subsample_rate", 0.0625)
        logger.info(f"Saved original primary subsample rate: {original_primary_subsample}")

        # Temporarily set all layers to subsample_rate=1.0
        if "per_layer" not in pipeline.config["sae"]["training"]:
            pipeline.config["sae"]["training"]["per_layer"] = {}
        for layer in layers:
            if layer not in pipeline.config["sae"]["training"]["per_layer"]:
                pipeline.config["sae"]["training"]["per_layer"][layer] = {}
            pipeline.config["sae"]["training"]["per_layer"][layer]["random_subsample_rate"] = 1.0

        # Create probe activation store
        pipeline._create_activation_store()
        log_gpu_mem("after_create_probe_store")

        # Initialize for probing
        pipeline.activation_store.expanded_hook_points = list(layers)
        pipeline.activation_store._probe_ran = True

        # Initialize queues
        for lname in layers:
            if lname not in pipeline.activation_store.queues:
                from src.core.sae.activation_stores.universal_activation_store import TokenBlockQueue
                cap_blocks = pipeline.activation_store.in_memory_blocks_per_layer
                spill_here = pipeline.activation_store.spill_dir if pipeline.activation_store.spill_to_disk else None
                allow_gpu = not pipeline.activation_store.buffer_on_cpu
                pipeline.activation_store.queues[lname] = TokenBlockQueue(
                    block_size_tokens=pipeline.activation_store.block_size_tokens,
                    spill_dir=spill_here,
                    in_memory_blocks_cap=cap_blocks,
                    lname=lname,
                    allow_gpu=allow_gpu,
                )

        # Set layer owners for DDP
        if world_size > 1:
            layer_owners = {layer: i % world_size for i, layer in enumerate(layers)}
            pipeline.activation_store.set_layer_owners(layer_owners)

        log_gpu_mem("before_probe")

        # Step 2: Run probe
        logger.info("Probing tokens per inference...")
        tokens_per_inference = probe_tokens_per_inference(
            pipeline.activation_store,
            layers,
            num_probe_batches=args.num_probe_batches,
            rank=rank,
            world_size=world_size,
        )

        log_gpu_mem("after_probe")

        # Step 3: Calculate auto subsample rates if not manually provided
        if not subsample_rates:
            logger.info("Auto-calculating subsample rates...")
            # Use the saved original primary subsample rate (not the config, which was overwritten to 1.0)
            primary_subsample = original_primary_subsample
            logger.info(f"Using saved primary subsample rate: {primary_subsample}")

            auto_rates = auto_calculate_subsample_rates(
                tokens_per_inference,
                primary_layer=args.primary_layer,
                primary_subsample=primary_subsample,
                min_rate=0.00001,  # At least 0.001% sampling (100x lower)
                max_rate=1.0,
            )

            logger.info(f"Auto-calculated subsample rates for {len(auto_rates)} layers:")
            for layer_name, rate in sorted(auto_rates.items()):
                tpi = tokens_per_inference.get(layer_name, 0)
                logger.info(f"  {layer_name}: {rate:.6f} (tokens/inf: {tpi:.0f})")

            # Update config with calculated rates (will be used when re-creating store)
            for layer_name, rate in auto_rates.items():
                if layer_name not in pipeline.config["sae"]["training"]["per_layer"]:
                    pipeline.config["sae"]["training"]["per_layer"][layer_name] = {}
                pipeline.config["sae"]["training"]["per_layer"][layer_name]["random_subsample_rate"] = rate
        elif subsample_rates:
            # Use manually provided rates
            logger.info(f"Using manually provided subsample rates for {len(subsample_rates)} layers")
            for layer_name, rate in subsample_rates.items():
                if layer_name not in pipeline.config["sae"]["training"]["per_layer"]:
                    pipeline.config["sae"]["training"]["per_layer"][layer_name] = {}
                pipeline.config["sae"]["training"]["per_layer"][layer_name]["random_subsample_rate"] = rate

        # Step 4: Cleanup probe activation store
        logger.info("Cleaning up probe activation store...")
        pipeline.activation_store.cleanup()
        del pipeline.activation_store
        gc.collect()
        torch.cuda.empty_cache()
        log_gpu_mem("after_cleanup_probe_store")

    elif not resumed and not args.auto_probe:
        logger.error(
            "No checkpoint found and --auto-probe not specified. "
            "Cannot determine target inference count. Exiting."
        )
        sys.exit(1)

    # ====================================================================
    # PHASE 2: Create final activation store with correct subsample rates
    # Hooks will now capture the correct subsample_rate values
    # ====================================================================

    logger.info("Creating final activation store with configured subsample rates...")
    pipeline._create_activation_store()
    log_gpu_mem("after_create_final_store")

    # Manually set expanded hook points (skip discovery to save memory)
    pipeline.activation_store.expanded_hook_points = list(layers)
    pipeline.activation_store._probe_ran = True  # Mark as if discovery ran

    # Initialize queues for specified layers
    for lname in layers:
        if lname not in pipeline.activation_store.queues:
            from src.core.sae.activation_stores.universal_activation_store import TokenBlockQueue
            cap_blocks = pipeline.activation_store.in_memory_blocks_per_layer
            spill_here = pipeline.activation_store.spill_dir if pipeline.activation_store.spill_to_disk else None
            allow_gpu = not pipeline.activation_store.buffer_on_cpu
            pipeline.activation_store.queues[lname] = TokenBlockQueue(
                block_size_tokens=pipeline.activation_store.block_size_tokens,
                spill_dir=spill_here,
                in_memory_blocks_cap=cap_blocks,
                lname=lname,
                allow_gpu=allow_gpu,
            )
            if pipeline.activation_store.enable_provenance:
                pipeline.activation_store.prov_queues[lname] = TokenBlockQueue(
                    block_size_tokens=pipeline.activation_store.block_size_tokens,
                    spill_dir=spill_here,
                    in_memory_blocks_cap=cap_blocks,
                    lname=f"{lname}__prov",
                )

    logger.info(f"Initialized {len(layers)} layer queues (discovery skipped)")
    log_gpu_mem("after_queue_init")

    # Set layer owners (required for DDP owner_stream pattern)
    if world_size > 1:
        layer_owners = {layer: i % world_size for i, layer in enumerate(layers)}
        pipeline.activation_store.set_layer_owners(layer_owners)
        logger.info(f"[Rank {rank}] Layer owners: {layer_owners}")

    # Create extractor with final activation store
    extractor = InferenceBasedExtractor(
        activation_store=pipeline.activation_store,
        **extractor_state
    )

    # Resume from checkpoint if available
    if args.resume:
        checkpoint_loaded = extractor.initialize_from_checkpoint()
        if checkpoint_loaded:
            # Extract subsample rates from layer_state (loaded from checkpoint)
            for layer_name, layer_info in extractor.layer_state.items():
                rate = layer_info.get("subsample_rate", 1.0)
                extractor.subsample_rates[layer_name] = rate
            logger.info(f"Restored subsample rates for {len(extractor.subsample_rates)} layers from checkpoint")

    # Update extractor's subsample_rates FIRST (before calculating target inferences)
    if auto_rates and not resumed:
        extractor.subsample_rates.update(auto_rates)
    elif subsample_rates and not resumed:
        extractor.subsample_rates.update(subsample_rates)
    else:
        # Fallback: Read subsample rates from config for all layers
        if not resumed:
            for layer_name in layers:
                if layer_name in pipeline.config["sae"]["training"].get("per_layer", {}):
                    rate = pipeline.config["sae"]["training"]["per_layer"][layer_name].get("random_subsample_rate", 1.0)
                    extractor.subsample_rates[layer_name] = rate
            if extractor.subsample_rates:
                logger.info(f"Loaded subsample rates from config for {len(extractor.subsample_rates)} layers")

    # Store tokens_per_inference in extractor state (from probe or checkpoint)
    if tokens_per_inference and not resumed:
        for layer_name, tpi in tokens_per_inference.items():
            if layer_name not in extractor.layer_state:
                extractor.layer_state[layer_name] = {}
            extractor.layer_state[layer_name]["tokens_per_inference"] = tpi

        # Calculate target inferences (AFTER subsample_rates are set)
        extractor.target_inferences = extractor.calculate_target_inferences(tokens_per_inference)

    # Prefill activation store
    if not resumed:
        logger.info("Prefilling activation store...")
        prefill_batches = 2
        pipeline.activation_store.collect_round(n_batches=prefill_batches)
        logger.info(f"Prefilled with {prefill_batches} batches")

    # Run extraction
    logger.info("Starting extraction...")
    try:
        extractor.extract(save_checkpoint_every=args.checkpoint_every)
        logger.info("Extraction completed successfully!")
    except KeyboardInterrupt:
        logger.warning("Extraction interrupted by user")
        logger.info("Saving checkpoint before exit...")
        # Flush any remaining data
        for layer_name in extractor.layers:
            if extractor.layer_buffers[layer_name]:
                extractor._flush_layer_buffer(layer_name)
        extractor.save_checkpoint()
        logger.info("Checkpoint saved. You can resume with --resume flag.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Extraction failed with error: {e}", exc_info=True)

        # Check if it's an NCCL error
        error_str = str(e).lower()
        is_nccl_error = "nccl" in error_str or "communicator" in error_str or "timeout" in error_str

        if is_nccl_error:
            logger.error("NCCL communication error detected. This may be caused by:")
            logger.error("  1. GPU memory exhaustion on one or more ranks")
            logger.error("  2. Network issues between GPUs")
            logger.error("  3. Imbalanced workload causing synchronization timeout")
            logger.error("  4. Try reducing batch_size or increasing checkpoint frequency")

        logger.info("Attempting to save checkpoint...")
        try:
            # Skip aggregation on NCCL errors (use local counts only)
            extractor.save_checkpoint()
            logger.info("Checkpoint saved despite error.")
            logger.info(f"Resume from inference {extractor.inferences_completed} with --resume flag")
        except Exception as e2:
            logger.error(f"Failed to save checkpoint: {e2}")
            logger.error("Checkpoint save failed. You may need to restart from scratch.")
        sys.exit(1)
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        pipeline.cleanup()

        # Cleanup DDP if initialized
        if world_size > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
                logger.info(f"[Rank {rank}] DDP process group destroyed")

    if rank == 0:  # Only owner rank prints summary
        logger.info("=" * 80)
        logger.info("Extraction completed successfully!")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Total inferences: {extractor.inferences_completed:,}")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
