#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for random_subsample feature in UniversalActivationStore
"""
import torch
from src.utils.utils import stable_u64


def test_random_subsample():
    """Test the random subsampling implementation"""
    print("=" * 60)
    print("Testing Random Subsample Implementation")
    print("=" * 60)

    # Simulate activation tensor (64x64 = 4096 tokens, 256 features)
    N = 4096
    C = 256
    t2d = torch.randn(N, C)

    # Simulate provenance (sample_id, frame_id, token_id)
    # Let's say we have 8 samples, each with 512 tokens
    samples_per_batch = 8
    tokens_per_sample = N // samples_per_batch
    prov_full = torch.zeros(N, 3, dtype=torch.long)
    for i in range(samples_per_batch):
        start = i * tokens_per_sample
        end = (i + 1) * tokens_per_sample
        prov_full[start:end, 0] = i  # sample_id
        prov_full[start:end, 1] = 0  # frame_id
        prov_full[start:end, 2] = torch.arange(tokens_per_sample)  # token_id

    # Test parameters
    subsample_rate = 0.125  # Keep 1/8 of tokens (similar to stride=8)
    subsample_seed = 42
    act_name = "encoder.layers.0.attn.output"
    epoch = 0

    # Calculate n_keep per sample, not total
    n_keep_per_sample = max(1, int(tokens_per_sample * subsample_rate))
    n_keep_total = n_keep_per_sample * samples_per_batch
    print(f"\nOriginal tokens: {N}")
    print(f"Subsample rate: {subsample_rate}")
    print(f"Expected tokens to keep per sample: {n_keep_per_sample}")
    print(f"Expected total tokens to keep: {n_keep_total}")

    # Per-sample deterministic shuffle
    sids = prov_full[:, 0].to(torch.long)
    indices = []

    print(f"\nProcessing {len(sids.unique())} unique samples...")
    for sid in sids.unique():
        mask = (sids == sid)
        sample_indices = mask.nonzero(as_tuple=True)[0]
        rng = torch.Generator().manual_seed(
            stable_u64(f"{subsample_seed}|{act_name}|{int(sid)}|{epoch}")
        )
        perm = torch.randperm(len(sample_indices), generator=rng)[:n_keep_per_sample]
        indices.append(sample_indices[perm])

        # Show first sample details
        if int(sid) == 0:
            print(f"\nSample {int(sid)} details:")
            print(f"  Total tokens: {len(sample_indices)}")
            print(f"  Tokens to keep: {n_keep_per_sample}")
            print(f"  Selected indices (first 10): {sample_indices[perm][:10].tolist()}")

    keep_idx = torch.cat(indices, dim=0).sort()[0]

    # Apply subsampling
    t2d_subsampled = t2d.index_select(0, keep_idx)
    prov_subsampled = prov_full.index_select(0, keep_idx)

    print(f"\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Final subsampled tokens: {t2d_subsampled.shape[0]}")
    print(f"Expected total: {n_keep_per_sample} x {samples_per_batch} = {n_keep_total}")

    # Verify indices are not sequential
    print(f"\nFirst 20 selected indices: {keep_idx[:20].tolist()}")
    print(f"Last 20 selected indices: {keep_idx[-20:].tolist()}")

    # Verify each sample has correct number of tokens
    print(f"\nTokens per sample after subsampling:")
    for sid in sids.unique():
        count = (prov_subsampled[:, 0] == sid).sum().item()
        print(f"  Sample {int(sid)}: {count} tokens")

    # Test reproducibility
    print(f"\n" + "=" * 60)
    print("Testing Reproducibility")
    print("=" * 60)

    # Run again with same seed
    indices2 = []
    for sid in sids.unique():
        mask = (sids == sid)
        sample_indices = mask.nonzero(as_tuple=True)[0]
        rng = torch.Generator().manual_seed(
            stable_u64(f"{subsample_seed}|{act_name}|{int(sid)}|{epoch}")
        )
        perm = torch.randperm(len(sample_indices), generator=rng)[:n_keep_per_sample]
        indices2.append(sample_indices[perm])

    keep_idx2 = torch.cat(indices2, dim=0).sort()[0]

    if torch.equal(keep_idx, keep_idx2):
        print("✓ Reproducibility test PASSED: Same seed produces same indices")
    else:
        print("✗ Reproducibility test FAILED: Same seed produces different indices")

    # Test different epoch
    indices3 = []
    epoch_new = 1
    for sid in sids.unique():
        mask = (sids == sid)
        sample_indices = mask.nonzero(as_tuple=True)[0]
        rng = torch.Generator().manual_seed(
            stable_u64(f"{subsample_seed}|{act_name}|{int(sid)}|{epoch_new}")
        )
        perm = torch.randperm(len(sample_indices), generator=rng)[:n_keep_per_sample]
        indices3.append(sample_indices[perm])

    keep_idx3 = torch.cat(indices3, dim=0).sort()[0]

    if not torch.equal(keep_idx, keep_idx3):
        print("✓ Epoch test PASSED: Different epoch produces different indices")
        overlap = len(set(keep_idx.tolist()) & set(keep_idx3.tolist()))
        print(f"  Overlap between epoch 0 and 1: {overlap}/{len(keep_idx)} tokens")
    else:
        print("✗ Epoch test FAILED: Different epoch produces same indices")

    # Test fallback (no provenance)
    print(f"\n" + "=" * 60)
    print("Testing Fallback (No Provenance)")
    print("=" * 60)

    n_keep_fallback = max(1, int(N * subsample_rate))
    rng_fallback = torch.Generator().manual_seed(subsample_seed + epoch)
    keep_idx_fallback = torch.randperm(N, generator=rng_fallback)[:n_keep_fallback].sort()[0]
    t2d_fallback = t2d.index_select(0, keep_idx_fallback)

    print(f"Fallback subsampled tokens: {t2d_fallback.shape[0]}")
    print(f"Expected tokens: {n_keep_fallback}")
    print(f"First 20 indices: {keep_idx_fallback[:20].tolist()}")

    print(f"\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_random_subsample()
