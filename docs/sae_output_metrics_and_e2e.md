# SAE Output Metrics and E2E KL+MSE

This document explains how to enable:

- output-side SAE fidelity metrics based on model logits / task loss
- optional end-to-end SAE training with `KL + MSE`

Implementation lives in:

- `src/core/sae/train/runner.py`
- `src/core/sae/activation_stores/universal_activation_store.py`

## Behavior

These paths use a recon-only SAE insertion:

- SAE wrapper `error` addback is disabled automatically
- extra forward passes temporarily disable activation-store capture

So:

- `val_out/*` metrics measure the model after replacing the target layer with the SAE reconstruction only
- `e2e KL+MSE` is also computed on the reconstruction-only path

Existing activation-space metrics are unchanged:

- `loss`
- `l2`
- `explained_var`
- `relative_l2`

Those still come from the normal local SAE forward on cached activations.

## 1. Output Metrics Only

If you only want evaluation metrics and do not want to change training loss:

```yaml
validation:
  enabled: true
  every_steps: 5000
  max_batches: 16
  output_metrics:
    enabled: true
    max_batches: 4
```

Logged metrics:

- `val_out/logit_mse`
- `val_out/logit_kl`
- `val_out/top1_consistency`
- `val_out/orig_ce`
- `val_out/recon_ce`
- `val_out/delta_ce`
- `val_out/orig_acc`
- `val_out/recon_acc`
- `val_out/delta_acc`

Notes:

- `delta_ce = recon_ce - orig_ce`
- positive `delta_ce` means SAE insertion hurt task loss
- `top1_consistency` is agreement between original and SAE-augmented predictions

## 2. Enable E2E KL+MSE Loss

The training loop supports an opt-in loss:

```yaml
sae:
  training:
    e2e_loss_name: kl_mse
    e2e_balance: dynamic
    e2e_start_step: 0
    e2e_every_steps: 1
    e2e_total_coeff: 1.0
```

Supported `e2e_loss_name` values:

- `kl_mse`
- `mse_kl`
- `kl+mse`
- `kl_mse_e2e`

If unset, no extra E2E loss is used.

### Dynamic Balance

Recommended if you want the paper-style setting from:

- Karvonen, “Revisiting End-To-End Sparse Autoencoder Training: A Short Finetune Is All You Need”

Config:

```yaml
sae:
  training:
    e2e_loss_name: kl_mse
    e2e_balance: dynamic
    e2e_total_coeff: 1.0
```

This uses:

```python
alpha_kl = (mse_loss / (kl_loss + 1e-8)).detach()
loss = e2e_total_coeff * 0.5 * (mse_loss + alpha_kl * kl_loss)
```

Implication:

- `e2e/total` will often stay numerically close to MSE scale
- watch `e2e/logits_kl` and `e2e/alpha_kl` together

### Fixed Balance

If you want explicit weighting:

```yaml
sae:
  training:
    e2e_loss_name: kl_mse
    e2e_balance: fixed
    e2e_mse_coeff: 1.0
    e2e_kl_coeff: 0.5
    e2e_total_coeff: 1.0
```

This uses:

```python
loss = e2e_total_coeff * (e2e_mse_coeff * mse_loss + e2e_kl_coeff * kl_loss)
```

## 3. Run E2E Less Frequently

E2E loss requires extra full-model forward passes.

To reduce cost:

```yaml
sae:
  training:
    e2e_loss_name: kl_mse
    e2e_balance: dynamic
    e2e_start_step: 5000
    e2e_every_steps: 100
```

This means:

- no E2E loss before step 5000
- then one E2E update every 100 SAE steps

This is the safest way to keep default training throughput mostly unchanged.

## 4. Variant-Specific Override

In multi-variant training, you can apply E2E loss to only one variant:

```yaml
sae:
  training:
    variants:
      - name: batchtopk
        sae_type: batch-topk

      - name: ra-all
        sae_type: ra-batchtopk
        e2e_loss_name: kl_mse
        e2e_balance: dynamic
        e2e_start_step: 10000
        e2e_every_steps: 200
```

Variant-local config overrides global training config.

## 5. Metrics Logged During E2E Training

When E2E loss is enabled, the following are logged under each layer/variant key:

- `e2e/mse`
- `e2e/logits_kl`
- `e2e/total`
- `e2e/alpha_kl` when `e2e_balance: dynamic`
- `e2e/orig_ce` when labels are available
- `e2e/recon_ce` when labels are available
- `e2e/delta_ce` when labels are available

## 6. Recommended Starting Points

For cheap monitoring only:

```yaml
validation:
  output_metrics:
    enabled: true
    max_batches: 2
```

For conservative E2E fine-tuning:

```yaml
sae:
  training:
    e2e_loss_name: kl_mse
    e2e_balance: dynamic
    e2e_start_step: 20000
    e2e_every_steps: 100
    e2e_total_coeff: 1.0
```

For aggressive debugging:

```yaml
sae:
  training:
    e2e_loss_name: kl_mse
    e2e_balance: dynamic
    e2e_start_step: 0
    e2e_every_steps: 1
```

## 7. Cost Notes

`validation.output_metrics`:

- adds extra validation forwards only at validation time

`e2e_loss_name`:

- adds extra training-time full-model forwards
- should be treated as expensive
- is default-off for this reason

If speed matters, prefer:

- `output_metrics.enabled: true`
- `e2e_loss_name` unset

Or run E2E sparsely with `e2e_every_steps`.

## 8. Example Debug Config

A tiny runnable example is provided at:

- `configs/clip_imagenet_e2e_debug.yaml`

It enables:

- `validation.output_metrics`
- `e2e_loss_name: kl_mse`
- `e2e_balance: dynamic`

## References

- Gemma Scope report: https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf
- SAEBench: https://arxiv.org/abs/2503.09532
- Revisiting End-To-End Sparse Autoencoder Training: A Short Finetune Is All You Need: https://arxiv.org/abs/2503.17272
- ApolloResearch official implementation: https://github.com/ApolloResearch/e2e_sae
