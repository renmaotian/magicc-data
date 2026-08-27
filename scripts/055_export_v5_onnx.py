#!/usr/bin/env python3
"""
Export best V5 model (Run 3, V2 architecture) to ONNX format and verify.

Model: MAGICCModel (V2, no attention), 42.4M params
  n_kmer_features=9249, n_assembly_features=7
  Completeness: Sigmoid*50+50 -> [50,100]
  Contamination: Sigmoid*100 -> [0,100]

Usage:
    python scripts/55_export_v5_onnx.py
"""

import sys
import os
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magicc.model import MAGICCModel


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(project_dir, 'models')
    best_model_path = os.path.join(model_dir, 'best_model_v5_run3.pt')
    onnx_path = os.path.join(model_dir, 'magicc_v5.onnx')

    print("=" * 70)
    print("V5 ONNX Export and Verification")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Load PyTorch model
    # ------------------------------------------------------------------ #
    print("\n1. Loading best V5 PyTorch model...")
    model = MAGICCModel(
        n_kmer_features=9249,
        n_assembly_features=7,
        use_gradient_checkpointing=False,  # No checkpointing for export
    )
    ckpt = torch.load(best_model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    best_epoch = ckpt.get('best_epoch', ckpt.get('epoch', 'unknown'))
    best_val_loss = ckpt.get('best_val_loss', 'unknown')
    print(f"  Loaded model from epoch {best_epoch} (val_loss={best_val_loss})")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------ #
    # 2. Export to ONNX
    # ------------------------------------------------------------------ #
    print("\n2. Exporting to ONNX FP32 (opset 17)...")
    dummy_kmer = torch.randn(1, 9249)
    dummy_asm = torch.randn(1, 7)

    torch.onnx.export(
        model,
        (dummy_kmer, dummy_asm),
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['kmer_features', 'assembly_features'],
        output_names=['predictions'],
        dynamic_axes={
            'kmer_features': {0: 'batch_size'},
            'assembly_features': {0: 'batch_size'},
            'predictions': {0: 'batch_size'},
        },
    )
    onnx_size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"  Exported to {onnx_path}")
    print(f"  File size: {onnx_size_mb:.1f} MB")

    # ------------------------------------------------------------------ #
    # 3. Validate ONNX model structure
    # ------------------------------------------------------------------ #
    print("\n3. Validating ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX model validation passed")

    # Print input/output names and shapes
    print("\n  Inputs:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}: {shape}")
    print("  Outputs:")
    for out in onnx_model.graph.output:
        shape = [d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: {shape}")

    # ------------------------------------------------------------------ #
    # 4. Verify predictions: 1000 random samples
    # ------------------------------------------------------------------ #
    print("\n4. Verifying predictions (1000 random samples)...")

    np.random.seed(42)
    test_kmer = np.random.randn(1000, 9249).astype(np.float32)
    test_asm = np.random.randn(1000, 7).astype(np.float32)

    # PyTorch predictions
    with torch.no_grad():
        pt_kmer = torch.from_numpy(test_kmer)
        pt_asm = torch.from_numpy(test_asm)
        pt_pred = model(pt_kmer, pt_asm).numpy()

    # ONNX Runtime predictions
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_pred = ort_session.run(
        None,
        {
            'kmer_features': test_kmer,
            'assembly_features': test_asm,
        },
    )[0]

    # Compare
    max_diff = np.max(np.abs(pt_pred - ort_pred))
    mean_diff = np.mean(np.abs(pt_pred - ort_pred))
    print(f"  Max absolute difference:  {max_diff:.8f}")
    print(f"  Mean absolute difference: {mean_diff:.8f}")

    if max_diff < 1e-4:
        print("  VERIFICATION PASSED: max diff < 1e-4")
    elif max_diff < 1e-2:
        print("  VERIFICATION PASSED: within acceptable tolerance (< 1e-2)")
    else:
        print("  WARNING: Predictions differ significantly!")

    # Prediction ranges
    print(f"\n  Completeness range:   [{ort_pred[:, 0].min():.2f}, {ort_pred[:, 0].max():.2f}]")
    print(f"  Contamination range:  [{ort_pred[:, 1].min():.2f}, {ort_pred[:, 1].max():.2f}]")

    # ------------------------------------------------------------------ #
    # 5. Inference speed benchmarks
    # ------------------------------------------------------------------ #
    print("\n5. ONNX inference speed (CPU)...")
    n_runs = 10
    batch_sizes = [1, 100, 1000]

    for bs in batch_sizes:
        kmer_batch = test_kmer[:bs]
        asm_batch = test_asm[:bs]

        # Warmup
        for _ in range(3):
            ort_session.run(None, {
                'kmer_features': kmer_batch,
                'assembly_features': asm_batch,
            })

        # Timed runs
        t0 = time.time()
        for _ in range(n_runs):
            ort_session.run(None, {
                'kmer_features': kmer_batch,
                'assembly_features': asm_batch,
            })
        elapsed = (time.time() - t0) / n_runs
        per_sample = elapsed / bs * 1000
        print(f"  Batch size {bs:>5}: {elapsed * 1000:.2f} ms/batch, "
              f"{per_sample:.3f} ms/sample")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  ONNX file:     {onnx_path}")
    print(f"  File size:     {onnx_size_mb:.1f} MB")
    print(f"  Max abs diff:  {max_diff:.8f}")
    print(f"  Mean abs diff: {mean_diff:.8f}")
    print(f"  Verification:  {'PASSED' if max_diff < 1e-4 else 'MARGINAL' if max_diff < 1e-2 else 'FAILED'}")
    print("=" * 70)


if __name__ == '__main__':
    main()
