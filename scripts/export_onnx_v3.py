#!/usr/bin/env python3
"""
Export MAGICCModelV3 (best_model.pt) to ONNX FP32 format.

- Opset 17, dynamic batch sizes
- Inputs: kmer_features (batch, 9249), assembly_features (batch, 26)
- Verifies ONNX outputs match PyTorch within tolerance
- Benchmarks ONNX Runtime inference speed
"""

import sys
import os
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magicc.model import MAGICCModelV3


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(project_dir, 'models')
    best_model_path = os.path.join(model_dir, 'best_model.pt')
    onnx_path = os.path.join(model_dir, 'magicc_v3.onnx')

    print("=" * 70)
    print("MAGICCModelV3 ONNX Export and Verification")
    print("=" * 70)

    # ----------------------------------------------------------------
    # 1. Load PyTorch model
    # ----------------------------------------------------------------
    print("\n1. Loading best PyTorch model (MAGICCModelV3)...")
    model = MAGICCModelV3(
        n_kmer_features=9249,
        n_assembly_features=26,
        use_gradient_checkpointing=False,  # No checkpointing for export
    )
    ckpt = torch.load(best_model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    best_epoch = ckpt.get('best_epoch', ckpt.get('epoch', 'unknown'))
    best_val_loss = ckpt.get('best_val_loss', 'unknown')
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")
    print(f"  Parameters: {n_params:,}")

    # ----------------------------------------------------------------
    # 2. Export to ONNX
    # ----------------------------------------------------------------
    print("\n2. Exporting to ONNX FP32 (opset 17, dynamic batch)...")
    dummy_kmer = torch.randn(2, 9249)  # batch=2 so BN works in eval mode
    dummy_asm = torch.randn(2, 26)

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
    onnx_size_bytes = os.path.getsize(onnx_path)
    onnx_size_mb = onnx_size_bytes / 1e6
    print(f"  Exported to {onnx_path}")
    print(f"  ONNX file size: {onnx_size_mb:.1f} MB ({onnx_size_bytes:,} bytes)")

    # ----------------------------------------------------------------
    # 3. Validate ONNX model structure
    # ----------------------------------------------------------------
    print("\n3. Validating ONNX model structure...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX model validation passed (checker.check_model)")

    # Print model graph summary
    graph = onnx_model.graph
    print(f"  Inputs:  {[(inp.name, [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]) for inp in graph.input]}")
    print(f"  Outputs: {[(out.name, [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]) for out in graph.output]}")
    print(f"  Nodes:   {len(graph.node)}")

    # ----------------------------------------------------------------
    # 4. Verify predictions match PyTorch model
    # ----------------------------------------------------------------
    print("\n4. Verifying predictions match PyTorch model...")

    # Create random test data at multiple batch sizes
    np.random.seed(42)
    torch.manual_seed(42)

    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    batch_sizes_verify = [1, 8, 64, 256]
    max_diff_overall = 0.0
    mean_diff_overall = 0.0

    for bs in batch_sizes_verify:
        test_kmer = np.random.randn(bs, 9249).astype(np.float32)
        test_asm = np.random.randn(bs, 26).astype(np.float32)

        # PyTorch predictions
        with torch.no_grad():
            pt_pred = model(
                torch.from_numpy(test_kmer),
                torch.from_numpy(test_asm),
            ).numpy()

        # ONNX Runtime predictions
        ort_pred = ort_session.run(
            None,
            {
                'kmer_features': test_kmer,
                'assembly_features': test_asm,
            }
        )[0]

        max_diff = np.max(np.abs(pt_pred - ort_pred))
        mean_diff = np.mean(np.abs(pt_pred - ort_pred))
        max_diff_overall = max(max_diff_overall, max_diff)
        mean_diff_overall = max(mean_diff_overall, mean_diff)

        print(f"  Batch {bs:>4}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")

    print(f"\n  Overall max absolute difference: {max_diff_overall:.8f}")
    if max_diff_overall < 1e-5:
        print("  VERIFICATION PASSED: Predictions are essentially identical (<1e-5)")
    elif max_diff_overall < 1e-4:
        print("  VERIFICATION PASSED: Predictions match within tight tolerance (<1e-4)")
    elif max_diff_overall < 1e-2:
        print("  VERIFICATION PASSED: Predictions match within acceptable tolerance (<1e-2)")
    else:
        print("  WARNING: Predictions differ significantly!")

    # ----------------------------------------------------------------
    # 5. Check output ranges
    # ----------------------------------------------------------------
    print("\n5. Output range check (batch=256, random inputs):")
    test_kmer = np.random.randn(256, 9249).astype(np.float32)
    test_asm = np.random.randn(256, 26).astype(np.float32)
    ort_pred = ort_session.run(None, {
        'kmer_features': test_kmer,
        'assembly_features': test_asm,
    })[0]
    print(f"  Completeness range:  [{ort_pred[:, 0].min():.2f}, {ort_pred[:, 0].max():.2f}] (expected [50, 100])")
    print(f"  Contamination range: [{ort_pred[:, 1].min():.2f}, {ort_pred[:, 1].max():.2f}] (expected [0, 100])")

    # ----------------------------------------------------------------
    # 6. Inference speed benchmark
    # ----------------------------------------------------------------
    print("\n6. ONNX Runtime inference speed benchmark (CPU)...")
    n_warmup = 5
    n_runs = 20
    batch_sizes_bench = [1, 16, 64, 256, 1024]

    for bs in batch_sizes_bench:
        kmer_batch = np.random.randn(bs, 9249).astype(np.float32)
        asm_batch = np.random.randn(bs, 26).astype(np.float32)

        # Warmup
        for _ in range(n_warmup):
            ort_session.run(None, {
                'kmer_features': kmer_batch,
                'assembly_features': asm_batch,
            })

        # Timed runs
        t0 = time.perf_counter()
        for _ in range(n_runs):
            ort_session.run(None, {
                'kmer_features': kmer_batch,
                'assembly_features': asm_batch,
            })
        elapsed = (time.perf_counter() - t0) / n_runs
        per_sample = elapsed / bs * 1000
        print(f"  Batch {bs:>5}: {elapsed*1000:8.2f} ms/batch, {per_sample:.3f} ms/sample")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  ONNX file:    {onnx_path}")
    print(f"  File size:    {onnx_size_mb:.1f} MB")
    print(f"  Opset:        17")
    print(f"  Max diff:     {max_diff_overall:.8f}")
    print(f"  Parameters:   {n_params:,}")
    print(f"  Source epoch:  {best_epoch}")
    print("=" * 70)


if __name__ == '__main__':
    main()
