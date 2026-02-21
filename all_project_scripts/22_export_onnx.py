#!/usr/bin/env python3
"""
Phase 5, Step 6: Export best PyTorch model to ONNX FP32 format.

Verifies that ONNX model produces identical predictions to PyTorch model.

Usage:
    python scripts/22_export_onnx.py
"""

import sys
import os
import numpy as np
import torch
import onnx
import onnxruntime as ort
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magicc.model import MAGICCModel


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(project_dir, 'models')
    best_model_path = os.path.join(model_dir, 'best_model.pt')
    onnx_path = os.path.join(model_dir, 'magicc_model.onnx')
    h5_path = os.path.join(project_dir, 'data', 'features', 'magicc_features.h5')

    print("=" * 70)
    print("ONNX Export and Verification")
    print("=" * 70)

    # Load PyTorch model
    print("\n1. Loading best PyTorch model...")
    model = MAGICCModel(
        n_kmer_features=9249,
        n_assembly_features=20,
        use_gradient_checkpointing=False,  # No checkpointing for export
    )
    ckpt = torch.load(best_model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    best_epoch = ckpt.get('best_epoch', ckpt.get('epoch', 'unknown'))
    best_val_loss = ckpt.get('best_val_loss', 'unknown')
    print(f"  Loaded model from epoch {best_epoch} (val_loss={best_val_loss})")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Export to ONNX
    print("\n2. Exporting to ONNX FP32...")
    dummy_kmer = torch.randn(1, 9249)
    dummy_asm = torch.randn(1, 20)

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
    onnx_size = os.path.getsize(onnx_path) / 1e6
    print(f"  Exported to {onnx_path} ({onnx_size:.1f} MB)")

    # Validate ONNX model
    print("\n3. Validating ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX model validation passed")

    # Verify predictions match
    print("\n4. Verifying predictions match PyTorch model...")

    # Load test samples from HDF5
    with h5py.File(h5_path, 'r') as f:
        test_kmer = f['test/kmer_features'][:1000]
        test_asm = f['test/assembly_features'][:1000]
        test_labels = f['test/labels'][:1000]

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
        }
    )[0]

    # Compare
    max_diff = np.max(np.abs(pt_pred - ort_pred))
    mean_diff = np.mean(np.abs(pt_pred - ort_pred))
    print(f"  Max absolute difference: {max_diff:.8f}")
    print(f"  Mean absolute difference: {mean_diff:.8f}")

    if max_diff < 1e-4:
        print("  VERIFICATION PASSED: Predictions are essentially identical")
    elif max_diff < 1e-2:
        print("  VERIFICATION PASSED: Predictions match within acceptable tolerance")
    else:
        print("  WARNING: Predictions differ significantly!")

    # Report prediction quality on test samples
    print("\n5. ONNX model prediction quality on 1000 test samples:")
    comp_mae = np.mean(np.abs(ort_pred[:, 0] - test_labels[:, 0]))
    cont_mae = np.mean(np.abs(ort_pred[:, 1] - test_labels[:, 1]))
    print(f"  Completeness MAE: {comp_mae:.4f}%")
    print(f"  Contamination MAE: {cont_mae:.4f}%")
    print(f"  Completeness range: [{ort_pred[:, 0].min():.2f}, {ort_pred[:, 0].max():.2f}]")
    print(f"  Contamination range: [{ort_pred[:, 1].min():.2f}, {ort_pred[:, 1].max():.2f}]")

    # Inference speed test
    print("\n6. ONNX inference speed test (CPU)...")
    import time
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
        # Timed
        t0 = time.time()
        for _ in range(n_runs):
            ort_session.run(None, {
                'kmer_features': kmer_batch,
                'assembly_features': asm_batch,
            })
        elapsed = (time.time() - t0) / n_runs
        per_sample = elapsed / bs * 1000
        print(f"  Batch size {bs:>5}: {elapsed*1000:.2f} ms/batch, "
              f"{per_sample:.3f} ms/sample")

    print(f"\nONNX model saved to: {onnx_path}")
    print("Export complete!")


if __name__ == '__main__':
    main()
