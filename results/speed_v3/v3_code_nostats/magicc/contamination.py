"""
Contamination Module for MAGICC.

Implements contamination mixing for synthetic genome generation:
- Random genome mixing with contamination rate uniformly distributed 0-100%
- CRITICAL: Contamination rate = total contaminant size / dominant genome FULL reference size
  (NOT the actual size after incompleteness reduction)
- Support within-phylum (1-3 genomes) and cross-phylum (1-5 genomes) contamination
- Each contaminant genome is independently fragmented and merged
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from magicc.fragmentation import simulate_fragmentation, read_fasta


def compute_contamination_rate(
    contaminant_total_bp: int,
    dominant_genome_full_length: int,
) -> float:
    """
    Compute contamination rate as per protocol.

    Contamination (%) = 100 * sum(contaminant lengths) / dominant genome FULL reference size

    Parameters
    ----------
    contaminant_total_bp : int
        Total base pairs from all contaminant genomes.
    dominant_genome_full_length : int
        Full reference length of the dominant genome (before any fragmentation).

    Returns
    -------
    float
        Contamination percentage (0-100).
    """
    if dominant_genome_full_length <= 0:
        return 0.0
    return 100.0 * contaminant_total_bp / dominant_genome_full_length


def select_contaminant_target_bp(
    target_contamination_rate: float,
    dominant_genome_full_length: int,
) -> int:
    """
    Compute the total contaminant bp needed for a target contamination rate.

    Parameters
    ----------
    target_contamination_rate : float
        Target contamination percentage (0-100).
    dominant_genome_full_length : int
        Full reference length of dominant genome.

    Returns
    -------
    int
        Target total contaminant bp.
    """
    return int(target_contamination_rate / 100.0 * dominant_genome_full_length)


def fragment_contaminant(
    sequence: str,
    target_bp: int,
    rng: np.random.Generator,
    quality_tier: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fragment a contaminant genome to produce a specified total bp.

    The contaminant genome is fragmented independently, then contigs are
    randomly selected until the target bp is reached. When target_bp exceeds
    the contaminant genome size, multiple copies are used (simulating
    multiple organisms of the same species or closely related species
    contributing to contamination).

    Parameters
    ----------
    sequence : str
        Full contaminant genome sequence.
    target_bp : int
        Target total base pairs to extract from this contaminant.
    rng : np.random.Generator
        Random number generator.
    quality_tier : str or None
        Quality tier for fragmentation. If None, randomly selected.

    Returns
    -------
    dict with keys:
        'contigs': list of str
        'total_length': int
        'genome_full_length': int
    """
    genome_len = len(sequence)
    if genome_len == 0 or target_bp <= 0:
        return {'contigs': [], 'total_length': 0, 'genome_full_length': genome_len}

    all_contigs = []
    remaining_bp = target_bp

    # Use multiple copies of the contaminant genome if target exceeds genome size
    # This handles high contamination levels (70-100%) where the contaminant
    # needs to contribute more bp than its own genome size
    while remaining_bp > 0:
        # How much of this copy do we need?
        if remaining_bp >= genome_len:
            # Use the full genome copy
            completeness = 1.0
        else:
            completeness = remaining_bp / genome_len

        # Fragment this copy
        result = simulate_fragmentation(
            sequence,
            target_completeness=completeness,
            quality_tier=quality_tier,
            rng=rng,
            apply_biases=True,
        )

        copy_contigs = result['contigs']
        copy_length = sum(len(c) for c in copy_contigs)

        all_contigs.extend(copy_contigs)
        remaining_bp -= copy_length

        # Safety: if this copy produced nothing, break to avoid infinite loop
        if copy_length == 0:
            break

    total_length = sum(len(c) for c in all_contigs)

    # If we got more than target, randomly drop contigs and/or truncate
    if total_length > target_bp * 1.2:
        if len(all_contigs) > 1:
            indices = np.arange(len(all_contigs))
            rng.shuffle(indices)
            kept = []
            kept_len = 0
            for idx in indices:
                c = all_contigs[idx]
                if kept_len + len(c) <= target_bp * 1.1:
                    kept.append(c)
                    kept_len += len(c)
                elif kept_len >= target_bp * 0.9:
                    break
            if len(kept) > 0:
                all_contigs = kept
                total_length = sum(len(c) for c in all_contigs)

        # If still over target, truncate the largest contig
        if total_length > target_bp * 1.2 and len(all_contigs) > 0:
            max_idx = max(range(len(all_contigs)), key=lambda i: len(all_contigs[i]))
            excess = total_length - int(target_bp * 1.1)
            if excess > 0 and excess < len(all_contigs[max_idx]):
                all_contigs[max_idx] = all_contigs[max_idx][:len(all_contigs[max_idx]) - excess]
            elif excess >= len(all_contigs[max_idx]):
                all_contigs[max_idx] = all_contigs[max_idx][:max(int(target_bp), 500)]
            total_length = sum(len(c) for c in all_contigs)

    return {
        'contigs': all_contigs,
        'total_length': total_length,
        'genome_full_length': genome_len,
    }


def generate_contaminated_sample(
    dominant_sequence: str,
    contaminant_sequences: List[str],
    target_completeness: float,
    target_contamination: float,
    rng: np.random.Generator,
    dominant_quality_tier: Optional[str] = None,
    contaminant_quality_tier: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a contaminated synthetic genome sample.

    Parameters
    ----------
    dominant_sequence : str
        Full sequence of the dominant genome.
    contaminant_sequences : list of str
        Full sequences of contaminant genomes.
    target_completeness : float
        Target completeness for the dominant genome (0.5 to 1.0).
    target_contamination : float
        Target contamination rate as percentage (0-100).
        Contamination = total contaminant bp / dominant genome FULL reference size.
    rng : np.random.Generator
        Random number generator.
    dominant_quality_tier : str or None
        Quality tier for dominant genome fragmentation.
    contaminant_quality_tier : str or None
        Quality tier for contaminant fragmentation.

    Returns
    -------
    dict with keys:
        'contigs': list of str - all contigs (dominant + contaminants merged)
        'dominant_contigs': list of str - dominant genome contigs only
        'contaminant_contigs': list of str - contaminant contigs only
        'completeness': float - actual completeness of dominant genome
        'contamination': float - actual contamination rate (%)
        'dominant_full_length': int - full reference length of dominant genome
        'n_contigs_dominant': int
        'n_contigs_contaminant': int
        'n_contaminant_genomes': int
        'dominant_quality_tier': str
    """
    dominant_full_length = len(dominant_sequence)

    if dominant_full_length == 0:
        return {
            'contigs': [],
            'dominant_contigs': [],
            'contaminant_contigs': [],
            'completeness': 0.0,
            'contamination': 0.0,
            'dominant_full_length': 0,
            'n_contigs_dominant': 0,
            'n_contigs_contaminant': 0,
            'n_contaminant_genomes': 0,
            'dominant_quality_tier': dominant_quality_tier or 'high',
        }

    # Step 1: Fragment the dominant genome
    dom_result = simulate_fragmentation(
        dominant_sequence,
        target_completeness=target_completeness,
        quality_tier=dominant_quality_tier,
        rng=rng,
    )
    dominant_contigs = dom_result['contigs']
    actual_completeness = dom_result['completeness']
    used_quality_tier = dom_result['quality_tier']

    # Step 2: Fragment contaminant genomes
    contaminant_contigs = []
    if target_contamination > 0 and len(contaminant_sequences) > 0:
        # Total target contaminant bp based on FULL reference size of dominant
        total_target_bp = select_contaminant_target_bp(
            target_contamination, dominant_full_length
        )

        # Distribute target bp among contaminant genomes
        n_contaminants = len(contaminant_sequences)
        if n_contaminants == 1:
            bp_per_contaminant = [total_target_bp]
        else:
            # Random proportional distribution
            proportions = rng.dirichlet(np.ones(n_contaminants))
            bp_per_contaminant = (proportions * total_target_bp).astype(int)
            # Adjust to match total
            bp_per_contaminant[-1] = total_target_bp - bp_per_contaminant[:-1].sum()

        for seq, target_bp in zip(contaminant_sequences, bp_per_contaminant):
            if target_bp <= 0 or len(seq) == 0:
                continue
            cont_result = fragment_contaminant(
                seq, target_bp, rng,
                quality_tier=contaminant_quality_tier,
            )
            contaminant_contigs.extend(cont_result['contigs'])

    # Cap total contaminant bp to ensure contamination stays strictly <= target
    # (handles cumulative overshoot from multiple contaminants)
    # Use strict 1.0x cap (no safety margin) so contamination never exceeds target.
    # After assembling all contaminant contigs, if total bp exceeds the target,
    # trim/truncate the last contig to hit exactly the target.
    contaminant_total_bp = sum(len(c) for c in contaminant_contigs)
    max_allowed_bp = int(target_contamination / 100.0 * dominant_full_length)
    if contaminant_total_bp > max_allowed_bp and len(contaminant_contigs) > 0:
        # Randomly shuffle and keep contigs until we hit the cap
        cc_indices = np.arange(len(contaminant_contigs))
        rng.shuffle(cc_indices)
        kept_contigs = []
        kept_bp = 0
        for idx in cc_indices:
            c = contaminant_contigs[idx]
            if kept_bp + len(c) <= max_allowed_bp:
                kept_contigs.append(c)
                kept_bp += len(c)
            else:
                # Truncate this contig to fit exactly at the cap
                remaining = max_allowed_bp - kept_bp
                if remaining >= 500:
                    kept_contigs.append(c[:remaining])
                    kept_bp += remaining
                break
        if len(kept_contigs) > 0:
            contaminant_contigs = kept_contigs

    # Final strict enforcement: if contaminant bp still exceeds target,
    # truncate the last (largest) contig to match exactly
    contaminant_total_bp = sum(len(c) for c in contaminant_contigs)
    if contaminant_total_bp > max_allowed_bp and len(contaminant_contigs) > 0:
        excess = contaminant_total_bp - max_allowed_bp
        # Truncate the last contig
        last = contaminant_contigs[-1]
        new_len = len(last) - excess
        if new_len >= 100:
            contaminant_contigs[-1] = last[:new_len]
        elif len(contaminant_contigs) > 1:
            # Last contig would be too small; remove it entirely
            contaminant_contigs.pop()
        # else: single contig, keep as-is (contamination may be slightly over
        # but only by a tiny amount from a single small contig)

    # Step 3: Merge contigs (randomly interleave)
    all_contigs = dominant_contigs + contaminant_contigs
    if len(all_contigs) > 1:
        indices = np.arange(len(all_contigs))
        rng.shuffle(indices)
        all_contigs = [all_contigs[i] for i in indices]

    # Compute actual contamination rate
    contaminant_total_bp = sum(len(c) for c in contaminant_contigs)
    actual_contamination = compute_contamination_rate(
        contaminant_total_bp, dominant_full_length
    )

    return {
        'contigs': all_contigs,
        'dominant_contigs': dominant_contigs,
        'contaminant_contigs': contaminant_contigs,
        'completeness': actual_completeness,
        'contamination': actual_contamination,
        'dominant_full_length': dominant_full_length,
        'n_contigs_dominant': len(dominant_contigs),
        'n_contigs_contaminant': len(contaminant_contigs),
        'n_contaminant_genomes': len(contaminant_sequences),
        'dominant_quality_tier': used_quality_tier,
    }


def generate_pure_sample(
    dominant_sequence: str,
    target_completeness: float,
    rng: np.random.Generator,
    quality_tier: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a pure (uncontaminated) synthetic genome sample.

    Parameters
    ----------
    dominant_sequence : str
        Full sequence of the dominant genome.
    target_completeness : float
        Target completeness (0.5 to 1.0).
    rng : np.random.Generator
        Random number generator.
    quality_tier : str or None
        Quality tier for fragmentation.

    Returns
    -------
    dict
        Same format as generate_contaminated_sample, with contamination=0.
    """
    return generate_contaminated_sample(
        dominant_sequence=dominant_sequence,
        contaminant_sequences=[],
        target_completeness=target_completeness,
        target_contamination=0.0,
        rng=rng,
        dominant_quality_tier=quality_tier,
    )
