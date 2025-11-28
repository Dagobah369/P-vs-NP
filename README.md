[![DOI](https://zenodo.org/badge/1101614439.svg)](https://doi.org/10.5281/zenodo.17752870)

November 22, 2025

Abstract

This manuscript presents a proof of the average-case exponential hardness of
Random 3-SAT above its phase transition threshold (αc ≈ 4.26), using the spectral
coherence invariant rigorously established in [1] without any unproven complexitytheoretic
assumptions. Our approach treats the execution traces of DPLL algorithms
as spectral signals. We transpose the coherence invariant C(SAT)
N to the
sequence of local computational costs. The proof is structured around three analytic
bridges:
• (A) Detection: We demonstrate that the "hard" regime (α > αc) induces
long-range correlations ("backtracking waves") in the search tree, creating a
measurable positive signature (ϵ > 0) in the variance of the coherence coefficient.
• (B) Exclusion: Using combinatorial positivity arguments and the structure
of efficient algorithms, we prove that a hypothetical polynomial-time solution
implies strict spectral stability (ϵ = 0).
• (C) Construction: We model the search process via a transition operator
whose spectral gap closes at the phase transition, explaining the divergence of
the solving time.
The logical contradiction between the spectral signature of hardness (ϵ > 0) and
the stability required for polynomial solvability implies that Random 3-SAT is exponentially
hard in the supercritical regime. This work is deliberately orthogonal
to the known barriers (relativization, natural proofs, algebrization) and relies only
on measurable statistical properties of random instances.
