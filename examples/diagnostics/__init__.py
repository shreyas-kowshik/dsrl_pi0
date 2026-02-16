"""Diagnostic plots for Residual RL evaluation.

Generates animated .mp4 diagnostic plots during evaluation:
- Q-function diagnostics (Q01-Q08)
- UMAP action landscape plots (L01-L05)
- Actor/residual diagnostics (E01-E02)

Output directory structure:
    {outputdir}/diagnostics/step_{i}/traj_{rollout_id}/plot_name.mp4
    {outputdir}/diagnostics/step_{i}/aggregate/plot_name.mp4
"""
