# Allen Brain Observatory - Neuropixels Visual Coding: End-to-End Analysis

This repository provides an end-to-end pipeline to analyze the Allen Brain Observatory Neuropixels Visual Coding dataset using AllenSDK. It sets up a standardized workflow to compute per-unit responses to drifting gratings (latency, OSI, DSI where available), aggregate metrics by anatomical area, perform statistical tests, and generate figures aligned with research goals on functional hierarchy in the mouse visual system.

Dataset
- Name: Allen Brain Observatory - Neuropixels Visual Coding
- Modality: Electrophysiology (Neuropixels)
- Source: https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html
- Notes: N=58 mice, ~100,000 units from cortical and thalamic visual areas

Research motivation
- Gap: Direct cellular-level functional hierarchy evidence has been limited due to technical constraints in simultaneous multi-region recordings.
- Methodological advance: The dataset enables large-scale, standardized multi-area spiking recordings across visual pathways.
- Questions:
  - Does spiking activity follow the anatomical hierarchy during visual stimulation?
  - How do classical hierarchical metrics (e.g., OSI, DSI) relate to hierarchy rank?
  - Is the hierarchical organization behaviorally relevant?

Repository structure
- neuropixels_project/
  - scripts/
    - analysis.py: Main analysis pipeline script
  - cache/: Data cache, manifes​t, and generated outputs (excluded from version control via .gitignore)
- README.md: This documentation
- requirements.txt: Python dependencies
- .gitignore: Exclusions for cache/ and generated artifacts

Environment setup
1. Create and activate a Python environment (Python 3.8 recommended)
2. Install dependencies:
   - pip install -r requirements.txt
   - Alternatively, use micromamba/conda with Python 3.8 and install the same set

Key dependencies
- allensdk==2.16.2
- numpy==1.23.5
- pandas==1.5.3
- matplotlib==3.6.3
- seaborn==0.12.2
- scipy==1.10.1

Usage
1. First run will populate a local cache under neuropixels_project/cache/ (manifest.json is maintained by AllenSDK):
   - python neuropixels_project/scripts/analysis.py --manifest neuropixels_project/cache/manifest.json
2. The script will:
   - Initialize the AllenSDK project cache
   - Select a session with broad area coverage
   - Detect available drifting grating stimuli (tries in order: drifting_gratings, drifting_gratings_75_repeats, drifting_gratings_contrast)
   - Compute per-unit metrics:
     - Latency (median first-spike latency during stimulus window)
     - Orientation Selectivity Index (OSI) via vector method
     - Direction Selectivity Index (DSI) when direction is available
   - Aggregate metrics by anatomical area
   - Run Spearman correlations vs hierarchy rank
   - Generate figures and write CSV outputs under cache/

Outputs (under neuropixels_project/cache/)
- metrics_session_<id>.csv: Per-unit metrics
- area_stats_session_<id>.csv: Area-level median metrics and hierarchy ranks
- stats_summary.txt: Spearman correlation results
- Figures:
  - latency_by_area.png
  - latency_vs_hierarchy.png
  - latency_vs_osi.png

Hierarchy mapping
- A coarse hierarchy map (structure_rank_map in analysis.py) assigns ranks to visual thalamus (LGd, LP), superior colliculus (SC variants), and cortical areas (VISp, VISl/rl/al, VISpm/am, VISpor).
- Extend/adjust this mapping as needed to cover more structures present in sessions.

Notes and limitations
- DSI may be unavailable for sessions lacking explicit direction values in drifting stimuli; consider using sessions with direction-bearing stimuli or alternative motion stimuli (e.g., dot_motion) to compute DSI.
- Single-session results provide preliminary trends; robust conclusions require multi-session aggregation and potentially mixed-effects modeling.

Reproducibility
- The pipeline uses AllenSDK’s EcephysProjectCache and reads stimulus tables per session.
- Cache path can be overridden via the ECEPHYS_CACHE_DIR environment variable.

License
- This analysis script uses data from the Allen Brain Observatory; refer to Allen Brain Atlas licensing for dataset usage.
