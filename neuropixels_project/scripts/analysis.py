import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

CACHE_DIR = Path(os.environ.get("ECEPHYS_CACHE_DIR", "/workspace/neuropixels_project/cache"))
MANIFEST_PATH = CACHE_DIR / "manifest.json"


def init_cache(manifest_path=None):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mp = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH
    cache = EcephysProjectCache.from_warehouse(manifest=str(mp))
    return cache


def list_sessions(cache):
    sessions = cache.get_session_table()
    print(f"Total sessions: {len(sessions)}")
    return sessions


def pick_session_with_areas(sessions, min_areas=5):
    mask = sessions["ecephys_structure_acronyms"].apply(lambda x: isinstance(x, list) and len([a for a in x if isinstance(a, str)]) >= min_areas)
    chosen = sessions[mask].sort_values("unit_count", ascending=False).head(1)
    if len(chosen) == 0:
        chosen = sessions.sort_values("unit_count", ascending=False).head(1)
    sid = int(chosen.index.values[0])
    print(f"Selected session_id: {sid}")
    return sid


def find_session_with_stimulus(cache, sessions, stimulus_name="drifting_gratings", min_areas=5, max_tries=5):
    candidates = sessions.copy()
    mask = candidates["ecephys_structure_acronyms"].apply(lambda x: isinstance(x, list) and len([a for a in x if isinstance(a, str)]) >= min_areas)
    candidates = candidates[mask].sort_values("unit_count", ascending=False)
    if len(candidates) == 0:
        candidates = sessions.sort_values("unit_count", ascending=False)
    tried = 0
    for sid in candidates.index.tolist():
        tried += 1
        if tried > max_tries:
            break
        try:
            sess = download_session(cache, int(sid))
            st = sess.get_stimulus_table(stimulus_name)
            if st is not None and len(st) > 0:
                print(f"Using session {sid} with {stimulus_name} ({len(st)} presentations)")
                return int(sid), sess
        except Exception as e:
            print(f"Failed to load session {sid}: {e}")
            continue
    # Fallback: use best available
    sid = int(candidates.index.values[0])
    print(f"Fallback to session {sid}")
    return sid, download_session(cache, sid)


def download_session(cache, session_id):
    session = cache.get_session_data(session_id,
                                     amplitude_cutoff_maximum=np.inf,
                                     presence_ratio_minimum=-np.inf,
                                     isi_violations_maximum=np.inf)
    return session


def compute_latency_and_selectivity(session):
    # Choose available drifting grating stimulus
    preferred = ["drifting_gratings", "drifting_gratings_75_repeats", "drifting_gratings_contrast"]
    st = None
    for name in preferred:
        try:
            tmp = session.get_stimulus_table(name)
            if tmp is not None and len(tmp) > 0:
                st = tmp
                break
        except Exception:
            continue
    if st is None or len(st) == 0:
        print("No drifting grating stimulus found in session.")
        return pd.DataFrame(columns=["unit_id","structure","g_osi_dg_est","g_dsi_dg_est","latency_dg_est"])

    units = session.units
    spike_times = session.spike_times

    # Normalize orientation/direction columns
    if "orientation" not in st.columns and "grating_orientation" in st.columns:
        st["orientation"] = st["grating_orientation"]
    if "direction" not in st.columns and "grating_direction" in st.columns:
        st["direction"] = st["grating_direction"]
    if "orientation" not in st.columns and "direction" in st.columns:
        try:
            st["orientation"] = np.mod(st["direction"].astype(float), 180.0)
        except Exception:
            pass

    metrics = []
    for unit_id, unit in units.iterrows():
        # Example quality filter
        if np.isfinite(unit.get("amplitude_cutoff", 0)) and unit.get("amplitude_cutoff", 0) > 0.2:
            continue
        trials = st.copy()
        # Per-trial spike count in stimulus window
        counts = []
        spikes = spike_times.get(unit_id, np.array([]))
        for _, row in trials.iterrows():
            t0 = row["start_time"]
            t1 = row["stop_time"]
            counts.append(((spikes >= t0) & (spikes <= t1)).sum())
        trials["spike_count"] = counts
        sc = trials["spike_count"].values

        # Orientation selectivity via vector method (robust)
        if "orientation" in trials.columns:
            df = pd.DataFrame({"ori": trials["orientation"].values, "count": sc}).dropna()
            if df.shape[0] >= 2 and df["ori"].nunique() >= 2 and df["count"].sum() > 0:
                # average responses per orientation
                mean_by_ori = df.groupby("ori")["count"].mean()
                thetas = np.deg2rad(mean_by_ori.index.values.astype(float))
                responses = mean_by_ori.values.astype(float)
                vec = np.sum(responses * np.exp(2j * thetas))
                g_osi = 1.0 - (np.abs(vec) / np.sum(responses))
            else:
                g_osi = np.nan
        else:
            g_osi = np.nan

        # Direction selectivity (if available)
        if "direction" in trials.columns:
            df2 = pd.DataFrame({"dir": trials["direction"].values, "count": sc}).dropna()
            if df2.shape[0] >= 2 and df2["dir"].nunique() >= 2 and df2["count"].sum() > 0:
                mean_by_dir = df2.groupby("dir")["count"].mean()
                phis = np.deg2rad(mean_by_dir.index.values.astype(float))
                responses_d = mean_by_dir.values.astype(float)
                vecd = np.sum(responses_d * np.exp(1j * phis))
                g_dsi = 1.0 - (np.abs(vecd) / np.sum(responses_d))
            else:
                g_dsi = np.nan
        else:
            g_dsi = np.nan

        # Response latency: median first-spike latency across trials
        latencies = []
        for _, row in st.iterrows():
            t0 = row["start_time"]
            t1 = row["stop_time"]
            s = spikes[(spikes >= t0) & (spikes <= t1)]
            if len(s) > 0:
                latencies.append(s[0] - t0)
        latency = np.median(latencies) if len(latencies) > 0 else np.nan

        metrics.append({
            "unit_id": unit_id,
            "structure": unit.get("ecephys_structure_acronym", None),
            "g_osi_dg_est": g_osi,
            "g_dsi_dg_est": g_dsi,
            "latency_dg_est": latency
        })

    return pd.DataFrame(metrics)




def structure_rank_map():
    # Coarse anatomical hierarchy ranks (lower = earlier). Extend to thalamus and SC variants present.
    return {
        "LGd": 0, "LP": 1, "SCs": 0, "SCig": 0, "SCop": 0,
        "VISp": 2,
        "VISl": 3, "VISrl": 3, "VISal": 3,
        "VISpm": 4, "VISam": 4,
        "VISpor": 5,
    }

from scipy.stats import spearmanr

def stats_tests(area_stats, outdir=CACHE_DIR):
    area_stats = area_stats.dropna(subset=["hierarchy_rank"]).copy()
    rho_lat, p_lat = spearmanr(area_stats["hierarchy_rank"], area_stats["latency_dg_est"], nan_policy="omit")
    rho_osi, p_osi = spearmanr(area_stats["hierarchy_rank"], area_stats["g_osi_dg_est"], nan_policy="omit")
    rho_dsi, p_dsi = spearmanr(area_stats["hierarchy_rank"], area_stats["g_dsi_dg_est"], nan_policy="omit")
    with open(outdir / "stats_summary.txt", "w") as f:
        f.write(f"Spearman latency vs hierarchy: rho={rho_lat:.3f}, p={p_lat:.3e}\n")
        f.write(f"Spearman OSI vs hierarchy: rho={rho_osi:.3f}, p={p_osi:.3e}\n")
        f.write(f"Spearman DSI vs hierarchy: rho={rho_dsi:.3f}, p={p_dsi:.3e}\n")
    print({
        "latency_rho": rho_lat, "latency_p": p_lat,
        "osi_rho": rho_osi, "osi_p": p_osi,
        "dsi_rho": rho_dsi, "dsi_p": p_dsi
    })

def aggregate_by_area(metrics_df):
    area_stats = metrics_df.groupby("structure").agg({
        "latency_dg_est": "median",
        "g_osi_dg_est": "median",
        "g_dsi_dg_est": "median"
    })
    rank_map = structure_rank_map()
    area_stats["hierarchy_rank"] = [rank_map.get(a, np.nan) for a in area_stats.index]
    area_stats = area_stats.sort_values("latency_dg_est")
    return area_stats


def plot_figures(area_stats, outdir=CACHE_DIR):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.barplot(x=area_stats.index, y=area_stats["latency_dg_est"], color="steelblue")
    plt.xticks(rotation=90)
    plt.ylabel("Median latency (s)")
    plt.title("Drifting gratings latency by area")
    plt.tight_layout()
    plt.savefig(outdir / "latency_by_area.png")

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=area_stats["hierarchy_rank"], y=area_stats["latency_dg_est"])    
    plt.xlabel("Hierarchy rank (lower=earlier)")
    plt.ylabel("Median latency (s)")
    plt.title("Latency vs hierarchy rank")
    plt.tight_layout()
    plt.savefig(outdir / "latency_vs_hierarchy.png")

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=area_stats["g_osi_dg_est"], y=area_stats["latency_dg_est"])    
    plt.xlabel("Median OSI")
    plt.ylabel("Median latency (s)")
    plt.title("Latency vs OSI across areas")
    plt.tight_layout()
    plt.savefig(outdir / "latency_vs_osi.png")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=str, default=str(MANIFEST_PATH))
    p.add_argument('--list-sessions', action='store_true')
    args = p.parse_args()

    cache = init_cache(args.manifest)
    sessions = list_sessions(cache)
    if args.list_sessions:
        sid = pick_session_with_areas(sessions)
    else:
        sid = pick_session_with_areas(sessions)
    session = download_session(cache, sid)
    metrics_df = compute_latency_and_selectivity(session)
    metrics_df.to_csv(CACHE_DIR / f"metrics_session_{sid}.csv", index=False)
    area_stats = aggregate_by_area(metrics_df)
    area_stats.to_csv(CACHE_DIR / f"area_stats_session_{sid}.csv")
    plot_figures(area_stats)
    stats_tests(area_stats)
    print("Analysis complete. Outputs saved to:", CACHE_DIR)


if __name__ == "__main__":
    main()
