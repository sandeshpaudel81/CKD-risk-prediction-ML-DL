import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.stats import mannwhitneyu, linregress
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from utils.common import create_folder

# Build patient-level summary with pre-onset features and label

# Coverters - who develop CKD at some point → use only pre-CKD years
# Stable - who never develop CKD → use all years

# mean and slope of continuous features
# proportion of years active for binary features
# static features (first value)

def build_patient_summary(df, CONTINUOUS_FEATS, BINARY_FEATS, STATIC_FEATS):
    records = []
    for pid, grp in df.groupby('patient_id'):
        grp = grp.sort_values('diabetic_year').reset_index(drop=True)

        ckd_ever = int(grp['CKD'].max())

        if ckd_ever == 1:
            # keep only pre-onset years (before first CKD year)
            first_ckd_idx = grp[grp['CKD'] == 1].index[0]
            pre = grp.loc[:first_ckd_idx - 1] if first_ckd_idx > 0 else grp.iloc[0:0]
        else:
            pre = grp

        if len(pre) < 2:
            # not enough years for slope — skip
            continue

        row = {'patient_id': pid, 'ckd_converter': ckd_ever}

        # mean of continuous features
        for feat in CONTINUOUS_FEATS:
            if feat in pre.columns:
                row[f'{feat}_mean'] = pre[feat].mean()

        # slope of continuous features (linear trend)
        x = np.arange(len(pre), dtype=float)
        for feat in CONTINUOUS_FEATS:
            if feat in pre.columns:
                slope = linregress(x, pre[feat].values.astype(float)).slope
                row[f'{feat}_slope'] = slope

        # binary feature: proportion of years active
        for feat in BINARY_FEATS:
            if feat in pre.columns:
                row[f'{feat}_prop'] = pre[feat].mean()

        # static features — first value
        for feat in STATIC_FEATS:
            if feat in pre.columns:
                row[feat] = pre[feat].iloc[0]

        # duration of pre-onset window
        row['pre_onset_years'] = len(pre)

        # causal indicators based on previous analysis 
        row['bmi_high']   = int(pre['bmi_change'].mean()     >= 0.3)
        row['poor_diet']  = int(pre['diet_regulation'].mean() < 0.5)
        row['insulin_high'] = int(pre['takes_insulin'].mean() >= 0.5)

        records.append(row)

    summary = pd.DataFrame(records).set_index('patient_id')
    return summary


# Cluster those patients who convert to CKD using their trajectory features (means and slopes).

def cluster_converters(summary_df, n_clusters=3, random_state=42):
    converters = summary_df[summary_df['ckd_converter'] == 1].copy()

    # drop label and any non-numeric cols
    feat_cols = [c for c in converters.columns
                 if c not in ['ckd_converter', 'pre_onset_years'] and converters[c].dtype != object]
    X = converters[feat_cols].dropna(axis=1)  # drop cols with NaN

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # silhouette score to find best k
    sil_scores = {}
    for k in range(2, 6):
        km  = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        lbl = km.fit_predict(X_sc)
        sil_scores[k] = silhouette_score(X_sc, lbl)
        print(f"  k={k}  silhouette={sil_scores[k]:.4f}")

    best_k = max(sil_scores, key=sil_scores.get)
    print(f"\n  Best k by silhouette: {best_k}  (score={sil_scores[best_k]:.4f})")

    km_final = KMeans(n_clusters=best_k, random_state=random_state, n_init=20)
    labels   = km_final.fit_predict(X_sc)

    converters = converters.loc[X.index].copy()
    converters['cluster'] = labels

    return converters, X, X_sc, feat_cols, best_k, sil_scores


# For each cluster, compute mean feature values and print a summary.

def profile_clusters(converters_clustered, feat_cols, results_out):
    print("\n══ Cluster Profiles ══")
    profile = converters_clustered.groupby('cluster')[feat_cols].mean()
    print(profile.T.to_string())

    # pre-onset window duration per cluster
    print ("\nPre-onset window duration (years) per cluster:")
    print(converters_clustered.groupby('cluster')['pre_onset_years'].describe())

    # causal exposure prevalence per cluster
    causal_flags = ['bmi_high', 'poor_diet', 'insulin_high']
    causal_flags = [f for f in causal_flags if f in converters_clustered.columns]
    if causal_flags:
        print("\nCausal Exposure Prevalence per Cluster:")
        print(converters_clustered.groupby('cluster')[causal_flags].mean().round(3).to_string())

    profile.T.to_csv(f'{results_out}/cluster_profiles.csv')
    return profile


# PCA plot of clusters 

def plot_pca_clusters(X_sc, labels, best_k, out_dir):
    pca   = PCA(n_components=2, random_state=42)
    X_2d  = pca.fit_transform(X_sc)
    ev    = pca.explained_variance_ratio_

    colours = cm.tab10(np.linspace(0, 0.5, best_k))
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in range(best_k):
        mask = labels == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=colours[c], label=f'Cluster {c}', alpha=0.7, edgecolors='k', s=60)
    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}% var)')
    ax.set_title('Converter Clusters — PCA Projection')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/pca_clusters.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  PCA cluster plot saved.")


# Radar chart comparing cluster profiles

def plot_radar_clusters(profile, out_dir):
    """
    Radar chart comparing cluster profiles across key mean features.
    """
    key_feats = [c for c in profile.columns if c.endswith('_mean') or c.endswith('_slope')]
    if len(key_feats) == 0:
        key_feats = profile.columns.tolist()[:8]

    sub = profile[key_feats].copy()
    # normalise each feature to [0,1] for radar
    sub = (sub - sub.min()) / (sub.max() - sub.min() + 1e-9)

    labels_feats = [f.replace('_mean','').replace('_slope',' slope').replace('_',' ') for f in key_feats]
    N = len(key_feats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colours = cm.tab10(np.linspace(0, 0.5, len(sub)))
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for idx, (cluster_id, row) in enumerate(sub.iterrows()):
        vals = row.tolist() + row.tolist()[:1]
        ax.plot(angles, vals, color=colours[idx], linewidth=2, label=f'Cluster {cluster_id}')
        ax.fill(angles, vals, color=colours[idx], alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels_feats, fontsize=8)
    ax.set_title('Cluster Feature Profiles (Normalised)', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(f'{out_dir}/radar_clusters.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Radar chart saved.")


# Trajectory plots of key features for each cluster

def plot_mean_trajectories(df, converters_clustered, out_dir):
    """
    For each cluster, plot the mean year-by-year trajectory of key features
    aligned relative to CKD onset (year -1, -2, ... from onset).
    """
    key_feats = ['weight_in_kg', 'weight_change', 'bmi_change',
                 'BMI_in_kg_per_m2', 'takes_insulin', 'diet_regulation']
    key_feats = [f for f in key_feats if f in df.columns]

    cluster_ids = sorted(converters_clustered['cluster'].unique())
    colours     = cm.tab10(np.linspace(0, 0.5, len(cluster_ids)))

    for feat in key_feats:
        fig, ax = plt.subplots(figsize=(8, 4))

        for cid, col in zip(cluster_ids, colours):
            pids = converters_clustered[converters_clustered['cluster'] == cid].index.tolist()
            traj_rows = []

            for pid in pids:
                grp = df[df['patient_id'] == pid].sort_values('diabetic_year')
                first_ckd = grp[grp['CKD'] == 1]['diabetic_year'].min()
                grp = grp[grp['diabetic_year'] < first_ckd].copy()
                grp['years_to_onset'] = grp['diabetic_year'] - first_ckd  # negative values
                traj_rows.append(grp[['years_to_onset', feat]])

            if traj_rows:
                traj_df = pd.concat(traj_rows)
                mean_traj = traj_df.groupby('years_to_onset')[feat].mean()
                ax.plot(mean_traj.index, mean_traj.values, marker='o',
                        color=col, label=f'Cluster {cid}', linewidth=2)

        ax.axvline(0, color='red', linestyle='--', linewidth=1, label='CKD onset')
        ax.set_xlabel('Years Relative to CKD Onset')
        ax.set_ylabel(feat.replace('_', ' ').title())
        ax.set_title(f'Mean Pre-Onset Trajectory: {feat.replace("_", " ").title()}')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fname = feat.replace(' ', '_')
        plt.savefig(f'{out_dir}/traj_{fname}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Mean trajectory plots saved for: {key_feats}")

# Coverter vs Stable slope comparison 

def plot_converter_vs_stable(summary_df, plots_out, results_out):
    """
    Compare slope features between converters and stable non-CKD patients.
    """
    slope_feats = [c for c in summary_df.columns if c.endswith('_slope')]
    if not slope_feats:
        return

    converters = summary_df[summary_df['ckd_converter'] == 1]
    stable     = summary_df[summary_df['ckd_converter'] == 0]

    rows = []
    for feat in slope_feats:
        c_vals = converters[feat].dropna()
        s_vals = stable[feat].dropna()
        if len(c_vals) < 3 or len(s_vals) < 3:
            continue
        stat, p = mannwhitneyu(c_vals, s_vals, alternative='two-sided')
        rows.append({
            'feature'         : feat,
            'converter_mean'  : c_vals.mean(),
            'stable_mean'     : s_vals.mean(),
            'p_value'         : p,
            'significant'     : p < 0.05
        })

    result_df = pd.DataFrame(rows).sort_values('p_value')
    print("\n══ Converter vs Stable — Slope Features (Mann-Whitney U) ══")
    print(result_df.to_string(index=False))
    result_df.to_csv(f'{results_out}/converter_vs_stable_slopes.csv', index=False)

    # bar chart of slope differences
    sig = result_df[result_df['significant']]
    if len(sig) == 0:
        print("  No significant slope differences found.")
        return

    fig, ax = plt.subplots(figsize=(8, max(3, len(sig) * 0.6)))
    y     = np.arange(len(sig))
    width = 0.35
    ax.barh(y - width/2, sig['converter_mean'], width, label='Converters', color='tomato')
    ax.barh(y + width/2, sig['stable_mean'],    width, label='Stable',     color='steelblue')
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace('_slope', '').replace('_', ' ') for f in sig['feature']])
    ax.set_xlabel('Mean Slope')
    ax.set_title('Significant Slope Differences: Converters vs Stable')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plots_out}/slope_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Slope comparison chart saved.")

    return result_df


# Silhouette score plot for different k values

def plot_silhouette(sil_scores, out_dir):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(list(sil_scores.keys()), list(sil_scores.values()), marker='o', color='steelblue')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score vs k (Converter Group)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/silhouette_scores.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Silhouette score plot saved.")


# Cluster size summary

def print_cluster_summary(converters_clustered):
    print("\n══ Cluster Size Summary ══")
    counts = converters_clustered['cluster'].value_counts().sort_index()
    for cid, cnt in counts.items():
        pct = cnt / len(converters_clustered) * 100
        print(f"  Cluster {cid}: {cnt} patients ({pct:.1f}%)")


# MAIN FUNCTION

def main(base_dir):
    create_folder(base_dir, ['plots/clustering', 'results/clustering'])
    PLOTS_OUT = f'{base_dir}/plots/clustering'
    RESULTS_OUT = f'{base_dir}/results/clustering'

    # Features used to characterise patient trajectories at patient level
    CONTINUOUS_FEATS = [
        'weight_in_kg', 'BMI_in_kg_per_m2', 'bmi_change', 'weight_change'
    ]
    BINARY_FEATS = [
        'takes_insulin', 'urinary_infection', 'has_heart_disease',
        'takes_pain_killer', 'has_hypertension', 'sufficient_sleep',
        'urinary_infection', 'takes_pain_killer',
        'smoking_habit', 'takes_tobacco',
    ]
    STATIC_FEATS = ['gender', 'job', 'family_diabetic_history', 'calorie_intake_per_day']

    print("Loading datasets...")
    df = pd.DataFrame(
        np.load(f'{base_dir}/dataset/clean_data.npy', allow_pickle=True),
        columns=pd.read_csv(f'{base_dir}/dataset/column_names.csv').iloc[:, 0].tolist()
    )

    print(f"  Total records: {len(df)}  |  Patients: {df['patient_id'].nunique()}")

    # Build patient-level summary
    print("\nBuilding patient-level pre-onset summaries...")
    summary_df = build_patient_summary(df, CONTINUOUS_FEATS, BINARY_FEATS, STATIC_FEATS)
    n_conv   = (summary_df['ckd_converter'] == 1).sum()
    n_stable = (summary_df['ckd_converter'] == 0).sum()
    print(f"  Converters : {n_conv}")
    print(f"  Stable     : {n_stable}")
    summary_df.to_csv(f'{RESULTS_OUT}/patient_summary.csv')

    # Converter vs Stable slope comparison
    print("\nComparing converter vs stable trajectories...")
    plot_converter_vs_stable(summary_df, PLOTS_OUT, RESULTS_OUT)

    # Cluster converters
    print("\nClustering converter group...")
    converters_clustered, X, X_sc, feat_cols, best_k, sil_scores = cluster_converters(summary_df)
    print_cluster_summary(converters_clustered)
    converters_clustered.to_csv(f'{RESULTS_OUT}/converters_with_clusters.csv')

    # Profile clusters
    profile = profile_clusters(converters_clustered, feat_cols, RESULTS_OUT)

    # Visualisations
    print("\nGenerating visualisations...")

    plot_silhouette(sil_scores, PLOTS_OUT)
    plot_pca_clusters(X_sc, converters_clustered['cluster'].values, best_k, PLOTS_OUT)
    plot_radar_clusters(profile, PLOTS_OUT)
    plot_mean_trajectories(df, converters_clustered, PLOTS_OUT)

    print("\nDone. Results saved to 'results/clustering/' and 'plots/clustering/'")