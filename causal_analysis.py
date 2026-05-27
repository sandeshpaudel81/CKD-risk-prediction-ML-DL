import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

nx.algorithms.d_separated = nx.algorithms.d_separation.is_d_separator
nx.d_separated = nx.algorithms.d_separation.is_d_separator
from dowhy import CausalModel

warnings.filterwarnings('ignore')

os.makedirs('plots/causal', exist_ok=True)
os.makedirs('causal_results', exist_ok=True)


# ── Config ────────────────────────────────────────────────────────────────────

EXPOSURES = {
    'insulin_high'   : 'takes_insulin',
    'weight_gaining' : 'weight_change',
    'long_diabetes'  : 'diabetic_year',
    'high_calorie'   : 'calorie_intake_per_day',
}

# Base confounders — mediators and colliders excluded
BASE_CONFOUNDERS = [
    'age_in_years', 'gender', 'family_diabetic_history',
]

# Added when not the exposure itself
EXTRA_CONFOUNDERS = {
    'insulin_high'   : ['BMI_in_kg_per_m2', 'diabetic_year_mean'],
    'weight_gaining' : ['diabetic_year_mean'],
    'long_diabetes'  : ['BMI_in_kg_per_m2'],
    'high_calorie'   : ['diabetic_year_mean'],
}

OUTCOME = 'ckd_ever'


# ── Step 1: Build patient-level pre-onset summary ─────────────────────────────

def build_patient_summary(data_df):
    records = []
    for pid, group in data_df.groupby('patient_id'):
        pre   = group[group['CKD'] == 0].sort_values('diabetic_year')
        label = int(group['CKD'].max())

        if len(pre) == 0:
            continue

        row = {'patient_id': pid, OUTCOME: label}

        # continuous — mean across pre-onset years
        for col in ['age_in_years', 'BMI_in_kg_per_m2', 'weight_change',
                    'calorie_intake_per_day', 'takes_insulin',
                    'has_hypertension', 'has_heart_disease',
                    'walk_regularly', 'sufficient_sleep']:
            row[col] = pre[col].mean()

        # diabetes duration — max of pre-onset years
        row['diabetic_year_mean'] = pre['diabetic_year'].max()

        # static features — first value
        for col in ['gender', 'family_diabetic_history', 'height_in_cm', 'job']:
            row[col] = pre[col].iloc[0]

        records.append(row)

    df = pd.DataFrame(records)
    print(f"Patient summary: {len(df)} patients | CKD=1: {df[OUTCOME].sum()} | CKD=0: {(df[OUTCOME]==0).sum()}")
    return df


# ── Step 2: Binarise exposures ────────────────────────────────────────────────

def binarise_exposures(df):
    df['insulin_high']   = (df['takes_insulin']          >= 0.5).astype(int)
    df['weight_gaining'] = (df['weight_change']           > 0.0).astype(int)
    df['long_diabetes']  = (df['diabetic_year_mean']      >= 10 ).astype(int)
    df['high_calorie']   = (df['calorie_intake_per_day']  >= df['calorie_intake_per_day'].median()).astype(int)

    print("\nExposure distributions:")
    for t in EXPOSURES:
        n1 = df[t].sum()
        print(f"  {t}: {n1} exposed ({100*n1/len(df):.1f}%) | CKD rate exposed: "
              f"{df[df[t]==1][OUTCOME].mean():.3f} | unexposed: {df[df[t]==0][OUTCOME].mean():.3f}")
    return df


# ── Step 3: Build DAG ─────────────────────────────────────────────────────────

def build_dag(treatment):
    G = nx.DiGraph()

    # Confounders → treatment and outcome
    for conf in ['age_in_years', 'gender', 'family_diabetic_history',
                 'BMI_in_kg_per_m2', 'diabetic_year_mean']:
        G.add_edge(conf, treatment)
        G.add_edge(conf, OUTCOME)

    # Treatment → outcome (direct effect)
    G.add_edge(treatment, OUTCOME)

    # Mediators — on causal path, NOT conditioned on
    if treatment == 'insulin_high':
        G.add_edge(treatment, 'urinary_infection_med')
        G.add_edge('urinary_infection_med', OUTCOME)
        G.add_edge('diabetic_year_mean', treatment)

    if treatment in ('weight_gaining', 'high_calorie'):
        G.add_edge(treatment, 'bmi_med')
        G.add_edge('bmi_med', OUTCOME)

    if treatment == 'long_diabetes':
        G.add_edge('age_in_years', treatment)
        G.add_edge('family_diabetic_history', treatment)

    return G


# ── Step 4: Run DoWhy for one exposure ───────────────────────────────────────

def run_causal(df, treatment, verbose=True):
    confounders = BASE_CONFOUNDERS + EXTRA_CONFOUNDERS.get(treatment, [])
    confounders = [c for c in confounders if c in df.columns]

    cols_needed = [treatment, OUTCOME] + confounders
    data        = df[cols_needed].dropna()

    G = build_dag(treatment)

    model = CausalModel(
        data      = data,
        treatment = treatment,
        outcome   = OUTCOME,
        graph     = nx.drawing.nx_pydot.to_pydot(G).to_string(),
    )

    identified = model.identify_effect(proceed_when_unidentifiable=True)
    if verbose:
        print(f"\n  Identified estimand:\n{identified}")

    results = {}

    # Propensity score matching
    try:
        est = model.estimate_effect(
            identified,
            method_name          = 'backdoor.propensity_score_matching',
            target_units         = 'ate',
            method_params        = {'number_of_matched_units': 50},
        )
        results['psm'] = float(est.value)
        print(f"  PSM   ATE: {est.value:.4f}")
    except Exception as e:
        results['psm'] = np.nan
        print(f"  PSM   failed: {e}")

    # Linear regression adjustment
    try:
        est_lr = model.estimate_effect(
            identified,
            method_name = 'backdoor.linear_regression',
            target_units = 'ate',
        )
        results['lr'] = float(est_lr.value)
        print(f"  LR    ATE: {est_lr.value:.4f}")
    except Exception as e:
        results['lr'] = np.nan
        print(f"  LR    failed: {e}")

    # Propensity score weighting (IPW)
    try:
        est_ipw = model.estimate_effect(
            identified,
            method_name  = 'backdoor.propensity_score_weighting',
            target_units = 'ate',
        )
        results['ipw'] = float(est_ipw.value)
        print(f"  IPW   ATE: {est_ipw.value:.4f}")
    except Exception as e:
        results['ipw'] = np.nan
        print(f"  IPW   failed: {e}")

    results['mean_ate'] = float(np.nanmean(list(results.values())))
    print(f"  Mean ATE across methods: {results['mean_ate']:.4f}")

    # Refutation tests — use LR estimate as it's most stable
    try:
        est_ref = model.estimate_effect(
            identified,
            method_name  = 'backdoor.linear_regression',
            target_units = 'ate',
        )

        ref_rc = model.refute_estimate(
            identified, est_ref,
            method_name = 'random_common_cause',
        )
        ref_pl = model.refute_estimate(
            identified, est_ref,
            method_name    = 'placebo_treatment_refuter',
            placebo_type   = 'permute',
            num_simulations= 100,
        )
        ref_sub = model.refute_estimate(
            identified, est_ref,
            method_name    = 'data_subset_refuter',
            subset_fraction= 0.8,
            num_simulations= 100,
        )

        results['ref_random_cause'] = float(ref_rc.new_effect)
        results['ref_placebo']      = float(ref_pl.new_effect)
        results['ref_subset']       = float(ref_sub.new_effect)

        print(f"\n  Refutation results:")
        print(f"    Random common cause : new ATE = {ref_rc.new_effect:.4f}  (should ≈ original)")
        print(f"    Placebo treatment   : new ATE = {ref_pl.new_effect:.4f}  (should ≈ 0)")
        print(f"    Data subset         : new ATE = {ref_sub.new_effect:.4f}  (should ≈ original)")

        # Robustness: placebo near 0, random_cause and subset near original LR ATE
        lr_ate = results['lr']
        results['robust'] = (
            abs(results['ref_placebo'])      < 0.05 and
            abs(results['ref_random_cause'] - lr_ate) < 0.05 and
            abs(results['ref_subset']       - lr_ate) < 0.05
        )
        print(f"    Robust: {results['robust']}")

    except Exception as e:
        print(f"  Refutation failed: {e}")
        results['ref_random_cause'] = np.nan
        results['ref_placebo']      = np.nan
        results['ref_subset']       = np.nan
        results['robust']           = False

    return results


# ── Step 5: Comparison table ──────────────────────────────────────────────────

SPEARMAN = {
    'insulin_high'   : 0.125,
    'weight_gaining' : -0.065,
    'long_diabetes'  : 0.125,
    'high_calorie'   : 0.067,
}

SHAP_PREONSET = {
    'insulin_high'   : 0.260,
    'weight_gaining' : 0.361,
    'long_diabetes'  : 0.426,
    'high_calorie'   : 0.177,
}

def build_comparison_table(all_results):
    rows = []
    for treatment, res in all_results.items():
        rows.append({
            'Exposure'        : treatment,
            'Spearman ρ'      : SPEARMAN.get(treatment, np.nan),
            'SHAP (pre-onset)': SHAP_PREONSET.get(treatment, np.nan),
            'ATE (PSM)'       : round(res.get('psm',      np.nan), 4),
            'ATE (LR)'        : round(res.get('lr',       np.nan), 4),
            'ATE (IPW)'       : round(res.get('ipw',      np.nan), 4),
            'Mean ATE'        : round(res.get('mean_ate', np.nan), 4),
            'Placebo ATE'     : round(res.get('ref_placebo', np.nan), 4),
            'Robust'          : res.get('robust', False),
        })
    df = pd.DataFrame(rows)
    print("\n\n══════ Comparison Table ══════")
    print(df.to_string(index=False))
    df.to_csv('causal_results/comparison_table.csv', index=False)
    return df


# ── Step 6: Visualisations ────────────────────────────────────────────────────

def plot_dag(treatment, out_dir='plots/causal'):
    G   = build_dag(treatment)
    pos = nx.spring_layout(G, seed=42)
    colors = []
    for node in G.nodes():
        if node == treatment:
            colors.append('steelblue')
        elif node == OUTCOME:
            colors.append('tomato')
        elif 'med' in node:
            colors.append('gold')
        else:
            colors.append('lightgrey')

    plt.figure(figsize=(9, 6))
    nx.draw_networkx(G, pos, node_color=colors, node_size=1800,
                     font_size=8, arrows=True, arrowsize=20,
                     edge_color='grey', width=1.5)
    plt.title(f'Causal DAG — {treatment}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/dag_{treatment}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_ate_comparison(comparison_df, out_dir='plots/causal'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ATE across methods
    x    = np.arange(len(comparison_df))
    w    = 0.25
    exps = comparison_df['Exposure'].tolist()
    axes[0].bar(x - w, comparison_df['ATE (PSM)'], w, label='PSM',   color='steelblue')
    axes[0].bar(x,     comparison_df['ATE (LR)'],  w, label='LR',    color='darkorange')
    axes[0].bar(x + w, comparison_df['ATE (IPW)'], w, label='IPW',   color='seagreen')
    axes[0].axhline(0, color='black', lw=0.8, linestyle='--')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(exps, rotation=15)
    axes[0].set_ylabel('ATE')
    axes[0].set_title('Causal ATE — Three Estimation Methods')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # SHAP vs Mean ATE
    axes[1].scatter(comparison_df['SHAP (pre-onset)'], comparison_df['Mean ATE'],
                    s=120, color='steelblue', zorder=3)
    for _, row in comparison_df.iterrows():
        axes[1].annotate(row['Exposure'],
                         (row['SHAP (pre-onset)'], row['Mean ATE']),
                         textcoords='offset points', xytext=(6, 4), fontsize=9)
    axes[1].axhline(0, color='black', lw=0.8, linestyle='--')
    axes[1].set_xlabel('Mean |SHAP| (Pre-onset)')
    axes[1].set_ylabel('Mean Causal ATE')
    axes[1].set_title('SHAP Importance vs Causal Effect')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/ate_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_refutation(all_results, out_dir='plots/causal'):
    treatments = list(all_results.keys())
    lr_ates    = [all_results[t].get('lr',          np.nan) for t in treatments]
    rc_ates    = [all_results[t].get('ref_random_cause', np.nan) for t in treatments]
    pl_ates    = [all_results[t].get('ref_placebo', np.nan) for t in treatments]
    sub_ates   = [all_results[t].get('ref_subset',  np.nan) for t in treatments]

    x = np.arange(len(treatments))
    w = 0.2
    plt.figure(figsize=(10, 5))
    plt.bar(x - 1.5*w, lr_ates,  w, label='Original LR ATE', color='steelblue')
    plt.bar(x - 0.5*w, rc_ates,  w, label='Random Cause',    color='darkorange')
    plt.bar(x + 0.5*w, pl_ates,  w, label='Placebo',         color='tomato')
    plt.bar(x + 1.5*w, sub_ates, w, label='Data Subset',     color='seagreen')
    plt.axhline(0, color='black', lw=0.8, linestyle='--')
    plt.xticks(x, treatments, rotation=15)
    plt.ylabel('ATE')
    plt.title('Refutation Test Results')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/refutation.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading dataset...")
    train_df = pd.DataFrame(
        np.load('dataset/train_df.npy', allow_pickle=True),
        columns=pd.read_csv('dataset/column_names.csv').iloc[:, 0].tolist()
    )
    val_df = pd.DataFrame(
        np.load('dataset/val_df.npy', allow_pickle=True),
        columns=train_df.columns
    )
    test_df = pd.DataFrame(
        np.load('dataset/test_df.npy', allow_pickle=True),
        columns=train_df.columns
    )
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"Full dataset: {len(full_df)} rows | {full_df['patient_id'].nunique()} patients")

    print("\nBuilding patient-level pre-onset summary...")
    patient_df = build_patient_summary(full_df)

    print("\nBinarising exposures...")
    patient_df = binarise_exposures(patient_df)

    print("\nRunning causal analysis...")
    all_results = {}
    for treatment in EXPOSURES:
        print(f"\n{'═'*50}")
        print(f"Exposure: {treatment}")
        print(f"{'═'*50}")
        all_results[treatment] = run_causal(patient_df, treatment)
        plot_dag(treatment)

    comparison_df = build_comparison_table(all_results)

    print("\nSaving visualisations...")
    plot_ate_comparison(comparison_df)
    plot_refutation(all_results)

    print("\nDone. Results saved to causal_results/ and plots/causal/")


if __name__ == "__main__":
    main()