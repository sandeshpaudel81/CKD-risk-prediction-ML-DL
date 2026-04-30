import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
 
from sklearn.model_selection import train_test_split
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
from scipy.stats import pointbiserialr, spearmanr
from statsmodels.stats.multitest import multipletests

def create_folder():
    cwd = os.getcwd()
    folder_name = os.path.join(cwd, 'plots')    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def mann_whitney_u_test(df, ckd1, ckd0, CONTINUOUS_FEATURES):
    print("\nMann-Whitney U Test for Continuous Variables")
    print("  (CKD=1 rows vs CKD=0 rows)")
    
    mwu_results = []
    
    for feat in CONTINUOUS_FEATURES:
        if feat not in df.columns:
            continue
    
        grp1 = ckd1[feat].dropna()
        grp0 = ckd0[feat].dropna()
    
        # Mann-Whitney U — non-parametric, no normality assumption
        stat, p = mannwhitneyu(grp1, grp0, alternative='two-sided')
    
        # Effect size: rank-biserial correlation r = 1 - 2U/(n1*n2)
        n1, n2   = len(grp1), len(grp0)
        r_effect = 1 - (2 * stat) / (n1 * n2)
    
        # Descriptive stats
        m1, m0   = grp1.median(), grp0.median()
        q1_1, q3_1 = grp1.quantile(0.25), grp1.quantile(0.75)
        q1_0, q3_0 = grp0.quantile(0.25), grp0.quantile(0.75)
    
        mwu_results.append({
            'Feature'          : feat,
            'CKD=1 median(IQR)': f"{m1:.2f} ({q1_1:.2f}–{q3_1:.2f})",
            'CKD=0 median(IQR)': f"{m0:.2f} ({q1_0:.2f}–{q3_0:.2f})",
            'U statistic'      : round(stat, 1),
            'p-value'          : p,
            'Effect size (r)'  : round(r_effect, 3),
            'Significance'      : 'High' if p < 0.001 else ('Mid' if p < 0.01 else ('Low' if p < 0.05 else 'NS'))
        })
    
    cont_df = pd.DataFrame(mwu_results).sort_values('p-value')
    print(cont_df[['Feature','CKD=1 median(IQR)','CKD=0 median(IQR)',
                'p-value','Effect size (r)','Significance']].to_string(index=False))
    return cont_df

def fisher_chi2_test(df, ckd1, ckd0, BINARY_FEATURES):

    # Study of continuous variable between CKD and non-CKD
    print("\nChi-square Test for Binary Variables (with Odds Ratio)")
    print("  (Prevalence in CKD and Non-CKD)")
    
    binary_results = []
    
    for feat in BINARY_FEATURES:
        if feat not in df.columns:
            continue
    
        # Contingency table
        ct = pd.crosstab(df['CKD'], df[feat])
        if ct.shape != (2, 2):
            continue   # skip non-binary after crosstab
    
        a = ct.iloc[1, 1]   # CKD=1, feature=1
        b = ct.iloc[1, 0]   # CKD=1, feature=0
        c = ct.iloc[0, 1]   # CKD=0, feature=1
        d = ct.iloc[0, 0]   # CKD=0, feature=0
    
        # Fisher's exact (safer than chi2 for small cells)
        odds_ratio, p = fisher_exact([[a, b], [c, d]])
    
        # Prevalence in each group
        prev_ckd1 = ckd1[feat].mean() * 100
        prev_ckd0 = ckd0[feat].mean() * 100
    
        binary_results.append({
            'Feature'       : feat,
            'CKD=1 (%)': round(prev_ckd1, 1),
            'CKD=0 (%)': round(prev_ckd0, 1),
            'Odds Ratio'    : round(odds_ratio, 3),
            'p-value'       : p,
            'Significance'   : 'High' if p < 0.001 else ('Mid' if p < 0.01 else ('Low' if p < 0.05 else 'NS')),
            'Interpretation': ('Risk factor' if odds_ratio > 1 and p < 0.05
                            else 'Protective' if odds_ratio < 1 and p < 0.05
                            else 'Not significant')
        })
    
    bin_df = pd.DataFrame(binary_results).sort_values('p-value')
    print(bin_df.to_string(index=False))
    return bin_df

def spearman_correlation(df, CONTINUOUS_FEATURES, BINARY_FEATURES):
    # Study of correlation between all features and CKD
    print("\nSPEARMAN CORRELATION — All features vs CKD label")
    
    all_features = CONTINUOUS_FEATURES + BINARY_FEATURES
    corr_results = []
    
    for feat in all_features:
        if feat not in df.columns:
            continue
        r, p = spearmanr(df[feat], df['CKD'])
        corr_results.append({
            'Feature'    : feat,
            'Spearman r' : round(r, 4),
            'p-value'    : round(p, 6),
            'Significant': 'High' if p < 0.001 else ('Mid' if p < 0.01 else ('Low' if p < 0.05 else 'NS'))
        })
    
    corr_df = (pd.DataFrame(corr_results)
            .assign(abs_r=lambda x: x['Spearman r'].abs())
            .sort_values('abs_r', ascending=False)
            .drop(columns='abs_r'))
    
    print(corr_df.to_string(index=False))
    return corr_df

def main():

    create_folder()
    print("Statistical analysis of the dataset...")

    # Import cleaned datasets from files
    print("Importing cleaned datasets from files...")
    df = pd.DataFrame(
        np.load('./dataset/clean_data.npy', allow_pickle=True),
        columns=pd.read_csv('./dataset/column_names.csv').iloc[:,0].tolist()
    )

    # Dataset shape, number of unique patients, and distribution of CKD classes
    print(f"Dataset shape: {df.shape}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"CKD=1 rows: {df['CKD'].sum()} / {len(df)}")

    # All columns in the dataset
    print(f"All columns ({len(df.columns)}): {df.columns}")

    # List of continuous and categorical features for further analysis
    CONTINUOUS_FEATURES = [
        'diabetic_year', 'age_in_years', 'height_in_cm', 'weight_in_kg', 'BMI_in_kg_per_m2',
        'calorie_intake_per_day', 'bmi_change', 'weight_change',
        'comorbidity_score', 'lifestyle_score', 'ckd_years_so_far'
    ]
    
    BINARY_FEATURES = [
        'gender', 'job', 'family_diabetic_history',
        'diet_regulation', 'takes_diabetes_medicine', 'takes_insulin',
        'has_hypertension', 'has_heart_disease', 'sufficient_sleep',
        'sufficient_water', 'smoking_habit', 'takes_tobacco', 'walk_regularly',
        'proper_urination', 'urinary_infection', 'takes_pain_killer',
        'risk_behaviour'
    ]

    # Separate datasets for CKD=1 and CKD=0 for further analysis
    ckd1 = df[df['CKD'] == 1]
    ckd0 = df[df['CKD'] == 0]
    print(f"\nCKD=1 records: {len(ckd1)}  |  CKD=0 records: {len(ckd0)}")

    # Patient level aggregation for understanding patient-level characteristics and CKD development patterns
    patient_df = df.groupby('patient_id').agg(
        ckd_label           = ('CKD', 'max'),   # 1 if ever developed CKD
        onset_year          = ('CKD', lambda x: x[x==1].index[0]
                            if x.sum() > 0 else np.nan),
        mean_age            = ('age_in_years', 'mean'),
        mean_bmi            = ('BMI_in_kg_per_m2', 'mean'),
        mean_weight         = ('weight_in_kg', 'mean'),
        mean_calorie        = ('calorie_intake_per_day', 'mean'),
        mean_diabetic_year  = ('diabetic_year', 'mean'),
        mean_comorbidity    = ('comorbidity_score', 'mean'),
        mean_lifestyle      = ('lifestyle_score', 'mean'),
        gender              = ('gender', 'first'),
        family_history      = ('family_diabetic_history','first'),
        hypertension_ever   = ('has_hypertension', 'max'),
        heart_disease_ever  = ('has_heart_disease', 'max'),
        smoking_ever        = ('smoking_habit', 'max'),
        tobacco_ever        = ('takes_tobacco', 'max'),
        urinary_inf_ever    = ('urinary_infection', 'max'),
        pain_killer_ever    = ('takes_pain_killer', 'max'),
    ).reset_index()
    ckd_pts    = patient_df[patient_df['ckd_label'] == 1]
    non_ckd_pts= patient_df[patient_df['ckd_label'] == 0]
    print(f"\nPatient-level summary:")
    print(f"  Developed CKD : {len(ckd_pts)} patients")
    print(f"  Never CKD     : {len(non_ckd_pts)} patients")

    # Mann-Whitney U test for continuous features between CKD=1 and CKD=0 groups
    cont_df = mann_whitney_u_test(df, ckd1, ckd0, CONTINUOUS_FEATURES)

    # Chi-square test for binary features between CKD=1 and CKD=0 groups
    bin_df = fisher_chi2_test(df, ckd1, ckd0, BINARY_FEATURES)

    # Spearman correlation of all features with CKD label
    corr_df = spearman_correlation(df, CONTINUOUS_FEATURES, BINARY_FEATURES)

    # Multiple testing correction using Bonferroni and Holm-Bonferroni methods
    all_results = pd.concat([
        cont_df[['Feature','p-value']],
        bin_df[['Feature','p-value']]
    ], ignore_index=True)
    reject_b, p_adj_b, _, _ = multipletests(all_results['p-value'], method='bonferroni')
    reject_hb, p_adj_hb, _, _ = multipletests(all_results['p-value'], alpha=0.05, method='holm')
    all_results['p-adjusted_b'] = p_adj_b
    all_results['Significance_Bonferroni'] = reject_b
    all_results['p-adjusted_hb'] = p_adj_hb
    all_results['Significance_Holm'] = reject_hb
    print("\nCORRECTED p-values\n")
    print("\nBonferroni Vs. Holm-Bonferroni\n")
    print(all_results.sort_values('p-adjusted_hb').to_string(index=False))

    # Plot and save spearman correlation
    plt.figure(figsize=(10, 8))
    plot_corr = corr_df.set_index('Feature')['Spearman r']
    colors     = ['salmon' if r > 0 else 'steelblue' for r in plot_corr]
    plot_corr.sort_values().plot(kind='barh', color=colors, alpha=0.85)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.xlabel('Spearman r  (positive = higher value → more CKD)')
    plt.title('Spearman correlation of each feature with CKD label\n'
            'Red = risk factor  |  Blue = protective factor')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/stat_spearman_correlation.png', dpi=150, bbox_inches='tight')

    # Plotting odds ratio between two groups based on binary features
    sig_bin = bin_df[bin_df['p-value'] < 0.05].sort_values('Odds Ratio', ascending=False)
    if not sig_bin.empty:
        plt.figure(figsize=(9, 5))
        colors = ['salmon' if or_ > 1 else 'steelblue'
                for or_ in sig_bin['Odds Ratio']]
        plt.barh(sig_bin['Feature'], sig_bin['Odds Ratio'], color=colors, alpha=0.85)
        plt.axvline(1, color='black', linestyle='--', linewidth=0.8, label='OR=1 (no effect)')
        plt.xlabel('Odds Ratio')
        plt.title('Odds Ratio — significant binary features (p<0.05)\n'
                'OR > 1 = risk factor  |  OR < 1 = protective')
        plt.legend(fontsize=8)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/stat_odds_ratio.png', dpi=150, bbox_inches='tight')

    # Distribution of continuous features using box plot across two groups
    sig_cont = cont_df[cont_df['p-value'] < 0.05]['Feature'].tolist()
    if sig_cont:
        n_cols = 3
        n_rows = int(np.ceil(len(sig_cont) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=(14, n_rows * 3.5))
        axes = axes.flatten()
        for i, feat in enumerate(sig_cont):
            ax = axes[i]
            data = [ckd0[feat].dropna(), ckd1[feat].dropna()]
            bp   = ax.boxplot(data, patch_artist=True,
                            labels=['No CKD', 'CKD'],
                            medianprops=dict(color='black', linewidth=2))
            bp['boxes'][0].set_facecolor('steelblue')
            bp['boxes'][1].set_facecolor('salmon')
            ax.set_title(feat, fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            # Add p-value annotation
            p_val = cont_df[cont_df['Feature'] == feat]['p-value'].values[0]
            sig   = cont_df[cont_df['Feature'] == feat]['Significance'].values[0]
            ax.set_xlabel(f'p={p_val:.6f}  {sig}', fontsize=8)
        # Hide unused axes
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle('Distribution of significant continuous features: CKD vs No CKD',
                    fontsize=11, y=1.01)
        plt.tight_layout()
        plt.savefig('plots/stat_boxplots.png', dpi=150, bbox_inches='tight')

    print("\nFINAL RESULT\n")
 
    # Merge continuous and binary results into one table
    cont_summary = cont_df[['Feature','p-value','Effect size (r)','Significance']].copy()
    cont_summary['Test']  = 'Mann-Whitney U'
    cont_summary['Metric Type'] = 'Effect Size'
    cont_summary['Metric']= cont_summary['Effect size (r)'].astype(str)
    
    bin_summary  = bin_df[['Feature','p-value','Odds Ratio','Significance']].copy()
    bin_summary['Test']   = "Fisher's Exact"
    bin_summary['Metric Type'] = "Odds Ratio"
    bin_summary['Metric'] = bin_summary['Odds Ratio'].astype(str)
    
    summary_table = pd.concat([
        cont_summary[['Feature','Test','Metric Type','Metric','p-value','Significance']],
        bin_summary[['Feature','Test','Metric Type','Metric','p-value','Significance']]
    ], ignore_index=True).sort_values('p-value')
    
    print(summary_table.to_string(index=False))

if __name__ == "__main__":
    main()