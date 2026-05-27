import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

COMPARE_FEATS = [
    'weight_in_kg',
    'BMI_in_kg_per_m2',
    'comorbidity_score',
    'lifestyle_score',
    'takes_insulin'
]

WINDOW = 5

def create_folder():
    os.makedirs("plots/transition_analysis", exist_ok=True)
    os.makedirs("transition_results", exist_ok=True)


# Extract converters and stable non-CKD

def extract_groups(df):
    converters=[]
    stable=[]
    for pid, g in df.groupby("patient_id"):

        g = g.sort_values("diabetic_year")
        labels = g["CKD"].values
        transition=np.where(
            (labels[:-1]==0)&(labels[1:]==1)
        )[0]

        # converter
        if len(transition)>0:
            onset = transition[0]+1
            if onset >= WINDOW:
                converters.append(
                    g.iloc[onset-WINDOW:onset]
                )
        # stable no CKD
        elif labels.sum()==0:
            if len(g)>=WINDOW:
                stable.append(
                    g.iloc[-WINDOW:]
                )
    return converters,stable


# Statistical comparison

def compare_trajectory_features(converters,stable):
    rows=[]
    for feat in COMPARE_FEATS:
        conv_slopes=[]
        stable_slopes=[]
        for c in converters:
            slope=np.polyfit(
                range(WINDOW),
                c[feat],
                1
            )[0]
            conv_slopes.append(slope)
        for s in stable:
            slope =np.polyfit(
                range(WINDOW),
                s[feat],
                1
            )[0]
            stable_slopes.append(slope)
        stat,p = mannwhitneyu(
            conv_slopes,
            stable_slopes
        )

        rows.append({
            "feature":feat,
            "converter_mean_slope":
                np.mean(conv_slopes),
            "stable_mean_slope":
                np.mean(stable_slopes),
            "p_value":p
        })
    out = pd.DataFrame(rows)
    out.to_csv(
        "transition_results/slope_comparison.csv",
        index=False
    )
    print(out.sort_values("p_value"))
    return out


# Event aligned trajectories

def plot_event_alignment(converters, stable, feat):
    conv = np.array([
        x[feat].values
        for x in converters
    ])
    stable = np.array([
        x[feat].values
        for x in stable
    ])
    years=["t-4","t-3","t-2","t-1","t0"]

    plt.figure(figsize=(6,4))
    plt.plot(
        years,
        conv.mean(0),
        label="Converters"
    )
    plt.plot(
        years,
        stable.mean(0),
        label="Stable no CKD"
    )
    plt.fill_between(
        years,
        conv.mean(0)-conv.std(0),
        conv.mean(0)+conv.std(0),
        alpha=.2
    )
    plt.ylabel(feat)
    plt.title(
        f"{feat}: before CKD onset"
    )
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(
        f"plots/transition_analysis/{feat}.png",
        dpi=150
    )
    plt.close()

# Main

def main():
    create_folder()
    train_df=pd.DataFrame(
        np.load(
            "dataset/train_df.npy",
            allow_pickle=True
        ),
        columns=pd.read_csv(
            "dataset/column_names.csv"
        ).iloc[:,0]
    )
    val_df=pd.DataFrame(
        np.load(
            "dataset/val_df.npy",
            allow_pickle=True
        ),
        columns=train_df.columns
    )
    test_df=pd.DataFrame(
        np.load(
            "dataset/test_df.npy",
            allow_pickle=True
        ),
        columns=train_df.columns
    )
    df=pd.concat([
        train_df,
        val_df,
        test_df
    ])
    converters,stable=extract_groups(df)
    print(
        "Converters:",
        len(converters)
    )
    print(
        "Stable:",
        len(stable)
    )
    compare_trajectory_features(
        converters,
        stable
    )
    for feat in COMPARE_FEATS:
        plot_event_alignment(
            converters,
            stable,
            feat
        )
    print("Done.")

if __name__=="__main__":
    main()