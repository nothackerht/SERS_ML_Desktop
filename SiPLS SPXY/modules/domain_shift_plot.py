# modules/domain_shift_plot.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import ks_2samp, ttest_ind

def plot_domain_shift(train_meta_path, test_meta_path, output_dir):
    """
    Generate violin + swarm plots for target_SI, HGS_pp_avg,
    and ADF_pp_avg using training data only if test values are missing.
    Also compute and print statistical comparison metrics where possible.
    """
    # Load metadata
    train_df = pd.read_csv(train_meta_path)
    test_df = pd.read_csv(test_meta_path)

    # Filter to DM1 and Control
    train_df = train_df[train_df['Type'].isin(['DM1', 'Control'])].copy()
    test_df = test_df[test_df['Type'].isin(['DM1', 'Control'])].copy()

    # Assign 0 to Control SI values
    train_df.loc[train_df['Type'] == 'Control', 'target_SI'] = 0
    test_df.loc[test_df['Type'] == 'Control', 'target_SI'] = 0

    # Label data source
    train_df['Set'] = 'Train'
    test_df['Set'] = 'Test'

    # Define targets and y-limits
    y_limits = {
        'target_SI': (-.1, 1),
        'HGS_pp_avg': (0, 120),
        'ADF_pp_avg': (0, 120)
    }

    stats_records = []
    os.makedirs(output_dir, exist_ok=True)

    for metric, y_lim in y_limits.items():
        train_vals = train_df[metric].dropna()
        test_vals = test_df[metric].dropna()

        # Skip both plot and stats if training set lacks data
        if train_vals.empty:
            print(f"⚠️ Skipping {metric}: no training data.")
            continue

        # Create plot from train only if test data is missing
        plot_df = train_df.copy() if test_vals.empty else pd.concat([train_df, test_df], ignore_index=True)

        plt.figure(figsize=(6, 5))
        subset_df = plot_df[['Sample_ID', 'Type', 'Set', metric]].rename(columns={metric: 'Value'})
        subset_df['Metric'] = metric

        ax = sns.violinplot(
            data=subset_df,
            x="Set",
            y="Value",
            color="lightgrey",
            inner=None,
            linewidth=1.2
        )

        for label, color in [('Control', 'red'), ('DM1', 'blue')]:
            df = subset_df[subset_df['Type'] == label].copy()
            df = df.dropna(subset=['Value'])  # Ensure no NaNs in y

            if not df.empty:
                sns.swarmplot(
                    data=df,
                    x="Set",
                    y="Value",
                    color=color,
                    label=label,
                    dodge=True,
                    linewidth=0.5
                )


        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')
        ax.set_title(f"{metric} Distribution by Dataset")
        ax.set_ylim(y_lim)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Value")
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"domain_shift_{metric}.png")
        plt.savefig(plot_path, dpi=300)
        plt.show()
        plt.close()

        print(f"✅ Saved domain shift plot: {plot_path}")

        # Skip stats if test is missing
        if test_vals.empty:
            print(f"⚠️ Skipping statistics for {metric}: no test data.")
            continue

        # === Compute statistical summaries ===
        train_mean, test_mean = train_vals.mean(), test_vals.mean()
        train_std, test_std = train_vals.std(), test_vals.std()
        train_iqr = train_vals.quantile(0.75) - train_vals.quantile(0.25)
        test_iqr = test_vals.quantile(0.75) - test_vals.quantile(0.25)
        ks_stat, ks_p = ks_2samp(train_vals, test_vals)
        t_stat, t_p = ttest_ind(train_vals, test_vals, equal_var=False)
        cohen_d = (train_mean - test_mean) / np.sqrt((train_std**2 + test_std**2) / 2)
        train_skew, test_skew = train_vals.skew(), test_vals.skew()
        train_kurt, test_kurt = train_vals.kurtosis(), test_vals.kurtosis()

        print("\n📊 Statistical Summary for", metric)
        print("----------------------------------")
        print(f"Train Mean: {train_mean:.4f}, Test Mean: {test_mean:.4f}")
        print(f"Train Std: {train_std:.4f}, Test Std: {test_std:.4f}")
        print(f"Train IQR: {train_iqr:.4f}, Test IQR: {test_iqr:.4f}")
        print(f"KS Statistic: {ks_stat:.4f}, p-value: {ks_p:.4e}")
        print(f"T Statistic: {t_stat:.4f}, p-value: {t_p:.4e}")
        print(f"Cohen's d: {cohen_d:.4f}")
        print(f"Train Skew: {train_skew:.4f}, Kurtosis: {train_kurt:.4f}")
        print(f"Test Skew: {test_skew:.4f}, Kurtosis: {test_kurt:.4f}")

        stats_records.append({
            'Metric': metric,
            'Train Mean': train_mean,
            'Test Mean': test_mean,
            'Train Std': train_std,
            'Test Std': test_std,
            'Train IQR': train_iqr,
            'Test IQR': test_iqr,
            'KS Statistic': ks_stat,
            'KS p-value': ks_p,
            'T Statistic': t_stat,
            'T p-value': t_p,
            "Cohen's d": cohen_d,
            'Train Skew': train_skew,
            'Test Skew': test_skew,
            'Train Kurtosis': train_kurt,
            'Test Kurtosis': test_kurt
        })

    stats_df = pd.DataFrame(stats_records)
    stats_output_path = os.path.join(output_dir, "domain_shift_statistics_summary.xlsx")
    stats_df.to_excel(stats_output_path, index=False)
    print(f"📊 Saved statistical summary: {stats_output_path}")
