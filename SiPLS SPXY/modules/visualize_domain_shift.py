# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:37:34 2025

@author: spect
"""

# modules/domain_shift_plot.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_domain_shift(train_meta_path, test_meta_path, output_dir):
    """
    Visualize target_SI, HGS_pp_avg, and ADF_pp_avg distributions
    between training and test metadata.
    """
    # Load metadata
    train_df = pd.read_csv(train_meta_path)
    test_df = pd.read_csv(test_meta_path)

    # Filter only DM1 and Control samples
    train_df = train_df[train_df['Type'].isin(['DM1', 'Control'])].copy()
    test_df = test_df[test_df['Type'].isin(['DM1', 'Control'])].copy()

    # Label data source
    train_df['Set'] = 'Train'
    test_df['Set'] = 'Test'

    # Combine for comparison
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # Melt for plotting
    melted = combined_df.melt(
        id_vars=['Sample_ID', 'Type', 'Set'],
        value_vars=['target_SI', 'HGS_pp_avg', 'ADF_pp_avg'],
        var_name='Metric',
        value_name='Value'
    )

    # Create violin + swarm plot
    sns.set(style="whitegrid", font_scale=1.2)
    g = sns.catplot(
        data=melted,
        x="Set",
        y="Value",
        hue="Type",
        col="Metric",
        kind="violin",
        split=True,
        inner=None,
        height=5,
        aspect=1
    )

    g.map_dataframe(
        sns.swarmplot, x="Set", y="Value", hue="Type",
        dodge=True, palette="dark:.3", linewidth=0.5
    )

    # Clean up labels
    for ax in g.axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("Value")

    g.set_titles("{col_name}")
    g.set_axis_labels("Dataset", "Value")
    g.tight_layout()

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "domain_shift_violin_plot.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Domain shift plot saved to: {output_path}")
