import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

# =====================================================================
# SHARED CONFIG
# =====================================================================
peptide = 'LIVAGK_tripled_time_new_ff'


# =====================================================================
# PART 1: HYDROGEN BOND NETWORK + HEATMAPS (First vs Second Half)
# =====================================================================

def parse_hbond_file(filepath):
    """Parse cpptraj-style inter-molecular hydrogen bond output."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # Expected columns:
            # Acceptor  DonorH  Donor  Frames  Frac  AvgDist  AvgAng
            if len(parts) < 7:
                continue
            try:
                acc_full = parts[0]   # e.g. ILE_3@O
                donorh   = parts[1]   # e.g. VAL_12@H
                donor    = parts[2]   # e.g. VAL_12@N
                frames   = int(parts[3])
                frac     = float(parts[4])
                avgdist  = float(parts[5])
                avgang   = float(parts[6])
            except (ValueError, IndexError):
                continue

            # Extract residue name: "ILE_3" from "ILE_3@O"
            acc_res = acc_full.split('@')[0]
            don_res = donor.split('@')[0]

            records.append({
                'Acc': acc_res,
                'Don': don_res,
                'Frames': frames,
                'Frac': frac,
                'AvgDist': avgdist,
                'AvgAng': avgang
            })
    return pd.DataFrame(records)


# --- Residue short-name mapping ---
AA_MAP = {'LEU': 'L', 'ILE': 'I', 'VAL': 'V', 'ALA': 'A',
           'GLY': 'G', 'PRO': 'P', 'PHE': 'F', 'TRP': 'W',
           'MET': 'M', 'SER': 'S', 'THR': 'T', 'CYS': 'C',
           'TYR': 'Y', 'HIS': 'H', 'ASP': 'D', 'GLU': 'E',
           'ASN': 'N', 'GLN': 'Q', 'LYS': 'K', 'ARG': 'R'}


def get_short_name(name):
    res, num = name.split('_')
    return f"{AA_MAP.get(res, res)}{num}"


# --- Define two peptide groups (shared across both halves) ---
group_a_labels = ["L2", "I3", "V4", "A5"]
group_b_labels = ["L10", "I11", "V12", "A13"]
all_labels = group_a_labels + group_b_labels

# --- Circular layout (shared so nodes sit in the same place in both plots) ---
theta = np.linspace(0, 2 * np.pi, len(all_labels), endpoint=False)
pos = {res: (np.cos(t), np.sin(t)) for res, t in zip(all_labels, theta)}

color_a = "#E74C3C"
color_b = "#3498DB"


def build_heatmap_matrix(df, donor_labels, acceptor_labels):
    """Build a 4x4 matrix of Frac values: rows = donors, cols = acceptors."""
    matrix = np.zeros((len(donor_labels), len(acceptor_labels)))
    for _, row in df.iterrows():
        don = row['Don_short']
        acc = row['Acc_short']
        if don in donor_labels and acc in acceptor_labels:
            i = donor_labels.index(don)
            j = acceptor_labels.index(acc)
            matrix[i, j] = row['Frac']
    return matrix


def load_half(filepath, half_label):
    """Parse one interhbond file and build its directed graph."""
    df = parse_hbond_file(filepath)
    df['Acc_short'] = df['Acc'].apply(get_short_name)
    df['Don_short'] = df['Don'].apply(get_short_name)

    print(f"\n=== {half_label} ===")
    print(f"Loaded {len(df)} hydrogen bond records from {filepath}.\n")
    print(df[['Acc_short', 'Don_short', 'Frac', 'AvgDist', 'AvgAng']].to_string(index=False))

    G = nx.DiGraph()
    for _, row in df.iterrows():
        if row['Acc_short'] in all_labels and row['Don_short'] in all_labels:
            G.add_edge(row['Don_short'], row['Acc_short'], weight=row['Frac'])

    return df, G


def draw_network(ax, df, G, half_label, max_frac):
    """Draw one peptide-peptide H-bond network onto the given axis.

    max_frac is passed in (shared across both halves) so edge opacity is on
    the same scale in both plots, rather than each half normalizing to its
    own maximum.
    """
    node_colors = [color_a if n in group_a_labels else color_b for n in all_labels]

    nx.draw_networkx_nodes(G, pos, nodelist=all_labels, node_color=node_colors,
                           node_size=2200, edgecolors='white', linewidths=2.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='white',
                            font_weight='bold', ax=ax)

    for u, v, d in G.edges(data=True):
        width = np.power(d['weight'], 0.5) * 55
        alpha = min(1.0, (d['weight'] / max_frac) + 0.25)

        if u in group_a_labels and v in group_b_labels:
            edge_color = "#E74C3C"   # Red  – A → B
        elif u in group_b_labels and v in group_a_labels:
            edge_color = "#3498DB"   # Blue – B → A
        else:
            edge_color = "#BDC3C7"   # Grey – intra-peptide

        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width,
                               edge_color=edge_color, alpha=alpha,
                               connectionstyle="arc3,rad=0.2",
                               arrowsize=20, ax=ax)

    ax.set_title(half_label, fontsize=16, pad=15, weight='bold')
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


def draw_heatmap(ax, df, donor_labels, acceptor_labels, cmap, vmax, title,
                  xlabel, ylabel):
    """Draw one 4x4 donor/acceptor heatmap onto the given axis."""
    mat = build_heatmap_matrix(df, donor_labels=donor_labels, acceptor_labels=acceptor_labels)

    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect='equal')
    ax.set_xticks(range(len(acceptor_labels)))
    ax.set_xticklabels(acceptor_labels, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(donor_labels)))
    ax.set_yticklabels(donor_labels, fontsize=11, fontweight='bold')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=25)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = 'white' if val > vmax * 0.55 else 'black'
            ax.text(j, i, f"{val:.4f}", ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    return im


def run_hbond_analysis():
    # --- Load both halves ---
    df_first, G_first = load_half('interhbond_first_half.txt', 'First Half')
    df_second, G_second = load_half('interhbond_second_half.txt', 'Second Half')

    # =================================================================
    # NETWORKS SIDE BY SIDE
    # =================================================================
    fig = plt.figure(figsize=(16, 8.5), facecolor='white')
    gs = fig.add_gridspec(1, 2, wspace=0.15)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Shared alpha scale so edge opacity is comparable between the two halves
    max_frac_first = df_first['Frac'].max() if len(df_first) else 0.0
    max_frac_second = df_second['Frac'].max() if len(df_second) else 0.0
    max_frac_shared = max(max_frac_first, max_frac_second) if (max_frac_first or max_frac_second) else 1.0

    draw_network(ax1, df_first, G_first, 'First Half', max_frac_shared)
    draw_network(ax2, df_second, G_second, 'Second Half', max_frac_shared)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               label='Peptide A (L2, I3, V4, A5)',
               markerfacecolor=color_a, markersize=15),
        Line2D([0], [0], marker='o', color='w',
               label='Peptide B (L10, I11, V12, A13)',
               markerfacecolor=color_b, markersize=15),
        Line2D([0], [0], color='#3498DB', lw=6, label='B → A Interaction'),
        Line2D([0], [0], color='#E74C3C', lw=6, label='A → B Interaction')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=13, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Hydrogen Bond Network {peptide} — First vs Second Half",
                 fontsize=20, weight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f'hbond_network_{peptide}_first_second_half.png', dpi=600, bbox_inches='tight')

    # =================================================================
    # HEATMAPS SIDE BY SIDE (First Half block | Second Half block)
    # =================================================================

    # Shared color scale across all four heatmaps for fair comparison
    def inter_vmax(df):
        mask = (
            (df['Don_short'].isin(group_a_labels) & df['Acc_short'].isin(group_b_labels)) |
            (df['Don_short'].isin(group_b_labels) & df['Acc_short'].isin(group_a_labels))
        )
        return df.loc[mask, 'Frac'].max() if mask.any() else 1.0

    vmax = max(inter_vmax(df_first), inter_vmax(df_second))

    fig2 = plt.figure(figsize=(15, 12), facecolor='white')
    gs2 = fig2.add_gridspec(2, 4, width_ratios=[0.06, 1, 1, 0.06], wspace=0.45, hspace=0.4)

    cax_red = fig2.add_subplot(gs2[:, 0])
    ax00 = fig2.add_subplot(gs2[0, 1])
    ax01 = fig2.add_subplot(gs2[0, 2])
    ax10 = fig2.add_subplot(gs2[1, 1])
    ax11 = fig2.add_subplot(gs2[1, 2])
    cax_blue = fig2.add_subplot(gs2[:, 3])

    im1 = draw_heatmap(ax00, df_first, group_a_labels, group_b_labels, 'Reds', vmax,
                        'First Half: A → B', 'Acceptor (Peptide B)', 'Donor (Peptide A)')
    im2 = draw_heatmap(ax01, df_first, group_b_labels, group_a_labels, 'Blues', vmax,
                        'First Half: B → A', 'Acceptor (Peptide A)', 'Donor (Peptide B)')
    im3 = draw_heatmap(ax10, df_second, group_a_labels, group_b_labels, 'Reds', vmax,
                        'Second Half: A → B', 'Acceptor (Peptide B)', 'Donor (Peptide A)')
    im4 = draw_heatmap(ax11, df_second, group_b_labels, group_a_labels, 'Blues', vmax,
                        'Second Half: B → A', 'Acceptor (Peptide A)', 'Donor (Peptide B)')

    # Red colorbar sits on its own column on the far left, ticks/label facing left
    cb_red = fig2.colorbar(im1, cax=cax_red)
    cax_red.yaxis.set_ticks_position('left')
    cax_red.yaxis.set_label_position('left')
    cb_red.set_label('H-Bond Fraction (A → B)', fontsize=11, fontweight='bold')

    # Blue colorbar sits on its own column on the far right
    cb_blue = fig2.colorbar(im2, cax=cax_blue)
    cb_blue.set_label('H-Bond Fraction (B → A)', fontsize=11, fontweight='bold')

    fig2.suptitle(f'Inter-Peptide H-Bond Heatmaps — {peptide}: First vs Second Half',
                  fontsize=18, fontweight='bold', y=0.98)
    fig2.savefig(f'hbond_heatmaps_{peptide}_first_second_half.png', dpi=600, bbox_inches='tight')


# =====================================================================
# PART 2: PEPTIDE-PEPTIDE ANGLE HISTOGRAM (First vs Second Half)
# =====================================================================

def fd_stats(data, label):
    """Compute IQR and Freedman-Diaconis bin width, print, and return values."""
    data = np.array(data)
    n = len(data)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (n ** (1 / 3))
    print(f"[{label}] iqr = {iqr}")
    print(f"[{label}] Bin width (Freedman–Diaconis): {bin_width}")
    return iqr, bin_width


def run_angle_analysis():
    # Load both halves
    angles_first = np.loadtxt("pep1pep2_angle_first_half.dat", skiprows=1, usecols=1)
    angles_second = np.loadtxt("pep1pep2_angle_second_half.dat", skiprows=1, usecols=1)

    # 4.5° bins from 0 to 180
    bins = np.arange(0, 184.5, 4.5)

    fig3 = plt.figure(figsize=(8, 6))

    plt.hist(angles_first, bins=bins, edgecolor='black', density=True,
             alpha=0.5, label="First half", color='tab:blue')
    plt.hist(angles_second, bins=bins, edgecolor='black', density=True,
             alpha=0.5, label="Second half", color='tab:orange')

    plt.xlim(0, 180)
    plt.xticks(np.arange(0, 181, 20))

    plt.ylim(0, 0.02)
    plt.yticks(np.arange(0, 0.0201, 0.002))

    plt.xlabel("Angle")
    plt.ylabel("Density")
    plt.title(f"Histogram of {peptide} & {peptide} Angles (First vs Second Half)")
    plt.legend(loc='upper left')

    iqr_first, bw_first = fd_stats(angles_first, "First half")
    iqr_second, bw_second = fd_stats(angles_second, "Second half")

    # Annotate IQR and bin width for both halves on the plot
    annotation = (
        f"First half:  IQR = {iqr_first:.2f}, Bin width (F-D) = {bw_first:.2f}\n"
        f"Second half: IQR = {iqr_second:.2f}, Bin width (F-D) = {bw_second:.2f}"
    )
    plt.text(0.97, 0.97, annotation, transform=plt.gca().transAxes,
              ha='right', va='top', fontsize=9,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    fig3.savefig(f"{peptide}_angle_histogram_first_vs_second_half.png", dpi=400)


# =====================================================================
# PART 3: END-TO-END DISTANCE HISTOGRAM (First vs Second Half)
# =====================================================================

def run_end_to_end_dist_analysis():
    # Load both halves (col 0 = frame number, col 1 = end-to-end distance)
    dist_first = np.loadtxt("end_to_end_dist_first_half.dat", skiprows=1, usecols=1)
    dist_second = np.loadtxt("end_to_end_dist_second_half.dat", skiprows=1, usecols=1)

    iqr_first, bw_first = fd_stats(dist_first, "First half")
    iqr_second, bw_second = fd_stats(dist_second, "Second half")

    # Shared bin width (finer of the two Freedman–Diaconis widths) and shared
    # range across both halves so the two histograms are directly comparable
    bin_width = min(bw_first, bw_second)
    data_min = min(dist_first.min(), dist_second.min())
    data_max = max(dist_first.max(), dist_second.max())
    bins = np.arange(data_min, data_max + bin_width, bin_width)

    fig4 = plt.figure(figsize=(8, 6))

    plt.hist(dist_first, bins=bins, edgecolor='black', density=True,
             alpha=0.5, label="First half", color='tab:blue')
    plt.hist(dist_second, bins=bins, edgecolor='black', density=True,
             alpha=0.5, label="Second half", color='tab:orange')

    plt.xlabel("End-to-End Distance (Å)")
    plt.ylabel("Density")
    plt.title(f"Histogram of {peptide} End-to-End Distance (First vs Second Half)")
    plt.legend(loc='upper left')

    # Annotate IQR and bin width for both halves on the plot
    annotation = (
        f"First half:  IQR = {iqr_first:.2f}, Bin width (F-D) = {bw_first:.2f}\n"
        f"Second half: IQR = {iqr_second:.2f}, Bin width (F-D) = {bw_second:.2f}"
    )
    plt.text(0.97, 0.97, annotation, transform=plt.gca().transAxes,
              ha='right', va='top', fontsize=9,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    fig4.savefig(f"{peptide}_end_to_end_dist_histogram_first_vs_second_half.png", dpi=400)


# =====================================================================
# RUN ALL ANALYSES
# =====================================================================
if __name__ == "__main__":
    run_hbond_analysis()
    run_angle_analysis()
    run_end_to_end_dist_analysis()

