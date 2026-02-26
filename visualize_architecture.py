"""
🏗️ CrisisSpot Architecture Visualization
==========================================
Trực quan hóa toàn bộ pipeline và quá trình gán nhãn
của paper: "Multimodal Classification of Social Media
Disaster Posts with GNN and Few-Shot Learning"

Run: python visualize_architecture.py
Output: 4 hình ảnh PNG trong thư mục visualizations/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import matplotlib.patheffects as pe
import numpy as np
import os

# Create output directory
os.makedirs('visualizations', exist_ok=True)

# ============================================================
# Color Palette
# ============================================================
COLORS = {
    'bg': '#0f0f1a',
    'bg_light': '#1a1a2e',
    'accent1': '#00d2ff',    # Cyan
    'accent2': '#7b2ff7',    # Purple
    'accent3': '#ff6b6b',    # Red/coral
    'accent4': '#51cf66',    # Green
    'accent5': '#ffd43b',    # Yellow
    'accent6': '#ff922b',    # Orange
    'text': '#e0e0e0',
    'text_dim': '#888899',
    'informative': '#51cf66',
    'not_informative': '#ff6b6b',
    'unlabeled': '#555577',
    'edge': '#333355',
}


def set_dark_style(ax, fig):
    """Apply dark theme to figure and axes."""
    fig.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    ax.tick_params(colors=COLORS['text'])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ============================================================
# FIGURE 1: Full Pipeline Architecture
# ============================================================
def draw_pipeline():
    fig, ax = plt.subplots(figsize=(20, 12))
    set_dark_style(ax, fig)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(10, 11.5, '🏗️ CrisisSpot — Full Pipeline Architecture',
            fontsize=22, fontweight='bold', color='white',
            ha='center', va='center',
            path_effects=[pe.withStroke(linewidth=3, foreground=COLORS['accent2'])])

    # --- Stage boxes ---
    stages = [
        (1.5,  8.5, 3, 2.5, COLORS['accent1'], '1. CLIP\nFeature Extraction',
         '• Text → Multilingual CLIP\n  → vector 512-d\n• Image → CLIP ViT-B/32\n  → vector 512-d'),
        (5.5,  8.5, 3, 2.5, COLORS['accent2'], '2. PCA\nDim Reduction',
         '• 512-d → 256-d\n• Giảm noise\n• Tránh overfitting\n• Nhanh hơn 2x'),
        (9.5,  8.5, 3, 2.5, COLORS['accent6'], '3. k-NN Graph\nConstruction',
         '• k = 16 neighbors\n• Cosine similarity\n• ~13,500 nodes\n• ~216,000 edges'),
        (13.5, 8.5, 3, 2.5, COLORS['accent4'], '4. GraphSAGE\nLate Fusion',
         '• 2 GNN layers (1024)\n• Separate text/image\n• Concat → Linear\n• Softmax → label'),
        (17.5, 8.5, 2.2, 2.5, COLORS['accent3'], '5. Output',
         '✅ Informative\n❌ Not Informative\n\nF1-Score\n~74-77%'),
    ]

    for x, y, w, h, color, title, desc in stages:
        # Box
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.15",
                             facecolor=color + '22', edgecolor=color,
                             linewidth=2)
        ax.add_patch(box)
        # Title
        ax.text(x, y + h/2 - 0.4, title,
                fontsize=10, fontweight='bold', color=color,
                ha='center', va='center')
        # Description
        ax.text(x, y - 0.3, desc,
                fontsize=7.5, color=COLORS['text_dim'],
                ha='center', va='center', family='monospace')

    # Arrows between stages
    arrow_style = "Simple,tail_width=1.5,head_width=8,head_length=6"
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + stages[i][2] / 2
        x2 = stages[i+1][0] - stages[i+1][2] / 2
        y = stages[i][1]
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color=COLORS['text_dim'],
                                    lw=2, connectionstyle='arc3,rad=0'))

    # --- Input section ---
    # Tweet box
    ax.add_patch(FancyBboxPatch((0.3, 4), 4.5, 3,
                                boxstyle="round,pad=0.2",
                                facecolor='#1a2a3a', edgecolor=COLORS['accent1'],
                                linewidth=1.5, linestyle='--'))
    ax.text(2.55, 6.7, '📱 Input: Crisis Tweet', fontsize=12, fontweight='bold',
            color=COLORS['accent1'], ha='center')
    ax.text(2.55, 6.1, '━━━━━━━━━━━━━━━━━━━━━', fontsize=8, color=COLORS['edge'], ha='center')
    ax.text(2.55, 5.6, '📝 Text: "Major flooding in\n    downtown Houston. Roads\n    completely submerged."',
            fontsize=8, color=COLORS['text'], ha='center', family='monospace')
    ax.text(2.55, 4.5, '🖼️ Image: [photo of flooded street]',
            fontsize=8, color=COLORS['text_dim'], ha='center', family='monospace')

    # Arrow from input to Stage 1
    ax.annotate('', xy=(1.5, 7.25), xytext=(2.55, 7),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent1'],
                                lw=2, connectionstyle='arc3,rad=-0.3'))

    # --- Few-Shot info ---
    ax.add_patch(FancyBboxPatch((5.5, 4), 4.5, 3,
                                boxstyle="round,pad=0.2",
                                facecolor='#2a1a2a', edgecolor=COLORS['accent5'],
                                linewidth=1.5, linestyle='--'))
    ax.text(7.75, 6.7, '🏷️ Few-Shot Labeling', fontsize=12, fontweight='bold',
            color=COLORS['accent5'], ha='center')
    ax.text(7.75, 6.1, '━━━━━━━━━━━━━━━━━━━━━', fontsize=8, color=COLORS['edge'], ha='center')
    ax.text(7.75, 5.5, '~13,500 tweets total\n'
            'Only 50-250 have labels\n'
            '(manually annotated)\n\n'
            'Rest: UNLABELED\n'
            '→ GNN predicts via graph',
            fontsize=8.5, color=COLORS['text'], ha='center', family='monospace')

    # --- Shuffle Split info ---
    ax.add_patch(FancyBboxPatch((11, 4), 4.5, 3,
                                boxstyle="round,pad=0.2",
                                facecolor='#1a2a1a', edgecolor=COLORS['accent4'],
                                linewidth=1.5, linestyle='--'))
    ax.text(13.25, 6.7, '🔀 Shuffle-Split Training', fontsize=12, fontweight='bold',
            color=COLORS['accent4'], ha='center')
    ax.text(13.25, 6.1, '━━━━━━━━━━━━━━━━━━━━━', fontsize=8, color=COLORS['edge'], ha='center')
    ax.text(13.25, 5.3,
            '2000 epochs (max)\n'
            'Early stop: 300 epochs\n\n'
            'Each epoch:\n'
            '  40% pseudo-train\n'
            '  60% pseudo-val\n'
            '  → Shuffle mỗi epoch',
            fontsize=8.5, color=COLORS['text'], ha='center', family='monospace')

    # --- Results box ---
    ax.add_patch(FancyBboxPatch((16.3, 4), 3.2, 3,
                                boxstyle="round,pad=0.2",
                                facecolor='#2a2a1a', edgecolor=COLORS['accent6'],
                                linewidth=1.5, linestyle='--'))
    ax.text(17.9, 6.7, '📊 Expected Results', fontsize=12, fontweight='bold',
            color=COLORS['accent6'], ha='center')
    ax.text(17.9, 6.1, '━━━━━━━━━━━━━━━━━━━', fontsize=8, color=COLORS['edge'], ha='center')
    ax.text(17.9, 5.3,
            'Labeled  Paper  Sirbu\n'
            '━━━━━━━━━━━━━━━━━━\n'
            '  50    74.8%  62.8%\n'
            ' 100    76.3%  66.9%\n'
            ' 250    77.1%  71.9%\n'
            '━━━━━━━━━━━━━━━━━━\n'
            'Metric: F1-Score',
            fontsize=8.5, color=COLORS['text'], ha='center', family='monospace')

    # --- Bottom info ---
    ax.text(10, 1.5,
            'Dataset: CrisisMMD (7 disaster events) │ GPU: Quadro RTX 8000 (paper) / T4 (Colab)\n'
            'Paper: "Multimodal Classification of Social Media Disaster Posts with GNN and Few-Shot Learning" — IEEE Access 2025',
            fontsize=9, color=COLORS['text_dim'], ha='center', va='center',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['bg_light'],
                      edgecolor=COLORS['edge']))

    plt.tight_layout()
    plt.savefig('visualizations/1_pipeline_architecture.png', dpi=150,
                bbox_inches='tight', facecolor=COLORS['bg'])
    print('✅ Saved: visualizations/1_pipeline_architecture.png')
    plt.close()


# ============================================================
# FIGURE 2: Graph Visualization with Few-Shot Labels
# ============================================================
def draw_graph_labels():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.set_facecolor(COLORS['bg'])

    for ax_idx, (ax, n_labeled, title_extra) in enumerate(zip(
        axes, [50, 250], ['50 labeled (0.37%)', '250 labeled (1.85%)']
    )):
        set_dark_style(ax, fig)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')

        # Generate nodes in clusters
        n_total = 300  # Scaled down for visualization (represents ~13,500)
        n_clusters = 5
        nodes_x, nodes_y, true_labels = [], [], []

        cluster_centers = [
            (0.5, 0.5), (-0.5, 0.5), (0, -0.5),
            (0.7, -0.3), (-0.7, -0.3)
        ]
        cluster_labels = [1, 1, 0, 0, 1]  # 1=info, 0=not_info
        cluster_names = ['Flood Damage', 'Building Collapse', 'Memes/Jokes',
                        'Unrelated Photos', 'Rescue Operations']

        for ci, (cx, cy) in enumerate(cluster_centers):
            n_per = n_total // n_clusters
            x = cx + np.random.randn(n_per) * 0.18
            y = cy + np.random.randn(n_per) * 0.18
            nodes_x.extend(x)
            nodes_y.extend(y)
            true_labels.extend([cluster_labels[ci]] * n_per)

        nodes_x = np.array(nodes_x)
        nodes_y = np.array(nodes_y)
        true_labels = np.array(true_labels)

        # Scale labeled count for visualization
        vis_labeled = int(n_labeled * n_total / 13500)
        vis_labeled = max(vis_labeled, 3)

        # Select labeled nodes (randomly from each cluster)
        labeled_mask = np.zeros(n_total, dtype=bool)
        per_cluster = max(1, vis_labeled // n_clusters)
        for ci in range(n_clusters):
            cluster_idx = np.where(np.arange(n_total) // (n_total // n_clusters) == ci)[0]
            chosen = np.random.choice(cluster_idx, size=min(per_cluster, len(cluster_idx)), replace=False)
            labeled_mask[chosen] = True

        # Draw edges (k-NN simplified)
        from scipy.spatial.distance import cdist
        coords = np.column_stack([nodes_x, nodes_y])
        dists = cdist(coords, coords)
        k = 4  # Reduced k for visibility
        for i in range(n_total):
            neighbors = np.argsort(dists[i])[1:k+1]
            for j in neighbors:
                ax.plot([nodes_x[i], nodes_x[j]], [nodes_y[i], nodes_y[j]],
                       color=COLORS['edge'], linewidth=0.3, alpha=0.4, zorder=1)

        # Draw unlabeled nodes
        unlabeled_idx = ~labeled_mask
        ax.scatter(nodes_x[unlabeled_idx], nodes_y[unlabeled_idx],
                  c=COLORS['unlabeled'], s=25, alpha=0.5, zorder=2,
                  edgecolors='none')

        # Draw labeled nodes (larger, with glow)
        for i in np.where(labeled_mask)[0]:
            color = COLORS['informative'] if true_labels[i] == 1 else COLORS['not_informative']
            # Glow effect
            ax.scatter(nodes_x[i], nodes_y[i], c=color, s=200, alpha=0.2, zorder=3,
                      edgecolors='none')
            ax.scatter(nodes_x[i], nodes_y[i], c=color, s=80, alpha=0.8, zorder=4,
                      edgecolors='white', linewidth=0.5)

        # Cluster labels
        for ci, (cx, cy) in enumerate(cluster_centers):
            ax.text(cx, cy + 0.3, cluster_names[ci],
                   fontsize=7, color=COLORS['text_dim'],
                   ha='center', va='center', fontstyle='italic',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS['bg'],
                            edgecolor='none', alpha=0.7))

        # Title
        ax.text(0, 1.2, f'🔍 Few-Shot: {title_extra}',
               fontsize=14, fontweight='bold', color='white', ha='center',
               path_effects=[pe.withStroke(linewidth=2, foreground=COLORS['accent2'])])

        # Stats
        n_info = np.sum(labeled_mask & (true_labels == 1))
        n_not = np.sum(labeled_mask & (true_labels == 0))
        ax.text(0, -1.15,
               f'Total: {n_total} nodes │ Labeled: {np.sum(labeled_mask)} '
               f'(✅ {n_info} info, ❌ {n_not} not-info) │ '
               f'Unlabeled: {n_total - np.sum(labeled_mask)}',
               fontsize=8, color=COLORS['text_dim'], ha='center', family='monospace')

    # Legend
    legend_elements = [
        plt.scatter([], [], c=COLORS['informative'], s=80, edgecolors='white',
                   linewidth=0.5, label='✅ Labeled: Informative'),
        plt.scatter([], [], c=COLORS['not_informative'], s=80, edgecolors='white',
                   linewidth=0.5, label='❌ Labeled: Not Informative'),
        plt.scatter([], [], c=COLORS['unlabeled'], s=25, alpha=0.5,
                   label='⬜ Unlabeled (GNN predicts)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
              fontsize=11, frameon=True, facecolor=COLORS['bg_light'],
              edgecolor=COLORS['edge'], labelcolor=COLORS['text'])

    fig.suptitle('📊 Graph-Based Few-Shot Classification — Label Propagation via GNN',
                fontsize=18, fontweight='bold', color='white', y=0.98,
                path_effects=[pe.withStroke(linewidth=2, foreground=COLORS['accent2'])])

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig('visualizations/2_graph_fewshot_labels.png', dpi=150,
                bbox_inches='tight', facecolor=COLORS['bg'])
    print('✅ Saved: visualizations/2_graph_fewshot_labels.png')
    plt.close()


# ============================================================
# FIGURE 3: Late Fusion Architecture Detail
# ============================================================
def draw_late_fusion():
    fig, ax = plt.subplots(figsize=(18, 10))
    set_dark_style(ax, fig)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(9, 9.5, '🧠 GraphSAGE Late Fusion — Architecture Detail',
            fontsize=20, fontweight='bold', color='white', ha='center',
            path_effects=[pe.withStroke(linewidth=3, foreground=COLORS['accent2'])])

    # ---- Text Branch (top) ----
    text_y = 7
    components_text = [
        (1.5, text_y, 2.2, 1.2, COLORS['accent1'], 'Text Input\n256-d (PCA)', '📝'),
        (5,   text_y, 2.2, 1.2, COLORS['accent2'], 'GraphSAGE\nLayer 1\n(256→1024)', ''),
        (8.5, text_y, 2.2, 1.2, COLORS['accent2'], 'GraphSAGE\nLayer 2\n(1024→1024)', ''),
        (12,  text_y, 1.8, 1.2, '#888899', 'h_text\n1024-d', ''),
    ]

    for x, y, w, h, color, label, icon in components_text:
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.1",
                             facecolor=color + '33', edgecolor=color,
                             linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, f'{icon}\n{label}' if icon else label,
               fontsize=8, fontweight='bold', color=color,
               ha='center', va='center')

    # Arrows text branch
    for i in range(len(components_text) - 1):
        x1 = components_text[i][0] + components_text[i][2] / 2
        x2 = components_text[i+1][0] - components_text[i+1][2] / 2
        ax.annotate('', xy=(x2, text_y), xytext=(x1, text_y),
                   arrowprops=dict(arrowstyle='->', color=COLORS['accent1'], lw=2))

    # ---- Image Branch (bottom) ----
    img_y = 3.5
    components_img = [
        (1.5, img_y, 2.2, 1.2, COLORS['accent6'], 'Image Input\n256-d (PCA)', '🖼️'),
        (5,   img_y, 2.2, 1.2, COLORS['accent2'], 'GraphSAGE\nLayer 1\n(256→1024)', ''),
        (8.5, img_y, 2.2, 1.2, COLORS['accent2'], 'GraphSAGE\nLayer 2\n(1024→1024)', ''),
        (12,  img_y, 1.8, 1.2, '#888899', 'h_image\n1024-d', ''),
    ]

    for x, y, w, h, color, label, icon in components_img:
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.1",
                             facecolor=color + '33', edgecolor=color,
                             linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, f'{icon}\n{label}' if icon else label,
               fontsize=8, fontweight='bold', color=color,
               ha='center', va='center')

    for i in range(len(components_img) - 1):
        x1 = components_img[i][0] + components_img[i][2] / 2
        x2 = components_img[i+1][0] - components_img[i+1][2] / 2
        ax.annotate('', xy=(x2, img_y), xytext=(x1, img_y),
                   arrowprops=dict(arrowstyle='->', color=COLORS['accent6'], lw=2))

    # ---- Concatenation ----
    concat_x, concat_y = 14.5, 5.25
    box = FancyBboxPatch((concat_x - 1, concat_y - 1.5), 2, 3,
                         boxstyle="round,pad=0.1",
                         facecolor=COLORS['accent5'] + '33',
                         edgecolor=COLORS['accent5'], linewidth=2)
    ax.add_patch(box)
    ax.text(concat_x, concat_y + 0.6, '⊕', fontsize=24, color=COLORS['accent5'],
           ha='center', va='center', fontweight='bold')
    ax.text(concat_x, concat_y - 0.2, 'CONCAT', fontsize=9, fontweight='bold',
           color=COLORS['accent5'], ha='center')
    ax.text(concat_x, concat_y - 0.8, '2048-d', fontsize=8,
           color=COLORS['text_dim'], ha='center')

    # Arrows to concat
    ax.annotate('', xy=(concat_x - 1, text_y), xytext=(12 + 0.9, text_y),
               arrowprops=dict(arrowstyle='->', color=COLORS['text_dim'], lw=2,
                              connectionstyle='arc3,rad=-0.2'))
    ax.annotate('', xy=(concat_x - 1, img_y), xytext=(12 + 0.9, img_y),
               arrowprops=dict(arrowstyle='->', color=COLORS['text_dim'], lw=2,
                              connectionstyle='arc3,rad=0.2'))

    # ---- Linear + Softmax ----
    linear_x = 16.5
    box = FancyBboxPatch((linear_x - 0.8, concat_y - 1), 1.6, 2,
                         boxstyle="round,pad=0.1",
                         facecolor=COLORS['accent4'] + '33',
                         edgecolor=COLORS['accent4'], linewidth=2)
    ax.add_patch(box)
    ax.text(linear_x, concat_y + 0.3, 'Linear\n2048→2', fontsize=9, fontweight='bold',
           color=COLORS['accent4'], ha='center')
    ax.text(linear_x, concat_y - 0.5, 'Softmax', fontsize=8,
           color=COLORS['text_dim'], ha='center')

    ax.annotate('', xy=(linear_x - 0.8, concat_y), xytext=(concat_x + 1, concat_y),
               arrowprops=dict(arrowstyle='->', color=COLORS['text_dim'], lw=2))

    # Output labels
    ax.text(17.8, concat_y + 0.4, '✅ 0.92', fontsize=11, fontweight='bold',
           color=COLORS['informative'], ha='left')
    ax.text(17.8, concat_y - 0.4, '❌ 0.08', fontsize=11, fontweight='bold',
           color=COLORS['not_informative'], ha='left')

    # ---- GraphSAGE detail box ----
    ax.add_patch(FancyBboxPatch((1, 0.5), 10, 1.8,
                                boxstyle="round,pad=0.2",
                                facecolor=COLORS['bg_light'],
                                edgecolor=COLORS['edge'], linewidth=1.5))
    ax.text(6, 2, '🔍 GraphSAGE: Aggregate Neighbors', fontsize=11, fontweight='bold',
           color=COLORS['accent2'], ha='center')
    ax.text(6, 1.2,
           "hᵥ⁽ˡ⁾ = σ( W · CONCAT( hᵥ⁽ˡ⁻¹⁾, MEAN({ hᵤ⁽ˡ⁻¹⁾ : u ∈ N(v) }) ) )\n"
           "Mỗi node cập nhật feature bằng mean-pool features của 16 neighbors + chính nó",
           fontsize=9, color=COLORS['text'], ha='center', family='monospace')

    # Labels
    ax.text(1.5, 8.6, 'TEXT BRANCH', fontsize=10, fontweight='bold',
           color=COLORS['accent1'], ha='left', fontstyle='italic')
    ax.text(1.5, 5.1, 'IMAGE BRANCH', fontsize=10, fontweight='bold',
           color=COLORS['accent6'], ha='left', fontstyle='italic')

    # "Late Fusion" annotation
    ax.text(14.5, 8.5, 'LATE\nFUSION', fontsize=13, fontweight='bold',
           color=COLORS['accent5'], ha='center', va='center', alpha=0.4,
           rotation=0)

    plt.tight_layout()
    plt.savefig('visualizations/3_late_fusion_detail.png', dpi=150,
                bbox_inches='tight', facecolor=COLORS['bg'])
    print('✅ Saved: visualizations/3_late_fusion_detail.png')
    plt.close()


# ============================================================
# FIGURE 4: Labeling Process & Shuffle-Split
# ============================================================
def draw_labeling_process():
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.set_facecolor(COLORS['bg'])

    # ---- Panel 1: Dataset Overview ----
    ax = axes[0]
    set_dark_style(ax, fig)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, '📁 CrisisMMD Dataset', fontsize=14, fontweight='bold',
           color=COLORS['accent1'], ha='center')

    # Stacked bar
    total = 13500
    labeled_sizes = [50, 100, 250]
    bar_y = [7.5, 5.5, 3.5]
    bar_w = 8

    for idx, (ls, by) in enumerate(zip(labeled_sizes, bar_y)):
        # Full bar (unlabeled)
        ax.add_patch(FancyBboxPatch((1, by - 0.3), bar_w, 0.6,
                                    boxstyle="round,pad=0.05",
                                    facecolor=COLORS['unlabeled'] + '44',
                                    edgecolor=COLORS['edge']))
        # Labeled portion
        labeled_w = bar_w * (ls / total)
        ax.add_patch(FancyBboxPatch((1, by - 0.3), max(labeled_w, 0.15), 0.6,
                                    boxstyle="round,pad=0.05",
                                    facecolor=COLORS['accent4'],
                                    edgecolor=COLORS['accent4']))

        pct = ls / total * 100
        ax.text(0.8, by, f'{ls}', fontsize=10, fontweight='bold',
               color=COLORS['accent4'], ha='right', va='center')
        ax.text(9.2, by, f'{pct:.1f}%', fontsize=9,
               color=COLORS['text_dim'], ha='left', va='center')

    ax.text(5, 2.2, f'Total: {total:,} tweets', fontsize=10,
           color=COLORS['text'], ha='center')
    ax.text(5, 1.5, '7 disaster events\n(hurricanes, earthquakes, floods...)',
           fontsize=8, color=COLORS['text_dim'], ha='center')

    # Legend
    ax.add_patch(FancyBboxPatch((2, 0.3), 0.4, 0.3, boxstyle="round,pad=0.02",
                                facecolor=COLORS['accent4']))
    ax.text(2.6, 0.45, 'Labeled', fontsize=8, color=COLORS['accent4'], va='center')
    ax.add_patch(FancyBboxPatch((5, 0.3), 0.4, 0.3, boxstyle="round,pad=0.02",
                                facecolor=COLORS['unlabeled'] + '44',
                                edgecolor=COLORS['edge']))
    ax.text(5.6, 0.45, 'Unlabeled', fontsize=8, color=COLORS['text_dim'], va='center')

    # ---- Panel 2: Shuffle-Split Training ----
    ax = axes[1]
    set_dark_style(ax, fig)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, '🔀 Shuffle-Split Strategy', fontsize=14, fontweight='bold',
           color=COLORS['accent5'], ha='center')

    # Show 3 epochs
    epochs_data = [
        ('Epoch 1', [1,1,0,1,0, 0,1,0,1,1], [0,0,0,0,1,1,1,1,1,1]),  # 1=train, 0=val
        ('Epoch 2', [0,1,1,0,1, 1,0,0,1,0], [0,0,0,0,1,1,1,1,1,1]),
        ('Epoch 3', [1,0,0,1,1, 0,0,1,0,1], [0,0,0,0,1,1,1,1,1,1]),
    ]

    for ep_idx, (ep_name, split, _) in enumerate(epochs_data):
        by = 7.5 - ep_idx * 2.5
        ax.text(1, by + 0.8, ep_name, fontsize=10, fontweight='bold',
               color=COLORS['text'], ha='left')

        # Draw 10 squares representing labeled samples
        for i, s in enumerate(split):
            x = 1.5 + i * 0.7
            color = COLORS['accent4'] if s == 1 else COLORS['accent6']
            label = 'T' if s == 1 else 'V'
            ax.add_patch(FancyBboxPatch((x, by - 0.25), 0.5, 0.5,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color + '55',
                                        edgecolor=color, linewidth=1.5))
            ax.text(x + 0.25, by, label, fontsize=8, fontweight='bold',
                   color=color, ha='center', va='center')

        # Ratio
        n_train = sum(split)
        n_val = len(split) - n_train
        ax.text(9, by, f'{n_train}T/{n_val}V', fontsize=8,
               color=COLORS['text_dim'], ha='center', va='center')

    # Legend
    ax.add_patch(FancyBboxPatch((2, 0.7), 0.4, 0.3, boxstyle="round,pad=0.02",
                                facecolor=COLORS['accent4'] + '55',
                                edgecolor=COLORS['accent4']))
    ax.text(2.6, 0.85, 'T = pseudo-Train (40%)', fontsize=7,
           color=COLORS['accent4'], va='center')
    ax.add_patch(FancyBboxPatch((6, 0.7), 0.4, 0.3, boxstyle="round,pad=0.02",
                                facecolor=COLORS['accent6'] + '55',
                                edgecolor=COLORS['accent6']))
    ax.text(6.6, 0.85, 'V = pseudo-Val (60%)', fontsize=7,
           color=COLORS['accent6'], va='center')

    ax.text(5, 0.2, '→ Mỗi epoch shuffle lại → Model thấy mọi mẫu', fontsize=7.5,
           color=COLORS['text_dim'], ha='center', fontstyle='italic')

    # ---- Panel 3: Label Propagation ----
    ax = axes[2]
    set_dark_style(ax, fig)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, '🌊 Label Propagation via GNN', fontsize=14, fontweight='bold',
           color=COLORS['accent3'], ha='center')

    # Draw a small graph showing propagation
    np.random.seed(123)
    nodes = [
        (3, 7, 'labeled', 1),      # Labeled: Informative
        (7, 7, 'labeled', 0),      # Labeled: Not Informative
        (2, 5, 'predicted', 1),    # Predicted via neighbor
        (4.5, 5.5, 'predicted', 1),
        (5.5, 5.5, 'predicted', 0),
        (8, 5, 'predicted', 0),
        (3, 3.5, 'predicted', 1),
        (5, 3.5, 'unlabeled', -1),
        (7, 3.5, 'predicted', 0),
        (5, 2, 'unlabeled', -1),
    ]

    edges = [
        (0, 2), (0, 3), (1, 4), (1, 5),
        (2, 6), (3, 4), (4, 5), (3, 7),
        (5, 8), (6, 7), (7, 9), (8, 9),
    ]

    # Draw edges
    for i, j in edges:
        ax.plot([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]],
               color=COLORS['edge'], linewidth=1.5, alpha=0.5, zorder=1)

    # Draw nodes
    for x, y, status, label in nodes:
        if status == 'labeled':
            color = COLORS['informative'] if label == 1 else COLORS['not_informative']
            ax.scatter(x, y, c=color, s=250, zorder=3, edgecolors='white', linewidth=2)
            ax.scatter(x, y, c=color, s=500, zorder=2, alpha=0.15)
            marker = '✅' if label == 1 else '❌'
            ax.text(x, y + 0.5, f'{marker} LABELED', fontsize=7, fontweight='bold',
                   color=color, ha='center')
        elif status == 'predicted':
            color = COLORS['informative'] if label == 1 else COLORS['not_informative']
            ax.scatter(x, y, c=color, s=120, zorder=3, edgecolors=color,
                      linewidth=1.5, alpha=0.7)
            ax.text(x, y - 0.5, 'predicted', fontsize=6,
                   color=COLORS['text_dim'], ha='center', fontstyle='italic')
        else:
            ax.scatter(x, y, c=COLORS['unlabeled'], s=100, zorder=3,
                      edgecolors=COLORS['text_dim'], linewidth=1)
            ax.text(x, y - 0.5, '?', fontsize=10, fontweight='bold',
                   color=COLORS['text_dim'], ha='center')

    ax.text(5, 1, 'GNN "lan truyền" nhãn từ labeled nodes\n'
            'sang unlabeled nodes qua edges trong graph',
           fontsize=8, color=COLORS['text_dim'], ha='center', fontstyle='italic',
           bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg_light'],
                    edgecolor=COLORS['edge']))

    fig.suptitle('📋 Quá trình Gán Nhãn & Training trong Few-Shot GNN',
                fontsize=18, fontweight='bold', color='white', y=1.02,
                path_effects=[pe.withStroke(linewidth=2, foreground=COLORS['accent2'])])

    plt.tight_layout()
    plt.savefig('visualizations/4_labeling_process.png', dpi=150,
                bbox_inches='tight', facecolor=COLORS['bg'])
    print('✅ Saved: visualizations/4_labeling_process.png')
    plt.close()


# ============================================================
# FIGURE 5: Results Comparison Chart
# ============================================================
def draw_results_comparison():
    fig, ax = plt.subplots(figsize=(12, 7))
    set_dark_style(ax, fig)

    S = [50, 100, 250]
    paper_f1 = [74.8, 76.3, 77.1]
    sirbu_f1 = [62.8, 66.9, 71.9]

    # Bar chart
    x = np.arange(len(S))
    width = 0.3

    bars1 = ax.bar(x - width/2, paper_f1, width, label='Paper: GNN + Late Fusion + CLIP',
                   color=COLORS['accent4'], alpha=0.85, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, sirbu_f1, width, label='Baseline: Sirbu (FixMatch + BT)',
                   color=COLORS['accent3'], alpha=0.85, edgecolor='white', linewidth=0.5)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{bar.get_height():.1f}%', ha='center', va='bottom',
               color=COLORS['accent4'], fontweight='bold', fontsize=11)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{bar.get_height():.1f}%', ha='center', va='bottom',
               color=COLORS['accent3'], fontweight='bold', fontsize=11)

    # Improvement arrows
    for i in range(len(S)):
        diff = paper_f1[i] - sirbu_f1[i]
        mid_x = x[i]
        mid_y = (paper_f1[i] + sirbu_f1[i]) / 2
        ax.annotate(f'+{diff:.1f}%', xy=(mid_x, mid_y),
                   fontsize=9, fontweight='bold', color=COLORS['accent5'],
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS['accent5'] + '33',
                            edgecolor=COLORS['accent5']))

    ax.set_xlabel('Number of Labeled Samples (Few-Shot)', fontsize=12,
                 color=COLORS['text'], labelpad=10)
    ax.set_ylabel('F1-Score (%)', fontsize=12, color=COLORS['text'], labelpad=10)
    ax.set_title('📊 Paper Results: GNN+CLIP vs Baseline (CrisisMMD)',
                fontsize=16, fontweight='bold', color='white', pad=15,
                path_effects=[pe.withStroke(linewidth=2, foreground=COLORS['accent2'])])
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s} samples' for s in S], fontsize=11, color=COLORS['text'])
    ax.set_ylim(55, 85)
    ax.yaxis.set_tick_params(labelsize=10, labelcolor=COLORS['text'])
    ax.grid(axis='y', alpha=0.15, color=COLORS['text_dim'])
    ax.legend(fontsize=10, loc='lower right', frameon=True,
             facecolor=COLORS['bg_light'], edgecolor=COLORS['edge'],
             labelcolor=COLORS['text'])

    # Add horizontal line for "target"
    ax.axhline(y=77.1, color=COLORS['accent4'], linestyle='--', alpha=0.3, linewidth=1)
    ax.text(2.4, 77.8, 'Best: 77.1%', fontsize=8, color=COLORS['accent4'], alpha=0.5)

    plt.tight_layout()
    plt.savefig('visualizations/5_results_comparison.png', dpi=150,
                bbox_inches='tight', facecolor=COLORS['bg'])
    print('✅ Saved: visualizations/5_results_comparison.png')
    plt.close()


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print('🎨 Generating CrisisSpot Architecture Visualizations...\n')

    draw_pipeline()
    draw_graph_labels()
    draw_late_fusion()
    draw_labeling_process()
    draw_results_comparison()

    print(f'\n🎉 All visualizations saved to: visualizations/')
    print('Files:')
    print('  1_pipeline_architecture.png  — Full pipeline overview')
    print('  2_graph_fewshot_labels.png   — Graph with few-shot labels')
    print('  3_late_fusion_detail.png     — GraphSAGE Late Fusion detail')
    print('  4_labeling_process.png       — Labeling & training process')
    print('  5_results_comparison.png     — Paper vs baseline results')
