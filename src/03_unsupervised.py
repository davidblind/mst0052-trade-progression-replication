"""STEP 03: justify the unsupervised country features.

Outputs: results/figures/pca_scree.png, results/figures/kmeans_k.png,
         results/figures/kmeans_k_wide.png, results/figures/dendrogram.png,
         results/tables/unsupervised_summary.csv

Run:  cd src && python 03_unsupervised.py

  [X] load_training_baskets(), pca_scree()
  [x] main()               choose N_PCA from the scree plot, write it into config.py
  [x] kmeans_sweep()       inertia (elbow) and silhouette for k = 2..10, plot
  [x] kmeans_stability()   ARI between seeds at the chosen k
  [x] ward_comparison()    Ward labels vs k-means (ARI) + dendrogram for one year
  [x] main()               choose N_CLUSTERS, write it into config.py, save
                           unsupervised_summary.csv
  [x] report               name the clusters (notebook C6)
"""
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # no window: figures are only saved to disk
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from common import train_years
from config import BASKETS_FILE, FIGURES_DIR, TABLES_DIR, N_PCA, N_CLUSTERS, SEED, TEST_YEAR

K_RANGE = range(2, 11)                   # the k values N_CLUSTERS is chosen from
K_WIDE = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100, 150, 300]   # sensitivity check only
DENDROGRAM_YEAR = 2014                   # one base year -> readable country-code leaves


def load_training_baskets():
    """The export baskets of the base years that train the FINAL model (step 05).

    ASSUMPTION: settings are chosen on base years <= TEST_YEAR - HORIZON (2014), every
    TRAIN_EVERY-th year (2000, 2002, ..., 2014), so the test year 2019 never influences
    N_PCA or N_CLUSTERS. Every 2nd year only: consecutive years of the same country are
    near-duplicate rows and would overweight slow-moving countries.
    One row = one country-year; columns after (base_year, country) = log(1 + RCA) per product.
    """
    baskets = pd.read_parquet(BASKETS_FILE)
    years = train_years(TEST_YEAR, sorted(baskets["base_year"].unique()))
    rows = baskets[baskets["base_year"].isin(years)]
    print(f"Baskets: {len(rows):,} country-years from base years {years}, {rows.shape[1] - 2} products")
    return rows, rows.columns[2:]


def pca_scree(X, max_components = 40):
    """Explained-variance ratio and its cumulative sum (L9 'Read the variance').

    ASSUMPTION: no StandardScaler. The basket columns are all log(1 + RCA), i.e. already in
    the same unit, so PCA runs on centred, unscaled data. Scaling would give a product that
    almost nobody exports the same weight as crude oil.
    """
    pca = PCA(n_components = max_components, random_state = SEED).fit(X)
    ratio = pca.explained_variance_ratio_           # share of total variance per component
    cum = np.cumsum(ratio)                          # share explained by the first k components
    fig, ax = plt.subplots(1, 2, figsize = (10, 3.5))
    ax[0].plot(range(1, len(ratio) + 1),
               ratio,
               marker = "o")
    ax[0].set(title = "Scree plot",
              xlabel = "component",
              ylabel = "explained variance ratio")
    ax[1].plot(range(1, len(cum) + 1),
               cum,
               marker ="o")
    ax[1].set(title = "Cumulative",
              xlabel = "components",
              ylabel = "share of variance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_scree.png", dpi=150)
    plt.close(fig)
    for k in (2,3,4,5,6,7,8,9,10,12,14,16,18,20):
        print(f"  {k:2d} components explain {cum[k - 1]:.1%} of the variance")
    return pca, ratio


def kmeans_sweep(Z, ks = range(2, 11)):
    """Elbow (inertia) and silhouette for each k (L13 'Choosing k'), in the PCA space Z.

    n_init=10: k-means only finds a local optimum, so it is restarted 10 times and the run
    with the lowest inertia is kept. random_state=SEED makes the sweep reproducible.
    inertia    = within-cluster sum of squared distances; always falls as k grows, so look
                 for the bend (elbow), not the minimum.
    silhouette = mean over points of (b - a) / max(a, b), with a = mean distance to the own
                 cluster and b = to the nearest other cluster; higher = better separated.
                 Computed in Z, so it can only be compared across k for the SAME Z.
    """
    rows = []
    for k in ks:
        km = KMeans(n_clusters = k,
                    init = "k-means++",
                    n_init = 10,
                    random_state = SEED).fit(Z)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Z, km.labels_)})
    return pd.DataFrame(rows)


def plot_sweep(sweep, title_suffix, filename):
    """Elbow and silhouette side by side (notebook C2)."""
    fig, ax = plt.subplots(1, 2, figsize = (10, 3.5))
    ax[0].plot(sweep["k"],
               sweep["inertia"],
               marker = "o")
    ax[0].set(title = f"Elbow: inertia{title_suffix}",
              xlabel = "k (clusters)",
              ylabel = "within-cluster sum of squares")
    ax[1].plot(sweep["k"],
               sweep["silhouette"],
               marker = "o")
    ax[1].set(title = f"Silhouette{title_suffix}",
              xlabel = "k (clusters)",
              ylabel = "mean silhouette score")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)


def kmeans_stability(Z, k, seeds=(0, 1, 2, 3, 4)):
    """Do the clusters survive a different random start? (L13 'Sensitivity checks')

    One KMeans(k, n_init=1) per seed, then the adjusted Rand index (ARI) between every
    pair of label vectors: 5 seeds -> 10 pairs. ARI = 1.0 means identical partitions
    (cluster numbers may be permuted), about 0 means no more agreement than chance.

    n_init=1 is DELIBERATE: with n_init=10, KMeans already keeps the best of 10 starts and
    would hide the instability we want to measure.
    """
    labels = [KMeans(n_clusters = k, 
                     init = "k-means++",
                     n_init = 1, 
                     random_state = s).fit_predict(Z) for s in seeds]
    aris = [adjusted_rand_score(a, b) for a, b in combinations(labels, 2)]   # 10 pairs
    return pd.DataFrame({"k": k, "mean_ari": np.mean(aris), "min_ari": np.min(aris)}, index=[0])


def ward_comparison(Z, rows, k, year=DENDROGRAM_YEAR, plot = True):
    """Does hierarchical clustering agree with k-means? (L13 'Hierarchical clustering')

    ARI between Ward and k-means labels on ALL training country-years. Ward merges, at each
    step, the two clusters whose merge raises the within-cluster sum of squares least, i.e.
    the same objective as k-means, but built bottom-up and without random starts. If two
    methods with different search strategies find similar groups, the groups are more
    credible (L13 pitfall 2: "stable and agrees").

    plot=True also draws a dendrogram for ONE base year, so the leaves are readable country
    codes. `rows` must be in the same order as Z.
    """
    km_labels = KMeans(n_clusters = k,
                        n_init = 10,
                        random_state = SEED).fit_predict(Z)
    ward_labels = AgglomerativeClustering(n_clusters = k,
                                          linkage = "ward").fit_predict(Z)
    ari = adjusted_rand_score(km_labels, ward_labels)
    print(f"k = {k}: Ward vs k-means ARI = {ari:.3f}")

    if plot:
        # ASSUMPTION: that year's rows of the SAME Z (PCA fitted on all training years), not a
        # PCA refitted on one year, so the dendrogram shows the same space as the ARI.
        one = rows["base_year"].eq(year).to_numpy()
        L = linkage(Z[one], method = "ward")   # L[i, 2] = height (merge distance) of merge i
        # L[-(k - 1), 2] is the height of the merge that turns k groups into k - 1. The cut is
        # drawn 8 units below it so the line does not lie on top of that merge.
        # CAUTION: -8 is a visual choice for this data; it is only correct while it stays
        # above the next-lower merge height L[-k, 2].
        cut = L[-(k - 1), 2]-8
        fig, ax = plt.subplots(figsize=(7, 22))
        dendrogram(L, labels = rows.loc[one, "country"].to_list(),
                   orientation = "left",
                   color_threshold = cut,
                   leaf_font_size = 7,
                   ax = ax)
        ax.axvline(cut,
                   color = "grey",
                   linestyle = "--")
        ax.set(title = f"Ward dendrogram, base year {year} (cut at k = {k})", xlabel="merge distance")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "dendrogram.png", dpi = 150)
        plt.close(fig)
    return ari, km_labels, ward_labels


def main():
    # C1: scree plot -> N_PCA (written into config.py by hand, with the reasons)
    rows, cols = load_training_baskets()
    X = rows[cols].to_numpy()
    pca_scree(X)

    # Z = the country-years in N_PCA-dimensional PCA space; all clustering below works on Z.
    # Fitted on all training base years at once: fine for CHOOSING settings, because the
    # models never see this Z (common.add_country_features refits per split).
    Z = PCA(n_components = N_PCA, random_state = SEED).fit_transform(X)
    print(f"Z: {Z.shape[0]:,} country-years x {Z.shape[1]} PCs")

    # C2: elbow + silhouette on k = 2..10, plus the large-k sensitivity check
    # (silhouette rises again near k = 163 countries: clusters become countries, L13 pitfall 2)
    sweep = kmeans_sweep(Z, ks = K_RANGE)
    plot_sweep(sweep, "", "kmeans_k.png")
    plot_sweep(kmeans_sweep(Z, ks = K_WIDE), " (wide k)", "kmeans_k_wide.png")

    # C3: seed stability per k (one k per call -> concat)
    stability = pd.concat([kmeans_stability(Z, k) for k in K_RANGE], ignore_index=True)

    # C4: Ward agreement per k; the dendrogram only for the chosen k
    ward = pd.DataFrame({"k": list(K_RANGE),
                         "ward_ari": [ward_comparison(Z, rows, k, plot = (k == N_CLUSTERS))[0]
                                      for k in K_RANGE]})

    # C5: one row per k, spec names, full precision (rounding only when printing)
    summary = (sweep.merge(stability, on = "k")
                    .merge(ward, on = "k")
                    .rename(columns = {"mean_ari": "seed_ari_mean", "min_ari": "seed_ari_min"})
                    .assign(n_pca = N_PCA))
    summary = summary[["n_pca", "k", "inertia", "silhouette", "seed_ari_mean", "seed_ari_min", "ward_ari"]]
    summary.to_csv(TABLES_DIR / "unsupervised_summary.csv", index = False)
    print(summary.round(3).to_string(index = False))
    print(f"Chosen in config.py: N_PCA = {N_PCA}, N_CLUSTERS = {N_CLUSTERS}. Saved unsupervised_summary.csv")


if __name__ == "__main__":
    main()
