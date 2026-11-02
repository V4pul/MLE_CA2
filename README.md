# MLE_CA2
 Customer Segmentation Using RFM Analysis, K-Means and DBSCAN

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)
![Status](https://img.shields.io/badge/Type-College%20CA%20Project-green)

**Course Assessment:** Machine Learning (MLE) – Continuous Assessment
**Author:** Vipul Kumar, B.Tech CSE (AIML), Section E, KIET Group of Institutions

---

## 1. Overview

This project segments the customers of a retail marketing dataset using **RFM analysis** (Recency, Frequency, Monetary) and two unsupervised learning algorithms, **K-Means** and **DBSCAN**. The segments are then profiled so that each group can be given a suitable marketing action.

## 2. Problem Statement and Objectives

Treating all customers the same wastes marketing budget. The goal is to group customers by purchasing behaviour so campaigns can be targeted.

Objectives:
- Engineer Recency, Frequency and Monetary features from raw transaction summaries.
- Scale the features and choose the number of K-Means clusters with the Elbow Method and Silhouette Score.
- Apply DBSCAN and examine how it treats noise.
- Visualise the clusters in 2D using PCA and t-SNE.
- Profile each cluster and attach a marketing persona and action.

## 3. Dataset

| Item | Details |
|---|---|
| File | `marketing_campaign.csv` |
| Format | Tab-separated values (read with `sep='\t'`) |
| Size | 2,240 customers, 29 columns |
| Missing values | 24 missing values in `Income` (median-filled; used only for profiling) |
| Duplicate rows | 0 |

Column groups used in this project:
- **Spending:** `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`
- **Purchases:** `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumDealsPurchases`
- **Recency:** `Recency` (days since last purchase, supplied in the data)
- **Profiling only:** `Income`, `NumWebVisitsMonth`

The dataset is not included in this repository. Place your copy of `marketing_campaign.csv` in the same folder as the notebook before running it.

## 4. Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (`StandardScaler`, `KMeans`, `DBSCAN`, `PCA`, `TSNE`, `silhouette_score`)
- Jupyter Notebook

## 5. RFM Methodology

| Feature | Definition used in this project |
|---|---|
| **Recency (R)** | The `Recency` column: days since the customer's last purchase. Lower is better. |
| **Frequency (F)** | `NumWebPurchases + NumCatalogPurchases + NumStorePurchases + NumDealsPurchases` |
| **Monetary (M)** | `MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProds` |

Notes on the design:
- The dataset has no invoice-level history, so Frequency is a **proxy**: the total of purchases recorded across channels, not a count of unique orders.
- Rows with missing, infinite or negative RFM values are dropped.
- Extreme values in Frequency and Monetary are removed with the **1.5 × IQR rule**. This left **2,235 of 2,240 customers** (5 removed).
- Features are standardised with `StandardScaler` before clustering because K-Means and DBSCAN are distance-based.

Summary of the retained RFM data:

| | Recency | Frequency | Monetary |
|---|---|---|---|
| Mean | 49.14 | 14.83 | 602.22 |
| Std | 28.98 | 7.63 | 597.86 |
| Min | 0 | 0 | 5 |
| Median | 49 | 15 | 395 |
| Max | 99 | 39 | 2,486 |

## 6. Project Workflow

1. Import libraries and configure plotting.
2. Load `marketing_campaign.csv`.
3. Clean data and engineer R, F and M features.
4. Scale the features.
5. Elbow Method for K-Means (K = 2 to 9).
6. Silhouette Score comparison to pick K.
7. Fit the final K-Means model.
8. Fit DBSCAN.
9. Reduce dimensions with PCA and t-SNE and plot the clusters.
10. Profile the K-Means clusters and assign personas.
11. Compare K-Means and DBSCAN assignments.

All random components use `random_state=42`, so results are reproducible.

## 7. Algorithms

### K-Means
K-Means partitions the data into K clusters by repeatedly assigning each point to its nearest centroid and recomputing the centroids, minimising the within-cluster sum of squares (inertia). K must be chosen in advance, so this project uses:
- **Elbow Method:** plots inertia against K.
- **Silhouette Score:** measures how well each point fits its own cluster compared with the nearest other cluster. The K with the highest score is selected.

Settings: `n_clusters=best_k`, `n_init=10`, `random_state=42`.

### DBSCAN
DBSCAN groups points that lie in dense regions and labels isolated points as **noise (-1)**. It does not need K, but it depends on `eps` (neighbourhood radius) and `min_samples` (minimum neighbours for a core point).

Settings used: `eps=0.8`, `min_samples=8`.

## 8. Dimensionality Reduction for Visualisation

- **PCA** projects the scaled data onto the directions of maximum variance. It is linear and the explained variance is directly interpretable.
- **t-SNE** is a non-linear method that preserves local neighbourhoods, which makes it good for viewing cluster structure. Its axes have no direct meaning, and distances between far-apart groups should not be over-interpreted. Perplexity is set to `min(30, max(5, (n-1)//3))`, which is 30 here, with `init='pca'`.

The notebook plots both K-Means and DBSCAN labels on the PCA and t-SNE projections (a 2 × 2 grid).

## 9. Results

### 9.1 Choosing K for K-Means

| K | Silhouette Score |
|---|---|
| **2** | **0.4192** |
| 3 | 0.3784 |
| 4 | 0.3779 |
| 5 | 0.3700 |
| 6 | 0.3541 |
| 7 | 0.3356 |
| 8 | 0.3423 |
| 9 | 0.3395 |

K = 2 has the highest silhouette score and was selected.

### 9.2 K-Means cluster profiles (original, unscaled values)

| Cluster | Customers | Avg Recency (days) | Avg Frequency (proxy) | Avg Monetary |
|---|---|---|---|---|
| 0 | 1,184 | 48.92 | 8.78 | 132.21 |
| 1 | 1,051 | 49.38 | 21.65 | 1,131.71 |

Supplementary characteristics (not used for clustering):

| Cluster | Avg Income | Avg Web Visits / Month |
|---|---|---|
| 0 | 37,669.04 | 6.30 |
| 1 | 68,487.84 | 4.22 |

### 9.3 DBSCAN

- Clusters found (excluding noise): **1**
- Noise points: **0.18%** of customers (about 4 of 2,235)

### 9.4 PCA

The first two principal components together explain **91.86%** of the variance of the three scaled RFM features.

### 9.5 K-Means vs DBSCAN

| K-Means cluster | Share labelled noise by DBSCAN | Share in DBSCAN cluster 0 |
|---|---|---|
| 0 | 0.1% | 99.9% |
| 1 | 0.3% | 99.7% |

DBSCAN places almost every customer in a single cluster, so it does not recover the low-spend / high-spend split that K-Means finds.

## 10. Customer Personas and Marketing Recommendations

The two K-Means clusters differ mainly in **Frequency and Monetary value**. Recency is almost identical (about 49 days), so it does not separate the groups.

| Cluster | Profile from the statistics | Suggested direction |
|---|---|---|
| **1** (1,051 customers) | About 2.5× the purchase count and about 8.6× the spend of Cluster 0, with higher income and fewer web visits per month | Treat as the high-value group: loyalty rewards, early access to new products, premium and cross-sell offers, referral incentives |
| **0** (1,184 customers) | Low spend and low purchase count, lower income, more web visits per month | Treat as the low-engagement group: onboarding offers, best-seller recommendations, bundles and a second-purchase incentive, using the web channel where they already browse |

Note: the notebook's rule-based persona function labels **both** clusters "Potential-Growth Customers". With only two clusters, its percentile-rank thresholds cannot tell them apart. The table above is based on the actual cluster statistics instead; see Limitations.

## 11. Key Findings

- Silhouette analysis supports **two** well-separated customer groups based on spend and purchase count.
- The high-value group (Cluster 1) buys more often and spends far more, and has higher average income.
- Recency does not distinguish the groups in this dataset.
- DBSCAN with `eps=0.8` and `min_samples=8` treats the data as one dense region with very little noise, so K-Means is the more useful algorithm here.
- PCA retains most of the information in just two components (91.86%).

## 12. Limitations

- **Frequency is a proxy.** It sums purchases across channels (including deal purchases) rather than counting unique orders.
- **Only K = 2 to 9 was tested**, and K was chosen on silhouette score alone. A higher K may give more actionable segments even though its silhouette is lower.
- **DBSCAN parameters were set manually** (`eps=0.8`, `min_samples=8`) and not tuned systematically.
- **Evaluation is limited to the Silhouette Score.** Davies-Bouldin and Calinski-Harabasz were not computed, and silhouette-style metrics are not well defined for a DBSCAN result with a single cluster plus noise.
- **Feature skew was not transformed.** Only standardisation was applied, with no log transform.
- **The automated persona rule** does not produce distinct labels for two clusters.

## 13. Future Improvements

- Tune DBSCAN with a k-distance plot and a grid over `eps` and `min_samples`.
- Add Davies-Bouldin and Calinski-Harabasz scores to compare algorithms.
- Apply a log or Box-Cox transform to reduce skewness before scaling.
- Try K = 3 to 5 and judge them by business interpretability as well as silhouette.
- Include campaign response columns (`Response`, `AcceptedCmp1`–`5`) to evaluate which segments respond to offers.
- Try other methods such as Gaussian Mixture Models or hierarchical clustering.

## 14. Installation and Execution

1. Clone or download this repository.
2. Install the dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Put `marketing_campaign.csv` in the same folder as the notebook.
4. Launch Jupyter and run all cells from top to bottom:
   ```bash
   jupyter notebook ca2-machinelearning.ipynb
   ```

## 15. Repository Structure

```text
.
├── ca2-machinelearning.ipynb   # Full analysis notebook
├── marketing_campaign.csv      # Dataset (add locally)
└── README.md
