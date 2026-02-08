# Web Accessibility Risk Analysis
Datathon Project — [AccessGuru Dataset](https://b2find.eudat.eu/dataset/e0aa764f-959c-51a9-9f39-60ac02bbb1c7?utm_source=chatgpt.com)

Project link: https://dubstech-datathon-2026.vercel.app/
Tableau visualizations:
- https://public.tableau.com/app/profile/nicole.ham5109/viz/DubsTech/ml_cluster_dash?publish=yes
- https://public.tableau.com/app/profile/tiffany.guan6074/viz/DatathonGraphs/Dashboard2

## Introduction

Technology is not neutral — digital systems often work well for some users while creating barriers for others. This project analyzes accessibility violations across real-world websites using the **AccessGuru dataset**, which contains over **3,500 accessibility violations collected from 448 websites** across domains including health, education, government, news, technology, and e-commerce.

Each violation is categorized according to **WCAG 2.1 accessibility guidelines**, including:
- Syntactic violations
- Semantic violations
- Layout violations

Our goal is to explore:
- Where accessibility failures occur most frequently
- Which domains face greater accessibility risk
- What patterns of exclusion exist in web design
- How machine learning can help identify accessibility-risk profiles

Rather than predicting labels, we focus on discovering **system-level accessibility patterns across websites**.

---

## Methodology

We combined **data analysis, visualization, and unsupervised machine learning** to study accessibility violations.

Our workflow included:

1. Cleaning and aggregating accessibility violations at the website level
2. Exploring violation distributions across domains and categories
3. Applying clustering to group websites by accessibility risk
4. Building visualizations to communicate accessibility patterns

This approach allowed us to move from individual accessibility errors to **higher-level insights about digital inclusion**.

---

## Machine Learning Approach

We used **K-means clustering** to group websites based on accessibility violation patterns.

Each website was represented using features such as:
- Total accessibility violations
- Average violation severity score
- Violation category counts
- Violation type distribution

Because these features exist on different scales, we applied **z-score standardization** before clustering.

We selected **k = 4 clusters** representing:
- Low-risk websites
- Moderate-risk websites
- High-risk websites
- Extreme outliers

To interpret cluster separation, we used **Principal Component Analysis (PCA)** to reduce the feature space to two dimensions.

Cluster quality was evaluated using the **silhouette score**, which measured **0.361**, indicating **moderate but meaningful cluster separation**.

---

## Data Visualization Findings

Visualization analysis revealed consistent accessibility patterns:

- **Syntactic violations are the most common across all domains**
- **Semantic violations occur less frequently**
- **Layout violations are comparatively rare**

This suggests that many accessibility barriers stem from **implementation-level issues**, such as:
- Missing HTML attributes
- Improper ARIA usage
- Invalid markup

Domain comparisons revealed:

- News and media websites appear more frequently in **medium- and high-risk clusters**
- Technology and research websites also appear in higher-risk groups
- Educational and government websites are more concentrated in the **low-risk cluster**
- Government websites show **fewer total violations but more moderate-to-serious accessibility issues when violations occur**

These patterns highlight **invisible accessibility barrier**
