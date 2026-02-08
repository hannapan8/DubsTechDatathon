import React from 'react';
import Section from '../components/Section';
import Embed from '../components/Embed';

const MLModel = () => {
  return (
    <Section
      title="ML Model"
    >
      <div className="model-content">

        {/* Overview */}
        <div className="model-overview">
          <p className="lead-text">
            We used unsupervised clustering to group websites based on accessibility violation patterns to identify
            high-risk domains and uncover recurring patterns of exclusion in web design.
            Rather than predicting a label, we used unsupervised machine learning to
            discover groups of websites with similar accessibility issues.
          </p>
        </div>

        {/* Tableau: Cluster Visualization */}
        <Embed
          type="tableau"
          title="Website Accessibility Clusters (PCA Projection)"
          height="700px"
          src="https://public.tableau.com/views/DubsTech/ml_cluster_dash?:showVizHome=no&:toolbar=no"
        />

        <div className="dashboard-explanation">
          <h3>What This Visualization Shows</h3>
          <p>
            Each point represents a website, positioned using PCA so that websites with similar
            accessibility violation profiles appear closer together. Colors indicate cluster
            membership from K-means (k = 4).
          </p>
          <ul>
            <li><strong>Dense cluster:</strong> low-risk websites with fewer or less severe violations</li>
            <li><strong>Separated groups:</strong> moderate/high-risk profiles with distinct violation patterns</li>
            <li><strong>Outlier:</strong> an extreme-risk website with unusually severe or frequent violations</li>
          </ul>
        </div>

        {/* Approach */}
        <div className="model-config">
          <h3>Model Approach</h3>

          <div className="summary-points">
            <div className="summary-point">
              <div className="point-marker"></div>
              <p>
                We used <strong>K-means clustering</strong> to group websites based on aggregated
                accessibility features derived from violation records.
              </p>
            </div>

            <div className="summary-point">
              <div className="point-marker"></div>
              <p>
                Each website was represented using features such as{' '}
                <strong>total violations</strong>, <strong>average severity score</strong>,{' '}
                <strong>violation category counts</strong>, and <strong>violation type distribution</strong>.
              </p>
            </div>

            <div className="summary-point">
              <div className="point-marker"></div>
              <p>
                Because feature scales differ, we standardized the data using{' '}
                <strong>z-score scaling</strong> prior to clustering.
              </p>
            </div>

            <div className="summary-point">
              <div className="point-marker"></div>
              <p>
                We selected <strong>k = 4 clusters</strong> to capture low-risk websites, moderate-risk
                websites, high-risk websites, and extreme outliers.
              </p>
            </div>

            <div className="summary-point">
              <div className="point-marker"></div>
              <p>
                To visualize cluster separation, we applied <strong>PCA</strong> to reduce the feature
                space to two dimensions while preserving similarity relationships.
              </p>
            </div>
          </div>
        </div>

        {/* Metrics (unsupervised-appropriate) */}
        <div className="model-metrics">
          <h3>Cluster Quality</h3>
          <div className="metrics-grid">

            <div className="metric-card primary">
              <div className="metric-value">0.361</div>
              <div className="metric-label">Silhouette Score</div>
              <div className="metric-description">
                Moderate but meaningful separation between clusters
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-value">4</div>
              <div className="metric-label">Clusters</div>
              <div className="metric-description">
                Low-risk, moderate-risk, high-risk, and extreme outliers
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-value">PCA</div>
              <div className="metric-label">2D Projection</div>
              <div className="metric-description">
                Used for interpretability and visualization of separation
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-value">Z-score</div>
              <div className="metric-label">Standardization</div>
              <div className="metric-description">
                Ensures features contribute comparably to clustering
              </div>
            </div>

          </div>
        </div>

        {/* Configuration */}
        <div className="model-config">
          <h3>Model Configuration</h3>
          <div className="config-grid">
            <div className="config-item">
              <div className="config-label">Pipeline</div>
              <div className="config-value">Aggregate → Scale → Cluster → Visualize</div>
            </div>

            <div className="config-item">
              <div className="config-label">Aggregation Level</div>
              <div className="config-value">Website-level feature table</div>
            </div>

            <div className="config-item">
              <div className="config-label">Scaling</div>
              <div className="config-value">Z-score standardization</div>
            </div>

            <div className="config-item">
              <div className="config-label">Clustering</div>
              <div className="config-value">K-means (k = 4)</div>
            </div>

            <div className="config-item">
              <div className="config-label">Visualization</div>
              <div className="config-value">PCA to 2 dimensions</div>
            </div>

            <div className="config-item">
              <div className="config-label">Evaluation</div>
              <div className="config-value">Silhouette score (0.361)</div>
            </div>
          </div>
        </div>

        {/* Interpretation */}
        <div className="analysis-summary">
          <h3>Interpretation</h3>
          <div className="summary-points">
            <div className="summary-point">
              <div className="point-marker"></div>
              <p>
                The silhouette score of <strong>0.361</strong> suggests moderate separation: accessibility
                violation profiles form distinguishable groups, though some overlap exists between
                medium- and high-risk websites.
              </p>
            </div>

            <div className="summary-point">
              <div className="point-marker"></div>
              <p>
                The PCA visualization supports this structure, showing a dense low-risk cluster,
                several moderate- and high-risk clusters, and a clear extreme-risk outlier.
              </p>
            </div>
          </div>
        </div>

      </div>
    </Section>
  );
};

export default MLModel;
