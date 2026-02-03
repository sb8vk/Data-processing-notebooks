# agents.md – Employee Attrition Persona Discovery

## 1. What we're building
An unsupervised workforce segmentation engine that discovers hidden employee personas through clustering and identifies which segments exhibit disproportionately high attrition risk, enabling targeted retention interventions.

## 2. How AI helps solve the business problem

- **Uncovers latent workforce structure**: Rather than relying on org charts or predefined job families, clustering reveals natural groupings based on tenure, role, compensation, and demographics—exposing personas HR may not have explicitly recognized.

- **Pinpoints high-risk segments**: By analyzing attrition rates within each persona, we can identify which groups are most vulnerable (e.g., "Stagnating Mid-Levels" with 40% attrition vs. organization average of 16%), enabling targeted retention interventions with measurable ROI.

- **Enables persona-specific strategies**: Understanding that "High-Potential Juniors" leave for different reasons than "Tenured Technical Specialists" allows Talent Development to design retention programs that actually resonate with each group rather than applying one-size-fits-all policies.

- **Scales discovery beyond intuition**: Manual personas often reflect leadership bias; algorithmic clustering surfaces data-driven segments that cut across traditional boundaries and are reproducible across reporting periods.

## 3. Key file locations and data structure

- `./mbai448_week03_assignment.ipynb` – Main analysis notebook (Steps 1–8)
- `./data/attrition.csv` – Employee dataset with features (tenure, role, salary, demographics, attrition label)
- `./agents.md` – This file (project context for Copilot)
- `./README.md` – Prototype summary, limitations, and open questions (created after Act II)
- `./copilot_chat_history.txt` – Exported interaction log demonstrating disciplined use of Copilot

**Expected data structure:**
- Rows: individual employees
- Columns: numeric (tenure, salary, age) + categorical (role, department, education) + binary label (attrition: Yes/No)
- No PII; all records anonymized

## 4. High-level execution plan

1. **Load & explore** – Ingest attrition data, inspect columns, identify label and features; validate data quality
2. **Transform** – Encode categorical variables, handle missing values, normalize numeric features, prepare feature matrix
3. **Cluster** – Apply K-Means with multiple k values; select optimal k using elbow method or silhouette score
4. **Validate clusters** – Ensure clusters are meaningful (not trivially equal in size) and replicable
5. **Visualize** – Reduce dimensionality with PCA; plot clusters in 2D space for interpretability
6. **Baseline classifier** – Train attrition predictor on full dataset; evaluate on held-out test set
7. **Per-cluster analysis** – Evaluate classifier separately within each persona; compute attrition rates by cluster; identify high-risk segments
8. **Cluster assignment model** – Train classifier to assign new employees to personas; validate accuracy on test set
9. **Document & validate** – Create README summarizing findings, limitations, and assumptions for stakeholder readiness

## 5. Code conventions and constraints

- **Simplicity first**: Prioritize readable, straightforward implementations using standard libraries (pandas, scikit-learn, matplotlib) unless instructed otherwise. Avoid custom algorithms or heavy optimization.
- **Reproducibility**: Set random seeds (`random_state=42`) for all clustering and train/test splits to ensure consistent results across notebook runs.
- **Visibility**: Prefer explicit intermediate outputs (`.head()`, `.value_counts()`, confusion matrices) over silent assignments; each step should produce inspectable artifacts.
- **No leakage**: Ensure attrition labels and cluster assignments are never included in feature matrices used for training.

## 6. Success criteria & guardrails

**When to consider this prototype "working":**
- Step 1: Dataset loads without errors; attrition label and 10+ feature columns are visible
- Step 3: Clusters are formed and assigned; no cluster is larger than 70% of total employees (avoiding trivial clustering)
- Step 5: Elbow curve or silhouette scores are computed across k=2 to k=10; chosen k is justified
- Step 6: Baseline classifier achieves >60% accuracy on test set; confusion matrix is interpretable
- Step 7: At least one cluster shows attrition rate >5pp different from organizational average
- Step 8: Cluster-assignment classifier achieves >70% accuracy; misclassified examples are inspected

**Flagged risks:**
- If all clusters have similar attrition rates, the segmentation may not be actionable; revisit feature selection or k value
- If a cluster has <10 employees, treat its metrics with skepticism (high variance)
- If classifier performance degrades significantly within clusters, the personas may be artifacts of the modeling process rather than true segments

## 7. Stakeholder context (Act III prep)

This prototype will be discussed with three roles:
- **Talent Development Lead**: Focus on fairness, ethics, and how personas may inform equitable retention programs
- **HR Analytics Associate**: Focus on data quality, statistical validity, and reproducibility across future periods
- **Line Manager**: Focus on actionability—can these personas help predict and prevent departures in their team?

Document trade-offs (e.g., cluster interpretability vs. statistical fit) clearly so these conversations are productive.
