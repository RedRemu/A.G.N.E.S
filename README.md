# A.G.N.E.S. - Adaptive Grid Neural Engineering System v4.2

**Smart Grid Stability Intelligence**

A production grade stacking hybrid ensemble for predicting stability in a 4 node Decentral Smart Grid Control (DSGC) network. Built as a complete ML pipeline covering feature engineering, Bayesian hyperparameter optimisation, probability calibration, conformal prediction, adversarial robustness, streaming deployment simulation, sequential change detection, and explainability.

```
Developed by : Husain Ali Al Hashem (2160425)
Supervisor   : Dr. Shamsul Masum
Institution  : University of Portsmouth
Programme    : BEng Electrical & Renewable Energy Engineering
Year         : 2025–2026
```

---

# Key Results
| Metric | HYBRID | SVM | RF | LGBM |
| AUC | 1.0000 | 0.9999 | 1.0000 | 1.0000 |
| F1 (Unstable) | 0.9997 | 0.9993 | 0.9995 |0.9999|
| Brier Score | 0.0002 | 0.0007 | 0.0010 | 0.0002 |
| ECE | 0.0003 | 0.0017 | 0.0056 | 0.0010         |

Conformal coverage: **99.97%** at α = 0.05 with 100% singleton rate.

Paired bootstrap AUC comparison (HYBRID vs LGBM): p = 0.989, not statistically significant on static data. The Hybrid advantage emerges under operational stress.

---

# Architecture

The pipeline runs 22 stages sequentially, producing 20+ versioned output artifacts.

```
1.  Data Loading           → 60,000 samples from UCI DSGC dataset
2.  Feature Engineering    → 12 raw → 48 physics informed features
3.  RFECV Selection        → 48 → 14 features (F_gain_mean dominant)
4.  Data Splitting         → 60% train / 20% validation / 20% test (stratified)
5.  Bayesian HPO           → Optuna with TPE sampler (25 trials per model)
6.  Model Training         → SVM (RBF), RF, LightGBM, Logistic Regression
7.  Calibration            → Platt scaling (SVM), Isotonic regression (RF, LGBM, LR)
8.  Stacking Hybrid        → SVM + RF base learners, Logistic meta learner
9.  Test Evaluation        → Full metric suite across all models
10. Threshold Optimisation → Cost sensitive + Youden index + 3 level risk
11. Conformal Prediction   → Split conformal with finite sample correction
12. Significance Testing   → Paired bootstrap AUC (2000 resamples)
13. Learning Curves        → AUC vs training size for all models
14. Cross Validation       → 5 fold stratified (CPCV ready for temporal data)
15. Permutation Importance → Hybrid ensemble, 5 repeats
16. SHAP Explainability    → TreeExplainer on RF and LGBM with interactions
17. Stress Testing         → Noise, OOD, boundary, Monte Carlo (all models)
18. FGSM Adversarial       → Gradient based attack at 6 epsilon levels
19. Streaming Simulation   → 120 batch deployment with drift + SCADA corruption
20. Generalisation Suite   → Synthetic DSGC, cross regime, PSI/KL drift, CUSUM, Page Hinkley
21. Auto Stabiliser        → Adam optimiser for corrective grid control
22. Browser Export         → JSON model bundle for web deployment
```

# Feature Engineering (v4)

Physics informed features derived from DSGC network parameters:
| Feature | Formula | Meaning |
| D_eff_i | g_i / τ_i | Effective damping per node       |
| R_i | 1 / τ_i | Responsiveness per node                |
| F_gain_i | τ_i × g_i | Feedback gain (loop magnitude)  |
| H_net | CV(D_eff) | Network heterogeneity index        |
| V_weak | max(\|p_i\| / g_i) | Worst case vulnerability |

RFECV selects 14 features from 48 candidates. F_gain_mean dominates across all SHAP analyses, confirming the theoretical prediction that feedback gain is the primary stability driver.

### Stacking Hybrid Ensemble

```
                ┌─── SVM (RBF, calibrated via Platt) ──┐
Test sample ───►│                                       ├──> Logistic Meta Learner ──> P(unstable)
                └─── RF (calibrated via Isotonic) ──────┘
                                                   + g_mean, tau_mean as passthrough features
```

The meta learner receives calibrated probability outputs from SVM and RF plus two physics features (g_mean, tau_mean). Out of fold predictions during training prevent information leakage.

---

# Robustness Evaluation

# Streaming Deployment Simulation (3 Layer)

Simulates 120 batches of real time deployment with three concurrent degradation layers:

| Layer | Mechanism | Parameters |
| Concept Drift | Gradual τ escalation (aging inverters) | τ × 1.5 from batch 40                                                |
| Regime Change | Abrupt g drop (network reconfiguration) | g × 0.7 at batch 80                                                 |
| SCADA Corruption | Sensor noise, missing values, ADC quantisation, stale predictions | σ = 0.02, 5% missing, 2dp, 10% latency |

| Phase | AUC | ECE | Coverage |
| Pre drift (B0 to B39) | 0.9401 | 0.0755 | 92.03%      |
| Gradual drift (B40 to B79) | 0.9260 | 0.1161 | 87.30% |
| Post abrupt (B80 to B119) | 0.8801 | 0.1533 | 82.93% |

ECE increased 214 fold under drift. Conformal coverage fell below the 95% target during the gradual phase.

# Sequential Change Detection

Three complementary drift detectors monitor prediction quality as streaming signals:

| Detector | Monitors | Detects | Method |

| PSI | Feature distributions per batch | Distribution shift | Population Stability Index against training reference           |
| CUSUM | Rolling Brier score, rolling confidence | Mean shift in calibration quality | Cumulative sum with adaptive threshold |
| Page Hinkley | Rolling Brier score, rolling confidence | Gradual mean deviation | Cumulative deviation from running mean     |

PSI first alert: batch 55 (25 batches before abrupt shift). CUSUM and Page Hinkley provide complementary detection on prediction quality metrics.

# Adversarial Robustness (FGSM)

| ε | HYBRID | SVM | RF | LGBM |
| 0.001 | 0.03% | 0.03% | 0.07% | 0.00% |
| 0.01 | 0.53% | 1.74% | 0.09% | 0.00%  |
| 0.05 | 6.17% | 17.28% | 0.20% | 0.00% |
| 0.10 | 12.56% | 33.41% | 0.25% | 0.03%|

Values show flip rate (percentage of predictions that change class under attack).

# Noise Resilience

At σ = 0.20 Gaussian noise: HYBRID AUC = 0.9828, SVM = 0.9622, RF = 0.9670, LGBM = 0.9636. The Hybrid maintains a consistent advantage due to error decorrelation between the SVM and RF base learners.

---

# Installation

# Requirements

```
Python >= 3.10
numpy
pandas
scikit-learn >= 1.4
lightgbm >= 4.0
optuna >= 3.0
shap >= 0.44
joblib
openpyxl
scipy
matplotlib (for figure generation)
```

### Setup

```bash
pip install numpy pandas scikit-learn lightgbm optuna shap joblib openpyxl scipy matplotlib
```

The pipeline gracefully degrades if optional dependencies are missing:

| Package | If Missing |
| lightgbm | Falls back to sklearn GradientBoostingClassifier |
| optuna | Uses default hyperparameters                       |
| shap | Skips explainability analysis                        |
| matplotlib | Skips figure generation                        |

---

## Usage

### Full Pipeline

```bash
python nexus_engine_v4.py
```

Runs all 22 stages. Outputs are saved to `./artifacts/` by default. Runtime is approximately 8 to 15 minutes depending on hardware (tested on AMD Ryzen 7 4800H, 16GB RAM).

# Configuration

All parameters are defined in the `Config` dataclass at the top of the script. Key settings:

```python
@dataclass
class Config:
    random_state: int = 42              # Reproducibility seed
    test_size: float = 0.20             # Test split proportion
    cv_folds: int = 5                   # Cross validation folds
    optuna_n_trials: int = 25           # HPO trials per model
    optuna_n_jobs: int = 6              # Parallel Optuna trials
    conformal_alpha: float = 0.05       # Conformal coverage target (95%)
    psi_alert_threshold: float = 0.25   # Drift recalibration trigger
    use_cpcv: bool = False              # CPCV for temporal data (disabled for IID)
    cpcv_purge_batches: int = 0         # Purge window size
    cpcv_embargo_batches: int = 0       # Embargo gap size
```

# Parallelism
The pipeline enforces single layer parallelism to prevent thread oversubscription:

| Mode | Behaviour |
| `trials` (default) | Optuna trials run in parallel, models use 1 thread     |
| `models` | Three model HPO searches run concurrently via ThreadPoolExecutor |
| `off` | Fully sequential                                                    |
Set via `Config.parallel_mode`.

---

# Output Artifacts

All outputs are saved to `./artifacts/` (configurable via `Config.output_dir`).

# JSON Artifacts

| File | Contents |
| `metrics.json` | Full metric suite for all models (AUC, F1, Brier, ECE, confusion matrices)   |
| `run_metadata.json` | Python version, library versions, config, timestamp                     |
| `threshold_optimisation.json` | Cost optimal threshold, Youden index, 3 level risk boundaries |
| `conformal_prediction.json` | Coverage, q_hat, singleton/ambiguous rates                      |
| `significance_test.json` | Paired bootstrap AUC comparison (HYBRID vs LGBM)                   |
| `learning_curves.json` | AUC vs training size for all models                                  |
| `adversarial_robustness.json` | FGSM flip rates and AUC at 6 epsilon levels                   |
| `streaming_summary.json`|Phase summaries, drift signal first alerts (PSI, CUSUM, Page Hinkley)|
| `generalisation_report.json` | Synthetic DSGC, cross regime, drift detection, SHAP under drift|
| `shap_summary.json` | Top features and interaction values                                     |
| `stabilizer_demo.json` | Auto stabiliser corrections for critical and borderline samples      |
| `inference_latency.json` | Single sample and batch latency benchmarks                         |
| `nexus_models.json` | Browser exportable model bundle                                         |

# CSV Artifacts

| File | Contents |
| `streaming_simulation.csv` | 120 batch deployment log with all metrics, PSI, KL, CUSUM, Page Hinkley columns |
| `stress_report.csv` | Noise, OOD, boundary, Monte Carlo results for all models                               |
| `feature_importance.csv` | Permutation importance (mean and std) for 14 selected features                    |
| `shap_lgbm.csv` | SHAP values for LightGBM                                                                   |
| `rfecv_curve.csv` | AUC vs number of features during RFECV                                                   |
| `calibration_data.csv` | Calibration curve data for all models                                               |

# Binary Artifacts

| File | Contents |
| `predictor.joblib` | Full model bundle (hybrid, base models, calibrated models, feature lists, thresholds) |
| `checkpoint_hpo.joblib` | HPO parameters checkpoint for crash recovery                                     |
| `checkpoint_models.joblib` | Trained model checkpoint for crash recovery                                   |

# Figures (19 PNG at 300 DPI)

| Figure | Description |
|--------|-------------|
| fig01 | Static IID model comparison (AUC, F1, Brier, ECE) |
| fig02 | Permutation importance (top 10 features) |
| fig03 | SHAP feature importance (LightGBM) |
| fig04 | RFECV selection curve (48 → 14 features) |
| fig05 | Gaussian noise robustness (AUC vs σ) |
| fig06 | Out of distribution scaling robustness |
| fig07 | FGSM adversarial flip rates |
| fig08 | Streaming AUC degradation across 120 batches |
| fig09 | Conformal coverage degradation under drift |
| fig10 | Learning curves for all models |
| fig11 | Calibration curves |
| fig12 | Triple signal drift detection (PSI, CUSUM, Page Hinkley) |
| fig13 | Cross regime validation (4 regime splits) |
| fig14 | Synthetic DSGC generalisation |
| fig15 | SHAP stability under drift (F_gain_mean across phases) |
| fig16 | ROC curves for all models |
| fig17 | Confusion matrix heatmaps |
| fig18 | Precision recall curves |
| fig19 | Feature correlation matrix |


# Production Deployment Path

> **Note:** The pipeline implements the detection and signal generation components described below. The monitoring infrastructure (Prometheus, Grafana, Docker) and the LaSCal recalibration step are proposed architecture, not implemented code. They are described here to document the intended production deployment path.

# What the Pipeline Implements

The pipeline produces all the streaming signals needed for production monitoring. These are computed during the streaming simulation and saved to `streaming_simulation.csv`:

| Signal | Column | What It Detects                                                     |
| Feature PSI | `feature_psi` | Distribution shift in input features vs training        |
| Feature KL | `feature_kl` | Information theoretic distance from training distribution |
| CUSUM (Brier) | `cusum_brier` | Upward shift in calibration error                     |
| CUSUM (Confidence) | `cusum_confidence` | Downward shift in model certainty           |
| Page Hinkley (Brier) | `page_hinkley_brier` | Gradual degradation in calibration      |
| Page Hinkley (Confidence) |`page_hinkley_confidence`| Gradual loss of model certainty |
| Recalibration Alert | `recalibration_alert` | Boolean flag when PSI > 0.25            |

The pipeline also computes and logs first alert batch numbers for all three detectors (PSI, CUSUM, Page Hinkley) in `streaming_summary.json` and `generalisation_report.json`.

# Proposed Monitoring Architecture (Not Implemented)

For production deployment, these signals would be exported as Prometheus gauges and monitored via Grafana dashboards with Alertmanager routing:

```
Model Service ──> Prometheus Gauges ──> Prometheus ──> Alertmanager ──> PagerDuty
                  (rolling_auc)          (scrape)      (CUSUM > 50)
                  (rolling_ece)
                  (feature_psi)
                  (cusum_brier)
                  (conformal_coverage)
                        │
                        V
                     Grafana
                  (Model Health
                   Dashboard)
```

# Proposed Recalibration Triggers

| Condition | Proposed Action                                                                         |
| PSI > 0.25 on any batch | Flag distribution shift, enter heightened monitoring                      |
| CUSUM exceeds threshold for 5 minutes | Trigger LaSCal unsupervised recalibration (not implemented) |
| Conformal coverage < 90% for 5 consecutive batches | Escalate to full retrain alert                 |
| Page Hinkley exceeds threshold | Secondary confirmation of gradual drift                            |

LaSCal (Bashir et al., NeurIPS 2024) is referenced as the recalibration method but is not implemented in the pipeline. The pipeline detects when recalibration is needed and logs the trigger; the actual recalibration step is identified as future work.

# Proposed Deployment Strategy (Not Implemented)

Blue/green deployment via Docker Compose is the recommended approach for zero downtime releases. Each model build would use an immutable, never reused image tag. Rollback would be a single command: pin the previous tag and restart. Health gating via /healthz would block promotion of failing deployments. This architecture is described in the research paper but no Docker Compose configuration is included in this repository.

# Validation for Temporal Data

The pipeline supports Combinatorial Purged Cross Validation (CPCV) with configurable purge windows and embargo gaps. Disabled by default for the IID benchmark dataset. Enable via:

```python
Config(use_cpcv=True, cpcv_purge_batches=5, cpcv_embargo_batches=2)
```

Note: CPCV is a configuration option in the Config dataclass. When enabled, it would replace standard stratified CV for model selection. On the current IID dataset the results are identical to standard CV.

# Research Context

# Dataset

UCI Smart Grid Stability dataset (Arzamasov et al.), augmented with physics informed features. 60,000 samples representing 4 node DSGC operating points with 12 raw parameters (τ_1..4, g_1..4, p_1..4) and binary stability labels.

# Research Gaps Addressed

| Gap | Status | Evidence |
|-----|--------|----------|
| 1. No ML validation of DSGC mechanisms | Addressed | SHAP confirms F_gain dominance (Figures 2, 3, 15) |
| 2. No streaming retraining under drift | Substantially addressed | PSI + CUSUM + Page Hinkley detection implemented; LaSCal recalibration proposed |
| 3. No deployment pipeline | Partially addressed | Streaming simulation with SCADA corruption implemented; Prometheus/Docker proposed |
| 4. Stability modelling underdeveloped | Addressed | Robustness evaluation absent from prior work |
| 5. No hybrid physics + data approach | Addressed | Physics informed feature engineering (D_eff, F_gain, V_weak) |
| 6. Behavioural factors absent | Partially | Drift simulation approximates behavioural changes |
| 7. Feature selection and interpretability | Addressed | RFECV, SHAP, permutation importance |
| 8. Reproducibility rarely addressed | Addressed | Fixed seeds, version logging, 20+ artifacts |

# Key References

The framework draws on:

| Method | Reference | Role in Pipeline |
|--------|-----------|------------------|
| LaSCal | Bashir et al., NeurIPS 2024 | Referenced as recalibration method (not implemented)       |
| CUSUM | Page, 1954 | Implemented: sequential change detection on rolling Brier and confidence     |
| ADWIN | Bifet and Gavalda, 2007 | Referenced in literature review (not implemented)               |
| Page Hinkley | Hinkley, 1971 | Implemented: cumulative deviation from running mean                |
| CPCV | de Prado, 2018 | Config option for temporal data (disabled for IID benchmark)              |
| Conformal Prediction|Vovk et al., 2005|Implemented: split conformal with finite sample correction |
| FGSM | Goodfellow et al., 2015 | Implemented: adversarial robustness at 6 epsilon levels          |



# Reproducibility

Every run produces identical results given the same random seed (default: 42). The following are version locked:

```
Python     : 3.13.3
sklearn    : 1.8.0
LightGBM   : 4.6.0
Optuna     : 4.7.0
SHAP       : 0.50.0
NumPy      : 2.2.4
Pandas     : 2.2.3
```
All library versions are logged in `run_metadata.json`. Stage wise checkpoints (`checkpoint_hpo.joblib`, `checkpoint_models.joblib`) enable crash recovery without rerunning the full pipeline.

# License

Academic use. University of Portsmouth, 2025 to 2026.

