# Methodology: Universal Framework for vdW Gap Engineering

## Overview

## Table of Contents
1. [Theoretical Framework](#theoretical-framework)
2. [Computational Methods](#computational-methods)
3. [Machine Learning Approach](#machine-learning-approach)
4. [Validation Procedures](#validation-procedures)
5. [Error Analysis](#error-analysis)

## Theoretical Framework

### Van der Waals Interactions in Layered Materials

Layered 2D materials are held together by weak van der Waals forces, creating an opportunity for intercalation without breaking strong covalent bonds. The interlayer binding energy can be expressed as:

```
E_binding = E_vdW + E_electrostatic + E_Pauli + E_covalent
```

Where:
- `E_vdW`: Dispersion interactions (always attractive)
- `E_electrostatic`: Coulomb interactions (can be attractive or repulsive)
- `E_Pauli`: Exchange repulsion (always repulsive)
- `E_covalent`: Directional bonding (context-dependent)

### Intercalation Mechanisms

#### Covalent Stapling
Transition metal intercalants with partially filled d-orbitals can form covalent bonds with host chalcogen p-orbitals:

**Orbital Overlap Condition:**
```
|E_d - E_p| < W_hybridization
```

Where `W_hybridization` is the hybridization energy scale (~2-3 eV for d-p interactions).

**Electronic Signature:**
- New bonding/antibonding states in the band structure
- Charge density accumulation between intercalant and host
- Reduced interlayer distance
- Increased mechanical stiffness

#### Ionic Spacing
Large, electropositive intercalants donate electrons and create repulsive interactions:

**Charge Transfer Model:**
```
Q_transfer ≈ (χ_host - χ_intercalant) / (η_host + η_intercalant)
```

Where `χ` is electronegativity and `η` is chemical hardness.

**Physical Effects:**
- Electrostatic repulsion between charged layers
- Pauli repulsion from electron cloud overlap
- Increased interlayer distance
- Decreased mechanical stiffness

## Computational Methods

### Density Functional Theory (DFT) Calculations

#### Software and Settings
- **Code:** Quantum ESPRESSO v7.0+
- **Exchange-Correlation:** PBE-D3 (includes van der Waals corrections)
- **Pseudopotentials:** SSSP efficiency library
- **Plane Wave Cutoff:** 80 Ry (wavefunction), 800 Ry (charge density)
- **k-point Sampling:** Γ-centered Monkhorst-Pack grid
  - Bulk calculations: 8×8×4
  - Slab calculations: 8×8×1

#### Structural Optimization
1. **Cell Parameters:** Variable cell optimization with isotropic pressure
2. **Atomic Positions:** BFGS algorithm until forces < 0.01 eV/Å
3. **Convergence Criteria:** 
   - Energy: 10⁻⁶ eV
   - Force: 10⁻² eV/Å
   - Stress: 0.1 GPa

#### Property Calculations

**Interlayer Binding Energy:**
```
E_binding = E_total - E_layer1 - E_layer2 - E_intercalant
```

**Force Constants:**
Calculated using finite differences with displacement δ = 0.01 Å:
```
k_inter = ∂²E/∂d²|d₀
```

**Electronic Structure:**
- Band structure along high-symmetry paths
- Projected density of states (PDOS)
- Charge density analysis using Bader method

### Validation Against Experiments

#### Literature Benchmarks
- Pristine interlayer distances compared to experimental values
- Electronic band gaps validated against ARPES measurements
- Elastic constants compared to mechanical testing data

#### Convergence Testing
- k-point convergence: tested up to 12×12×6 grids
- Plane wave cutoff: tested up to 120 Ry
- Vacuum spacing: minimum 15 Å for slab calculations

## Machine Learning Approach

### Feature Engineering

#### Atomic Descriptors
Physics-based features derived from fundamental atomic properties:

1. **Electronic Properties:**
   - Electronegativity (Pauling scale)
   - Ionization energy (eV)
   - Electron affinity (eV)
   - Number of valence electrons

2. **Structural Properties:**
   - Atomic radius (Å)
   - Ionic radius (Å, multiple oxidation states)
   - Covalent radius (Å)

3. **Combined Descriptors:**
   - Electronegativity difference: Δχ = χ_host - χ_intercalant
   - Size ratio: r_intercalant / d_vdW_pristine
   - Orbital energy matching: |E_d - E_p|

#### Host Material Features
1. **Electronic Structure:**
   - Average electronegativity
   - Work function (eV)
   - Band gap (eV)
   - Effective mass

2. **Structural Properties:**
   - Pristine interlayer distance (Å)
   - Layer thickness (Å)
   - In-plane lattice parameters

### Model Architecture

#### Graph Neural Network (GNN)
Primary model for structure-property relationships:

```python
class VdWGapGNN(torch.nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.predictor = torch.nn.Linear(hidden_dim, 3)  # 3 targets
        
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.predictor(x)


#### XGBoost Baseline
Gradient boosting model for tabular features:

```python
xgb_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}


### Training Protocol

#### Data Splitting
- **Training:** 70% (stratified by host material)
- **Validation:** 15% (for hyperparameter tuning)
- **Test:** 15% (held out for final evaluation)

#### Cross-Validation
5-fold cross-validation with stratification by:
- Host material type
- Intercalant mechanism (stapling vs spacing)
- Concentration range

#### Hyperparameter Optimization
Bayesian optimization using Optuna:
- Search space: 100+ hyperparameter combinations
- Optimization metric: Mean Absolute Error on validation set
- Early stopping: 50 trials without improvement

### Model Interpretability

#### SHAP Analysis
SHapley Additive exPlanations for feature importance:

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```


#### Partial Dependence Plots
Understanding individual feature effects:

```python
from sklearn.inspection import partial_dependence
pd_results = partial_dependence(model, X_train, features=[0, 1, 2])
```

## Validation Procedures

### Internal Validation

#### Cross-Validation Metrics
- **R² Score:** Coefficient of determination
- **RMSE:** Root Mean Square Error
- **MAE:** Mean Absolute Error
- **MAPE:** Mean Absolute Percentage Error

#### Residual Analysis
- Residual vs predicted plots
- Q-Q plots for normality testing
- Heteroscedasticity testing

### External Validation

#### Literature Comparison
Systematic comparison with experimental and computational literature:
- 50+ literature values for pristine materials
- 20+ experimental intercalation studies
- Validation against other DFT studies

#### Transfer Learning Tests
Model trained on one host material tested on others:
- Training on Sb₂Te₃ → Testing on MoS₂
- Training on TMDs → Testing on post-transition metal chalcogenides

### Uncertainty Quantification

#### Prediction Intervals
Bootstrap aggregating for confidence estimates:

```python
def bootstrap_predict(model, X, n_bootstrap=100):
    predictions = []
    for i in range(n_bootstrap):
        # Resample training data
        X_boot, y_boot = resample(X_train, y_train)
        model_boot = clone(model).fit(X_boot, y_boot)
        pred = model_boot.predict(X)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return mean_pred, std_pred
```

#### Conformal Prediction
Distribution-free prediction intervals:

```python
from mapie import MapieRegressor

mapie = MapieRegressor(estimator=model, cv=5)
mapie.fit(X_train, y_train)
y_pred, y_pis = mapie.predict(X_test, alpha=0.1)  # 90% prediction intervals
```

## Error Analysis

### Systematic Errors

#### DFT Limitations
1. **Exchange-Correlation Functional:** PBE tends to underestimate band gaps
2. **van der Waals Corrections:** D3 correction may overestimate binding
3. **Finite Size Effects:** Supercell size limitations

#### Model Limitations
1. **Training Data Coverage:** Limited to calculated systems
2. **Extrapolation:** Uncertainty outside training domain
3. **Feature Completeness:** May miss important physical descriptors

### Error Propagation

#### From DFT to ML
Uncertainty in DFT calculations propagates to ML training:

```
σ²_ML = σ²_DFT + σ²_model + σ²_noise
```

#### Prediction Uncertainty
Total prediction uncertainty includes:
- Aleatoric uncertainty (data noise)
- Epistemic uncertainty (model uncertainty)
- Systematic uncertainty (method limitations)

### Quality Control

#### Outlier Detection
Multiple methods for identifying problematic data:
1. **Statistical:** Z-score > 3σ
2. **Physics-based:** Unphysical property values
3. **Model-based:** High prediction uncertainty

#### Data Validation Pipeline
Automated checks for data quality:
1. Convergence verification
2. Energy consistency checks
3. Structural reasonableness
4. Electronic property validation

## Reproducibility

### Computational Environment
- **Operating System:** Linux (Ubuntu 20.04+)
- **Python Version:** 3.8+
- **Key Dependencies:** See requirements.txt
- **Hardware:** GPU recommended for GNN training

### Random Seeds
All random processes use fixed seeds:
- NumPy: `np.random.seed(42)`
- PyTorch: `torch.manual_seed(42)`
- Scikit-learn: `random_state=42`

### Version Control
- Git repository with tagged releases
- Docker containers for exact environment reproduction
- Conda environment files provided

## References

1. Hohenberg, P. & Kohn, W. Inhomogeneous electron gas. Phys. Rev. 136, B864–B871 (1964).
2. Kohn, W. & Sham, L. J. Self-consistent equations including exchange and correlation effects. Phys. Rev. 140, A1133–A1138 (1965).
3. Grimme, S. et al. A consistent and accurate ab initio parametrization of density functional dispersion correction (DFT-D) for the 94 elements H-Pu. J. Chem. Phys. 132, 154104 (2010).
4. Giannozzi, P. et al. QUANTUM ESPRESSO: a modular and open-source software project for quantum simulations of materials. J. Phys. Condens. Matter 21, 395502 (2009).
5. Chen, T. & Guestrin, C. XGBoost: A scalable tree boosting system. in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining 785–794 (2016).