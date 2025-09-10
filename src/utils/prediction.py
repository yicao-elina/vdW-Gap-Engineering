"""
Prediction utilities for vdW gap engineering
Contains helper functions for making predictions and analyzing results
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns

def predict_properties(intercalant: str, 
                      host: str, 
                      concentration: float = 0.125,
                      model_path: str = 'models/universal_vdw_model.pkl') -> Dict[str, float]:
    """
    Convenient wrapper function for property prediction
    
    Args:
        intercalant: Chemical symbol of intercalant
        host: Host material formula
        concentration: Intercalant concentration in ML
        model_path: Path to trained model
        
    Returns:
        Dictionary with predicted properties and metadata
    """
    from src.models.universal_design_model import UniversalDesignModel
    
    # Load model
    model = UniversalDesignModel()
    try:
        model.load_pretrained(model_path)
    except:
        print("Warning: Could not load pre-trained model. Using demo predictions.")
        return _demo_prediction(intercalant, host, concentration)
    
    # Make prediction
    results = model.predict_properties(intercalant, host, concentration)
    
    # Add metadata
    results['intercalant'] = intercalant
    results['host'] = host
    results['concentration'] = concentration
    results['prediction_method'] = model.model_type
    
    return results

def _demo_prediction(intercalant: str, host: str, concentration: float) -> Dict[str, float]:
    """Generate demo predictions for testing purposes"""
    
    # Simplified prediction logic based on known trends
    staplers = ['Cr', 'V', 'Ti', 'Fe', 'Co', 'Ni']
    spacers = ['Cs', 'Ba', 'K', 'Ca', 'Rb']
    
    if intercalant in staplers:
        delta_d_vdw = -0.3 * concentration  # Contraction
        force_constant = 50 + 200 * concentration  # Stiffening
        mechanism = 'covalent_stapling'
    elif intercalant in spacers:
        delta_d_vdw = 0.8 * concentration  # Expansion
        force_constant = 50 - 30 * concentration  # Softening
        mechanism = 'ionic_spacing'
    else:
        delta_d_vdw = 0.1 * concentration  # Minimal change
        force_constant = 50 + 10 * concentration
        mechanism = 'mixed'
    
    return {
        'delta_d_vdw': delta_d_vdw,
        'force_constant': force_constant,
        'formation_energy': -1.5 * concentration,  # Generally favorable
        'mechanism': mechanism,
        'confidence': 0.75,
        'intercalant': intercalant,
        'host': host,
        'concentration': concentration,
        'prediction_method': 'demo'
    }

def batch_predict(intercalant_list: List[str],
                 host_list: List[str],
                 concentration_list: List[float] = None) -> pd.DataFrame:
    """
    Make predictions for multiple intercalant-host combinations
    
    Args:
        intercalant_list: List of intercalant symbols
        host_list: List of host materials
        concentration_list: List of concentrations (optional)
        
    Returns:
        DataFrame with all predictions
    """
    if concentration_list is None:
        concentration_list = [0.125] * len(intercalant_list)
    
    results = []
    
    for i, (intercalant, host) in enumerate(zip(intercalant_list, host_list)):
        conc = concentration_list[i % len(concentration_list)]
        pred = predict_properties(intercalant, host, conc)
        results.append(pred)
    
    return pd.DataFrame(results)

def create_design_space_grid(electronegativity_range: Tuple[float, float] = (0.5, 2.5),
                           atomic_radius_range: Tuple[float, float] = (1.0, 3.0),
                           grid_size: int = 50) -> np.ndarray:
    """
    Create a grid for design space visualization
    
    Args:
        electronegativity_range: Min and max electronegativity values
        atomic_radius_range: Min and max atomic radius values
        grid_size: Number of grid points per dimension
        
    Returns:
        Grid arrays for visualization
    """
    en_values = np.linspace(electronegativity_range[0], electronegativity_range[1], grid_size)
    radius_values = np.linspace(atomic_radius_range[0], atomic_radius_range[1], grid_size)
    
    EN, RADIUS = np.meshgrid(en_values, radius_values)
    
    return EN, RADIUS

def classify_intercalation_regime(delta_d_vdw: float, 
                                force_constant: float,
                                threshold_gap: float = 0.1,
                                threshold_stiffness: float = 10.0) -> str:
    """
    Classify intercalation behavior based on predicted properties
    
    Args:
        delta_d_vdw: Change in vdW gap distance
        force_constant: Change in interlayer force constant
        threshold_gap: Threshold for significant gap change
        threshold_stiffness: Threshold for significant stiffness change
        
    Returns:
        Classification string
    """
    if delta_d_vdw < -threshold_gap and force_constant > threshold_stiffness:
        return 'strong_stapling'
    elif delta_d_vdw < -threshold_gap:
        return 'weak_stapling'
    elif delta_d_vdw > threshold_gap and force_constant < -threshold_stiffness:
        return 'strong_spacing'
    elif delta_d_vdw > threshold_gap:
        return 'weak_spacing'
    else:
        return 'neutral'

# Example usage
if __name__ == "__main__":
    print("vdW Gap Engineering - Prediction Utilities")
    print("=" * 50)
    
    # Single prediction example
    result = predict_properties('Cr', 'Sb2Te3', 0.125)
    print(f"\nSingle Prediction Example:")
    print(f"Intercalant: {result['intercalant']}")
    print(f"Host: {result['host']}")
    print(f"Δd_vdW: {result['delta_d_vdw']:.3f} Å")
    print(f"Force constant: {result['force_constant']:.1f} N/m")
    print(f"Mechanism: {result['mechanism']}")
    
    # Batch prediction example
    intercalants = ['Cr', 'V', 'Cs', 'Ba']
    hosts = ['Sb2Te3', 'MoS2', 'InSe', 'Sb2Te3']
    
    batch_results = batch_predict(intercalants, hosts)
    print(f"\nBatch Prediction Results:")
    print(batch_results[['intercalant', 'host', 'delta_d_vdw', 'mechanism']].to_string(index=False))