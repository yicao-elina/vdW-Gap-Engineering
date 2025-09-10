# Contributing to vdW Gap Engineering Framework

Thank you for your interest in contributing to our universal framework for van der Waals gap engineering! This project aims to accelerate materials discovery through AI-driven computational methods.

## 🚀 Ways to Contribute

### 🐛 Bug Reports
- Found a calculation error or code bug?
- Unexpected behavior in predictions?
- Documentation unclear or incorrect?

### 💡 Feature Requests
- New intercalant or host materials to study
- Additional ML models or analysis methods
- Improved visualization or user interface
- Integration with other materials databases

### 📊 Data Contributions
- New DFT calculations following our protocols
- Experimental validation data
- Literature data compilation
- Property measurements

### 🔬 Research Collaborations
- Experimental validation of predictions
- Extension to new material classes
- Novel applications and use cases
- Theoretical method improvements

## 📋 Getting Started

### Prerequisites
1. **Python Environment:** Python 3.8+ with scientific computing packages
2. **Git:** For version control and collaboration
3. **Materials Science Background:** Understanding of DFT and 2D materials helpful
4. **Optional:** Access to computational resources for DFT calculations

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/your-username/vdW-Gap-Engineering.git
cd vdW-Gap-Engineering

# Create development environment
conda create -n vdw-gap python=3.8
conda activate vdw-gap
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Additional development tools

# Install pre-commit hooks
pre-commit install
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test categories
pytest tests/test_models.py  # ML model tests
pytest tests/test_dft.py     # DFT calculation tests
pytest tests/test_utils.py   # Utility function tests
```

## 📝 Contribution Guidelines

### Code Style
We follow PEP 8 with some modifications:
- **Line Length:** 88 characters (Black formatter default)
- **Imports:** Use isort for consistent import ordering
- **Docstrings:** Google style docstrings for all functions
- **Type Hints:** Required for all public functions

```python
def predict_properties(
    intercalant: str, 
    host: str, 
    concentration: float = 0.125
) -> Dict[str, float]:
    """
    Predict material properties for intercalation system.
    
    Args:
        intercalant: Chemical symbol of intercalant atom
        host: Host material formula (e.g., 'MoS2')
        concentration: Intercalant concentration in ML
        
    Returns:
        Dictionary containing predicted properties and metadata
        
    Raises:
        ValueError: If intercalant or host not recognized
    """
```

### Git Workflow
1. **Branch Naming:**
   - `feature/description` for new features
   - `bugfix/description` for bug fixes
   - `docs/description` for documentation
   - `experiment/description` for research experiments

2. **Commit Messages:**
   ```
   type(scope): brief description
   
   Longer explanation if needed, including:
   - What changed and why
   - Any breaking changes
   - References to issues (#123)
   ```

3. **Pull Request Process:**
   - Create feature branch from `main`
   - Make changes with clear, atomic commits
   - Add tests for new functionality
   - Update documentation as needed
   - Submit PR with detailed description

### Testing Requirements
All contributions must include appropriate tests:

#### Unit Tests
```python
def test_predict_properties():
    """Test basic property prediction functionality."""
    result = predict_properties('Cr', 'Sb2Te3', 0.125)
    
    assert 'delta_d_vdw' in result
    assert 'mechanism' in result
    assert result['mechanism'] == 'covalent_stapling'
    assert -1.0 < result['delta_d_vdw'] < 0.0  # Reasonable range
```

#### Integration Tests
```python
def test_full_prediction_pipeline():
    """Test complete workflow from data to prediction."""
    # Load test data
    df = pd.read_csv('tests/data/test_dataset.csv')
    
    # Train model
    model = UniversalDesignModel()
    model.train(df)
    
    # Make predictions
    results = model.predict_batch(test_cases)
    
    # Validate results
    assert len(results) == len(test_cases)
    assert all('confidence' in r for r in results)
```

## 📊 Data Contribution Guidelines

### DFT Calculations
If contributing new DFT data, please follow these standards:

#### Computational Parameters
- **Exchange-Correlation:** PBE-D3 or equivalent vdW-corrected functional
- **Convergence:** Energy < 10⁻⁶ eV, Forces < 0.01 eV/Å
- **k-points:** Sufficient for convergence (typically 8×8×4 for bulk)
- **Cutoff:** 80 Ry for wavefunctions, 800 Ry for charge density

#### Required Properties
For each system, please calculate:
1. **Structural:** Optimized atomic positions and cell parameters
2. **Energetic:** Total energy, formation energy, binding energy
3. **Mechanical:** Interlayer force constants (finite differences)
4. **Electronic:** Band structure, DOS, Bader charges

#### Data Format
```json
{
  "system_id": "Cr_Sb2Te3_0.125ML",
  "host": "Sb2Te3",
  "intercalant": "Cr",
  "concentration": 0.125,
  "calculation_details": {
    "software": "Quantum ESPRESSO 7.0",
    "functional": "PBE-D3",
    "cutoff_wfc": 80,
    "cutoff_rho": 800,
    "kpoints": [8, 8, 4]
  },
  "results": {
    "total_energy": -1234.567,
    "formation_energy": -1.23,
    "interlayer_distance": 3.45,
    "force_constant": 123.4,
    "band_gap": 0.89
  },
  "files": {
    "input": "path/to/input.in",
    "output": "path/to/output.out",
    "structure": "path/to/structure.cif"
  }
}
```

### Experimental Data
Experimental contributions are highly valuable for validation:

#### Preferred Measurements
- **Structural:** X-ray diffraction, STEM imaging
- **Electronic:** ARPES, STS, transport measurements
- **Mechanical:** Nanoindentation, AFM force curves
- **Thermal:** Thermal conductivity, thermoelectric properties

#### Data Requirements
- Clear experimental conditions and uncertainties
- Raw data files when possible
- Detailed methodology description
- Literature references for comparison

## 🔬 Research Collaboration

### Academic Partnerships
We welcome collaborations with:
- **Experimental Groups:** For validation and new applications
- **Theory Groups:** For method development and understanding
- **Industry Partners:** For real-world applications

### Publication Policy
- **Open Science:** All code and data openly available
- **Attribution:** Contributors acknowledged in publications
- **Co-authorship:** Significant contributions may warrant co-authorship
- **Preprints:** Results shared via arXiv before journal submission

### Intellectual Property
- **MIT License:** All code contributions under MIT license
- **Data:** Creative Commons Attribution (CC-BY) for datasets
- **Publications:** Open access preferred when possible

## 📞 Communication

### Getting Help
- **GitHub Issues:** For bugs, feature requests, and general questions
- **Discussions:** For broader conversations and ideas
- **Email:** [your-email@jhu.edu](mailto:your-email@jhu.edu) for private matters
- **Slack/Discord:** Real-time chat (link in repository)

### Community Guidelines
- **Be Respectful:** Treat all contributors with respect and kindness
- **Be Constructive:** Provide helpful feedback and suggestions
- **Be Patient:** Remember that everyone is learning and contributing voluntarily
- **Be Inclusive:** Welcome contributors from all backgrounds and experience levels

### Regular Meetings
- **Weekly Lab Meetings:** Tuesdays 2-3 PM EST (open to collaborators)
- **Monthly Community Calls:** First Friday of each month
- **Annual Workshop:** Summer workshop for major updates and planning

## 🏆 Recognition

### Contributor Acknowledgment
- **README:** All contributors listed in main README
- **Publications:** Significant contributors acknowledged in papers
- **Presentations:** Contributors recognized in conference talks
- **Website:** Contributor profiles on project website

### Contribution Types
We recognize many forms of contribution:
- 💻 **Code:** New features, bug fixes, optimizations
- 📊 **Data:** DFT calculations, experimental measurements
- 📖 **Documentation:** Tutorials, examples, method descriptions
- 🐛 **Testing:** Bug reports, test cases, validation studies
- 💡 **Ideas:** Feature suggestions, research directions
- 🎨 **Design:** Visualizations, user interface improvements
- 📢 **Outreach:** Presentations, blog posts, social media

## 📜 Code of Conduct

### Our Pledge
We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, visible or invisible disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards
**Positive Behaviors:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable Behaviors:**
- Harassment, trolling, or discriminatory comments
- Publishing others' private information without permission
- Professional misconduct or conflicts of interest
- Any conduct that could reasonably be considered inappropriate

### Enforcement
Instances of abusive, harassing, or otherwise unacceptable behavior may be reported to the project maintainers. All complaints will be reviewed and investigated promptly and fairly.

---

## 🎯 Quick Start Checklist

Ready to contribute? Here's your quick start checklist:

- [ ] Fork the repository
- [ ] Set up development environment
- [ ] Read relevant documentation
- [ ] Pick an issue or propose new feature
- [ ] Create feature branch
- [ ] Make changes with tests
- [ ] Submit pull request
- [ ] Respond to review feedback
- [ ] Celebrate your contribution! 🎉

Thank you for helping advance materials science through open collaboration!