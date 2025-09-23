# Flow Net Analysis Tool

A comprehensive Python application for analyzing groundwater flow around sheet pile excavations with interactive visualization using Streamlit.

## 🌟 Features

- **Interactive Web Interface**: User-friendly Streamlit application for parameter input and visualization
- **Advanced Flow Analysis**: Finite Difference Method (FDM) solver for groundwater flow equations
- **Multiple Visualizations**:
  - Flow nets with equipotential lines and flow lines
  - Hydraulic head distribution
  - Seepage velocity fields
  - Flow channels analysis
  - Comprehensive dashboard view
- **Engineering Calculations**:
  - Seepage discharge rates
  - Exit hydraulic gradients
  - Safety factor against piping failure
  - Mass balance verification
- **Flexible Configuration**:
  - Multiple soil layers with different properties
  - Customizable geometry and water levels
  - Adjustable numerical grid resolution
- **Export Capabilities**:
  - CSV export of numerical results
  - PNG export of visualizations
  - Comprehensive analysis reports

## 📋 Requirements

```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
streamlit>=1.25.0
pandas>=1.3.0
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/flownet-analysis.git
cd flownet-analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Configure parameters in the sidebar:
   - **Geometry**: Domain dimensions, sheet pile length, excavation depth and width
   - **Water Levels**: Groundwater levels inside and outside excavation
   - **Soil Layers**: Define multiple layers with hydraulic conductivity
   - **Numerical Settings**: Grid resolution and solver options

4. Click "🚀 Run Analysis" to perform calculations

5. View results in different tabs:
   - Dashboard: Overview with key metrics
   - Flow Net: Equipotentials and flow lines
   - Hydraulic Head: Head distribution contours
   - Velocity Field: Seepage velocities
   - Numerical Results: Detailed calculations

6. Export results using the export options in the sidebar

## 📁 Project Structure

```
flownet-analysis/
│
├── app.py                 # Main Streamlit application
├── geometry.py            # Domain and geometry definitions
├── fem_solver.py          # Numerical solvers (FDM/FEM)
├── visualize.py           # Plotting and visualization functions
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
│
├── examples/             # Example configurations (optional)
│   ├── simple_cofferdam.json
│   └── complex_multilayer.json
│
└── tests/                # Unit tests (optional)
    ├── test_geometry.py
    ├── test_solver.py
    └── test_validation.py
```

## 🔧 Configuration Examples

### Basic Cofferdam
```python
domain = create_cofferdam_domain(
    sheet_pile_length=10.0,      # 10m sheet piles
    excavation_depth=6.0,         # 6m excavation
    excavation_width=10.0,        # 10m wide
    water_level_outside=2.0,      # 2m below ground
    water_level_inside=4.0,       # Dewatered to 4m
    soil_layers_config=[
        {"name": "Sand", "depth_top": 0, "depth_bottom": 5, 
         "hydraulic_conductivity": 1e-5},
        {"name": "Clay", "depth_top": 5, "depth_bottom": 15, 
         "hydraulic_conductivity": 1e-6}
    ]
)
```

## 📊 Key Outputs

1. **Seepage Discharge**: Total flow rate through excavation bottom and boundaries
2. **Exit Gradients**: Hydraulic gradients at critical locations (sheet pile toes)
3. **Safety Factor**: Against piping failure (should be > 1.5)
4. **Mass Balance Error**: Numerical accuracy check (should be < 2%)

## 🧮 Mathematical Model

The tool solves the steady-state groundwater flow equation:

```
∇·(K∇h) = 0
```

Where:
- `h` = hydraulic head
- `K` = hydraulic conductivity tensor
- `∇` = gradient operator

Seepage velocity is calculated using Darcy's law:
```
v = -K∇h
```

## 🎯 Use Cases

- **Geotechnical Engineering**: Design of excavation support systems
- **Construction Planning**: Dewatering requirements estimation
- **Risk Assessment**: Piping and heave failure analysis
- **Education**: Teaching groundwater flow concepts
- **Research**: Parametric studies of seepage behavior

## ⚠️ Limitations

- Currently implements 2D analysis (plane strain)
- Assumes steady-state flow conditions
- Isotropic soil properties within each layer
- No time-dependent analysis
- Sheet piles modeled as low-permeability barriers

## 🔄 Future Enhancements

- [ ] Finite Element Method (FEM) solver implementation
- [ ] Anisotropic permeability support
- [ ] Time-dependent (transient) analysis
- [ ] 3D flow analysis capability
- [ ] Mesh refinement algorithms
- [ ] More complex boundary conditions
- [ ] Integration with geotechnical databases

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🙏 Acknowledgments

- Built with Streamlit for interactive visualization
- Numerical methods based on classical groundwater flow theory
- Inspired by traditional hand-drawn flow net techniques

---

**Note**: This tool is for educational and preliminary analysis purposes. For critical engineering projects, always validate results with established software and professional engineering judgment.
