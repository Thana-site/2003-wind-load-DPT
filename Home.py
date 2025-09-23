"""
app.py - Streamlit application for interactive flow net analysis
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import json

# Import our modules
from geometry import Domain, SoilLayer, SheetPile, Excavation, BoundaryCondition, create_cofferdam_domain
from fem_solver import FDMSolver
from visualize import FlowNetVisualizer

# Page configuration
st.set_page_config(
    page_title="Flow Net Analysis Tool",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
    .plot-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'domain' not in st.session_state:
        st.session_state.domain = None
    if 'solver' not in st.session_state:
        st.session_state.solver = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'soil_layers' not in st.session_state:
        st.session_state.soil_layers = [
            {"name": "Sand", "depth_top": 0, "depth_bottom": 5, 
             "hydraulic_conductivity": 1e-5, "porosity": 0.3},
            {"name": "Clay", "depth_top": 5, "depth_bottom": 15, 
             "hydraulic_conductivity": 1e-6, "porosity": 0.4}
        ]
    if 'last_run_params' not in st.session_state:
        st.session_state.last_run_params = None


def create_sidebar():
    """Create sidebar with input parameters"""
    st.sidebar.header("⚙️ Model Configuration")
    
    # Collapsible sections
    with st.sidebar.expander("📐 Geometry", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            domain_width = st.number_input(
                "Domain Width (m)", 
                min_value=10.0, max_value=100.0, value=40.0, step=5.0,
                help="Total width of the analysis domain"
            )
            domain_depth = st.number_input(
                "Domain Depth (m)", 
                min_value=5.0, max_value=50.0, value=15.0, step=1.0,
                help="Total depth of the analysis domain"
            )
        
        with col2:
            sheet_pile_length = st.number_input(
                "Sheet Pile Length (m)", 
                min_value=2.0, max_value=30.0, value=10.0, step=1.0,
                help="Total length of sheet pile from surface"
            )
            excavation_depth = st.number_input(
                "Excavation Depth (m)", 
                min_value=1.0, max_value=min(sheet_pile_length - 1, 20.0), 
                value=6.0, step=0.5,
                help="Depth of excavation below ground surface"
            )
        
        excavation_width = st.number_input(
            "Excavation Width (m)", 
            min_value=2.0, max_value=domain_width - 10, value=10.0, step=1.0,
            help="Width between sheet piles"
        )
    
    with st.sidebar.expander("💧 Water Levels", expanded=True):
        water_level_outside = st.number_input(
            "Water Level Outside (m below GL)", 
            min_value=0.0, max_value=excavation_depth, value=2.0, step=0.5,
            help="Groundwater level outside excavation"
        )
        
        water_inside_type = st.radio(
            "Excavation Condition",
            ["Dewatered (Dry)", "Partially Filled", "Fully Filled"],
            help="Water condition inside excavation"
        )
        
        if water_inside_type == "Partially Filled":
            water_level_inside = st.number_input(
                "Water Level Inside (m below GL)", 
                min_value=excavation_depth * 0.5, 
                max_value=excavation_depth, 
                value=excavation_depth - 1.0, 
                step=0.5
            )
        elif water_inside_type == "Fully Filled":
            water_level_inside = water_level_outside
        else:  # Dewatered
            water_level_inside = excavation_depth
    
    with st.sidebar.expander("🪨 Soil Layers", expanded=False):
        st.write("Define soil layers from top to bottom:")
        
        num_layers = st.number_input(
            "Number of Layers", 
            min_value=1, max_value=5, value=2, step=1
        )
        
        soil_layers = []
        previous_bottom = 0
        
        for i in range(num_layers):
            st.write(f"**Layer {i+1}**")
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input(
                    f"Name", 
                    value=st.session_state.soil_layers[i]["name"] if i < len(st.session_state.soil_layers) else f"Layer {i+1}",
                    key=f"layer_name_{i}"
                )
                k = st.number_input(
                    f"K (m/s)", 
                    min_value=1e-12, max_value=1e-2, 
                    value=st.session_state.soil_layers[i]["hydraulic_conductivity"] if i < len(st.session_state.soil_layers) else 1e-6,
                    format="%.2e", key=f"layer_k_{i}"
                )
            
            with col2:
                depth_top = previous_bottom
                depth_bottom = st.number_input(
                    f"Bottom (m)", 
                    min_value=depth_top + 0.5, 
                    max_value=domain_depth,
                    value=min(st.session_state.soil_layers[i]["depth_bottom"] if i < len(st.session_state.soil_layers) else depth_top + 5, domain_depth),
                    step=0.5, key=f"layer_bottom_{i}"
                )
                porosity = st.number_input(
                    f"Porosity", 
                    min_value=0.1, max_value=0.6, 
                    value=st.session_state.soil_layers[i]["porosity"] if i < len(st.session_state.soil_layers) else 0.3,
                    step=0.05, key=f"layer_porosity_{i}"
                )
            
            soil_layers.append({
                "name": name,
                "depth_top": depth_top,
                "depth_bottom": depth_bottom,
                "hydraulic_conductivity": k,
                "porosity": porosity
            })
            
            previous_bottom = depth_bottom
            
            if i < num_layers - 1:
                st.divider()
        
        st.session_state.soil_layers = soil_layers
    
    with st.sidebar.expander("🔢 Numerical Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            nx = st.number_input(
                "Grid Points (X)", 
                min_value=51, max_value=501, value=201, step=50,
                help="Number of grid points in horizontal direction"
            )
        
        with col2:
            ny = st.number_input(
                "Grid Points (Y)", 
                min_value=51, max_value=501, value=151, step=50,
                help="Number of grid points in vertical direction"
            )
        
        solver_type = st.selectbox(
            "Solver Method",
            ["Finite Difference Method (FDM)", "Finite Element Method (FEM)"],
            help="Numerical method for solving flow equations"
        )
    
    st.sidebar.divider()
    
    # Run analysis button
    run_button = st.sidebar.button(
        "🚀 Run Analysis", 
        type="primary", 
        use_container_width=True
    )
    
    # Export options
    with st.sidebar.expander("💾 Export Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            export_csv = st.button("📊 Export CSV", use_container_width=True)
        
        with col2:
            export_plots = st.button("📈 Export Plots", use_container_width=True)
    
    # Return parameters
    params = {
        'domain_width': domain_width,
        'domain_depth': domain_depth,
        'sheet_pile_length': sheet_pile_length,
        'excavation_depth': excavation_depth,
        'excavation_width': excavation_width,
        'water_level_outside': water_level_outside,
        'water_level_inside': water_level_inside,
        'soil_layers': soil_layers,
        'nx': nx,
        'ny': ny,
        'solver_type': solver_type,
        'run_analysis': run_button,
        'export_csv': export_csv,
        'export_plots': export_plots
    }
    
    return params


def run_analysis(params):
    """Run the flow net analysis with given parameters"""
    
    # Create progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Create domain
        status_text.text("Creating domain geometry...")
        progress_bar.progress(20)
        
        domain = create_cofferdam_domain(
            sheet_pile_length=params['sheet_pile_length'],
            excavation_depth=params['excavation_depth'],
            excavation_width=params['excavation_width'],
            domain_width=params['domain_width'],
            domain_depth=params['domain_depth'],
            water_level_outside=params['water_level_outside'],
            water_level_inside=params['water_level_inside'],
            soil_layers_config=params['soil_layers']
        )
        
        # Update grid resolution
        domain.nx = int(params['nx'])
        domain.ny = int(params['ny'])
        domain.__post_init__()  # Recalculate derived properties
        
        # Step 2: Validate domain
        status_text.text("Validating configuration...")
        progress_bar.progress(30)
        
        warnings = domain.validate()
        if warnings:
            for warning in warnings:
                st.warning(warning)
        
        # Step 3: Create and run solver
        status_text.text("Solving groundwater flow equations...")
        progress_bar.progress(50)
        
        # Currently only FDM is implemented
        solver = FDMSolver(domain)
        H = solver.solve()
        
        # Step 4: Calculate velocities
        status_text.text("Calculating seepage velocities...")
        progress_bar.progress(70)
        
        qx, qy = solver.calculate_velocities()
        
        # Step 5: Calculate results
        status_text.text("Computing results...")
        progress_bar.progress(90)
        
        seepage = solver.calculate_seepage_discharge()
        gradients = solver.calculate_exit_gradients()
        
        # Complete
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # Store results in session state
        st.session_state.domain = domain
        st.session_state.solver = solver
        st.session_state.results = {
            'seepage': seepage,
            'gradients': gradients,
            'H': H,
            'qx': qx,
            'qy': qy
        }
        st.session_state.last_run_params = params
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return True
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Analysis failed: {str(e)}")
        return False


def display_results():
    """Display analysis results"""
    
    if st.session_state.solver is None:
        st.info("👈 Configure parameters and click 'Run Analysis' to start")
        return
    
    # Create visualizer
    viz = FlowNetVisualizer(st.session_state.domain, st.session_state.solver)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🌊 Flow Net", 
        "📈 Hydraulic Head", 
        "➡️ Velocity Field", 
        "📋 Numerical Results"
    ])
    
    with tab1:
        st.subheader("Analysis Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        results = st.session_state.results
        
        with col1:
            st.metric(
                "Total Seepage", 
                f"{abs(results['seepage'].get('excavation_bottom', 0)):.2e} m³/s/m",
                help="Seepage flow rate per unit width"
            )
        
        with col2:
            st.metric(
                "Safety Factor", 
                f"{results['gradients']['safety_factor']:.2f}",
                help="Against piping failure (>1.5 is safe)"
            )
        
        with col3:
            st.metric(
                "Max Exit Gradient", 
                f"{results['gradients']['max_exit_gradient']:.3f}",
                help="Maximum hydraulic gradient at exit points"
            )
        
        with col4:
            st.metric(
                "Mass Balance Error", 
                f"{results['seepage']['mass_balance_error']:.1f}%",
                help="Numerical accuracy check (<2% is good)"
            )
        
        # Dashboard plot
        st.pyplot(viz.plot_summary_dashboard(figsize=(16, 10)))
    
    with tab2:
        st.subheader("Flow Net Visualization")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.write("**Visualization Options**")
            num_equipotentials = st.slider(
                "Equipotentials", 
                min_value=5, max_value=30, value=15, step=1
            )
            num_streamlines = st.slider(
                "Flow Lines", 
                min_value=3, max_value=20, value=10, step=1
            )
            show_vectors = st.checkbox("Show Velocity Vectors", value=True)
            show_mesh = st.checkbox("Show Mesh", value=False)
        
        with col1:
            fig = viz.plot_flow_net(
                num_equipotentials=num_equipotentials,
                num_streamlines=num_streamlines,
                show_velocity_vectors=show_vectors,
                show_mesh=show_mesh,
                figsize=(12, 8)
            )
            st.pyplot(fig)
    
    with tab3:
        st.subheader("Hydraulic Head Distribution")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.write("**Statistics**")
            H = results['H']
            st.write(f"Min Head: {np.min(H):.2f} m")
            st.write(f"Max Head: {np.max(H):.2f} m")
            st.write(f"Mean Head: {np.mean(H):.2f} m")
            st.write(f"Std Dev: {np.std(H):.2f} m")
        
        with col1:
            st.pyplot(viz.plot_hydraulic_head(figsize=(12, 8)))
    
    with tab4:
        st.subheader("Seepage Velocity Field")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.write("**Velocity Statistics**")
            v_mag = np.sqrt(results['qx']**2 + results['qy']**2)
            st.write(f"Max Velocity: {np.max(v_mag):.2e} m/s")
            st.write(f"Mean Velocity: {np.mean(v_mag):.2e} m/s")
            st.write(f"Min Velocity: {np.min(v_mag[v_mag > 0]):.2e} m/s")
        
        with col1:
            st.pyplot(viz.plot_velocity_field(figsize=(12, 8)))
    
    with tab5:
        st.subheader("Detailed Numerical Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Seepage Flow Rates**")
            seepage_df = pd.DataFrame(
                [(k.replace('_', ' ').title(), f"{v:.6e}" if 'error' not in k else f"{v:.2f}%") 
                 for k, v in results['seepage'].items()],
                columns=["Parameter", "Value"]
            )
            st.dataframe(seepage_df, use_container_width=True)
        
        with col2:
            st.write("**Exit Gradients**")
            gradient_df = pd.DataFrame(
                [(k.replace('_', ' ').title(), f"{v:.4f}") 
                 for k, v in results['gradients'].items()],
                columns=["Location", "Gradient"]
            )
            st.dataframe(gradient_df, use_container_width=True)
        
        # Domain configuration
        st.write("**Domain Configuration**")
        config_data = {
            "Domain Width": f"{st.session_state.domain.width} m",
            "Domain Depth": f"{st.session_state.domain.depth} m",
            "Grid Resolution": f"{st.session_state.domain.nx} × {st.session_state.domain.ny}",
            "Sheet Pile Length": f"{st.session_state.last_run_params['sheet_pile_length']} m",
            "Excavation Depth": f"{st.session_state.last_run_params['excavation_depth']} m",
            "Excavation Width": f"{st.session_state.last_run_params['excavation_width']} m"
        }
        config_df = pd.DataFrame(list(config_data.items()), columns=["Parameter", "Value"])
        st.dataframe(config_df, use_container_width=True)


def handle_exports(params):
    """Handle export functions"""
    
    if params['export_csv'] and st.session_state.solver is not None:
        viz = FlowNetVisualizer(st.session_state.domain, st.session_state.solver)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flownet_results_{timestamp}.csv"
        
        try:
            viz.export_results_to_csv(filename)
            with open(filename, 'rb') as f:
                st.download_button(
                    label="📥 Download CSV Results",
                    data=f.read(),
                    file_name=filename,
                    mime='text/csv'
                )
            os.remove(filename)  # Clean up
            st.success("CSV export ready for download!")
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
    
    if params['export_plots'] and st.session_state.solver is not None:
        viz = FlowNetVisualizer(st.session_state.domain, st.session_state.solver)
        
        # Create a zip file with all plots
        # (Implementation would require additional libraries like zipfile)
        st.info("Plot export feature - would save all visualizations as PNG files")


def main():
    """Main application function"""
    
    # Page header
    st.title("💧 Flow Net Analysis Tool")
    st.markdown("**Interactive groundwater flow analysis for sheet pile excavations**")
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar with parameters
    params = create_sidebar()
    
    # Run analysis if requested
    if params['run_analysis']:
        with st.spinner("Running analysis..."):
            success = run_analysis(params)
            if success:
                st.success("✅ Analysis completed successfully!")
                st.balloons()
    
    # Handle exports
    handle_exports(params)
    
    # Display results
    display_results()
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        Flow Net Analysis Tool v1.0 | Built with Streamlit & Python<br>
        For educational and engineering analysis purposes
        </div>
        """, 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
