"""
app.py - Streamlit application for flow net analysis with corrected layer interfaces
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

# CONSTANTS
MAX_DOMAIN_DEPTH = 20.0  # Maximum analysis depth limited to 20 meters
MIN_LAYER_THICKNESS = 0.5  # Minimum thickness for a soil layer

def initialize_session_state():
    """Initialize session state variables with proper defaults"""
    if 'domain' not in st.session_state:
        st.session_state.domain = None
    if 'solver' not in st.session_state:
        st.session_state.solver = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'soil_layers' not in st.session_state:
        # Initialize with depth ranges for soil layers
        st.session_state.soil_layers = [
            {"name": "Sand", "depth_from": 0.0, "depth_to": 5.0, 
             "hydraulic_conductivity": 1e-5, "porosity": 0.35},
            {"name": "Clay", "depth_from": 5.0, "depth_to": 12.0, 
             "hydraulic_conductivity": 1e-7, "porosity": 0.45},
            {"name": "Gravel", "depth_from": 12.0, "depth_to": 20.0,
             "hydraulic_conductivity": 1e-3, "porosity": 0.30}
        ]
    if 'last_run_params' not in st.session_state:
        st.session_state.last_run_params = None


def validate_and_adjust_depth(value, min_val, max_val):
    """Validate and adjust depth values to prevent errors"""
    try:
        val = float(value)
        return max(min_val, min(val, max_val))
    except:
        return min_val


def create_sidebar():
    """Create sidebar with corrected input validation"""
    st.sidebar.header("⚙️ Model Configuration")
    
    # Domain Configuration
    with st.sidebar.expander("📐 Domain Geometry", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            domain_width = st.number_input(
                "Domain Width (m)", 
                min_value=10.0, 
                max_value=100.0, 
                value=40.0, 
                step=5.0,
                help="Total width of the analysis domain"
            )
            
            # Enforce maximum depth of 20m
            domain_depth = st.number_input(
                "Domain Depth (m)", 
                min_value=5.0, 
                max_value=MAX_DOMAIN_DEPTH,  # Limited to 20m
                value=15.0, 
                step=1.0,
                help=f"Total depth of the analysis domain (max {MAX_DOMAIN_DEPTH}m)"
            )
        
        with col2:
            # Sheet pile length cannot exceed domain depth
            max_pile_length = min(domain_depth - 0.5, MAX_DOMAIN_DEPTH - 0.5)
            sheet_pile_length = st.number_input(
                "Sheet Pile Length (m)", 
                min_value=2.0, 
                max_value=max_pile_length,
                value=min(10.0, max_pile_length), 
                step=0.5,
                help=f"Total length of sheet pile from surface (max {max_pile_length:.1f}m)"
            )
            
            # Excavation depth must be less than sheet pile length
            max_excavation_depth = min(sheet_pile_length - 1.0, domain_depth - 1.0)
            excavation_depth = st.number_input(
                "Excavation Depth (m)", 
                min_value=1.0, 
                max_value=max_excavation_depth,
                value=min(6.0, max_excavation_depth), 
                step=0.5,
                help=f"Depth of excavation below ground (max {max_excavation_depth:.1f}m)"
            )
        
        excavation_width = st.number_input(
            "Excavation Width (m)", 
            min_value=2.0, 
            max_value=domain_width - 10.0, 
            value=min(10.0, domain_width - 10.0), 
            step=1.0,
            help="Width between sheet piles"
        )
    
    # Water Levels Configuration
    with st.sidebar.expander("💧 Water Levels", expanded=True):
        water_level_outside = st.number_input(
            "Water Table Depth (m below ground)", 
            min_value=0.0, 
            max_value=min(excavation_depth, domain_depth/2),
            value=2.0, 
            step=0.5,
            help="Groundwater level outside excavation"
        )
        
        water_inside_type = st.radio(
            "Excavation Condition",
            ["Dewatered (Dry)", "Partially Filled", "Natural Water Table"],
            help="Water condition inside excavation"
        )
        
        if water_inside_type == "Partially Filled":
            water_level_inside = st.number_input(
                "Water Level Inside (m below ground)", 
                min_value=excavation_depth * 0.5, 
                max_value=excavation_depth, 
                value=min(excavation_depth - 1.0, excavation_depth * 0.75), 
                step=0.5
            )
        elif water_inside_type == "Natural Water Table":
            water_level_inside = water_level_outside
        else:  # Dewatered
            water_level_inside = excavation_depth
    
    # Soil Layers Configuration with Depth Ranges
    with st.sidebar.expander("🪨 Soil Layers (Depth Ranges)", expanded=False):
        st.write("Define soil layers by depth ranges (max depth: 20m):")
        
        # Dynamic number of layers
        num_layers = st.number_input(
            "Number of Layers", 
            min_value=1, 
            max_value=5, 
            value=min(3, len(st.session_state.soil_layers)),
            step=1
        )
        
        soil_layers = []
        current_depth = 0.0
        
        for i in range(num_layers):
            st.write(f"**Layer {i+1}**")
            
            # Get previous values if they exist
            if i < len(st.session_state.soil_layers):
                prev_layer = st.session_state.soil_layers[i]
                default_name = prev_layer["name"]
                default_k = prev_layer["hydraulic_conductivity"]
                default_porosity = prev_layer["porosity"]
                default_to = min(prev_layer["depth_to"], domain_depth)
            else:
                default_name = f"Layer {i+1}"
                default_k = 1e-6
                default_porosity = 0.35
                default_to = min(current_depth + 5.0, domain_depth)
            
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input(
                    f"Name", 
                    value=default_name,
                    key=f"layer_name_{i}"
                )
                
                # Display depth from (read-only)
                st.text(f"From: {current_depth:.1f} m")
                
                # Depth to
                max_depth_to = min(domain_depth, MAX_DOMAIN_DEPTH)
                min_depth_to = current_depth + MIN_LAYER_THICKNESS
                
                depth_to = st.number_input(
                    f"To (m)", 
                    min_value=min_depth_to,
                    max_value=max_depth_to,
                    value=min(max(default_to, min_depth_to), max_depth_to),
                    step=0.5,
                    key=f"layer_to_{i}",
                    help=f"Layer extends from {current_depth:.1f}m to this depth"
                )
            
            with col2:
                # Hydraulic conductivity with scientific notation
                k_options = {
                    "Gravel (1e-3 m/s)": 1e-3,
                    "Coarse Sand (1e-4 m/s)": 1e-4,
                    "Fine Sand (1e-5 m/s)": 1e-5,
                    "Silt (1e-6 m/s)": 1e-6,
                    "Sandy Clay (1e-7 m/s)": 1e-7,
                    "Clay (1e-8 m/s)": 1e-8,
                    "Dense Clay (1e-9 m/s)": 1e-9,
                    "Custom": None
                }
                
                k_selection = st.selectbox(
                    f"Hydraulic Conductivity",
                    options=list(k_options.keys()),
                    index=3,  # Default to Silt
                    key=f"layer_k_select_{i}"
                )
                
                if k_selection == "Custom":
                    k = st.number_input(
                        f"K (m/s)", 
                        min_value=1e-12, 
                        max_value=1e-2, 
                        value=default_k,
                        format="%.2e", 
                        key=f"layer_k_custom_{i}"
                    )
                else:
                    k = k_options[k_selection]
                
                porosity = st.slider(
                    f"Porosity", 
                    min_value=0.1, 
                    max_value=0.6, 
                    value=default_porosity,
                    step=0.05, 
                    key=f"layer_porosity_{i}"
                )
            
            soil_layers.append({
                "name": name,
                "depth_from": current_depth,
                "depth_to": depth_to,
                "hydraulic_conductivity": k,
                "porosity": porosity
            })
            
            current_depth = depth_to
            
            # Add layer transition note
            if i < num_layers - 1:
                st.info(f"Layer interface at {depth_to:.1f}m")
                st.divider()
        
        # Check if layers cover entire domain
        if current_depth < domain_depth:
            st.warning(f"⚠️ Layers only extend to {current_depth:.1f}m. Domain is {domain_depth:.1f}m deep.")
            # Add a default layer to fill the gap
            soil_layers.append({
                "name": "Base Layer",
                "depth_from": current_depth,
                "depth_to": domain_depth,
                "hydraulic_conductivity": 1e-8,
                "porosity": 0.40
            })
        
        st.session_state.soil_layers = soil_layers
    
    # Numerical Settings
    with st.sidebar.expander("🔢 Numerical Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            nx = st.selectbox(
                "Grid Resolution (X)", 
                options=[101, 151, 201, 251, 301],
                index=2,
                help="Number of grid points in horizontal direction"
            )
        
        with col2:
            ny = st.selectbox(
                "Grid Resolution (Y)", 
                options=[101, 151, 201, 251, 301],
                index=1,
                help="Number of grid points in vertical direction"
            )
        
        st.info(f"Total grid points: {nx * ny:,}")
        
        solver_type = st.selectbox(
            "Solver Method",
            ["Finite Difference Method (FDM)"],
            help="Numerical method for solving flow equations"
        )
    
    st.sidebar.divider()
    
    # Run Analysis Button
    run_button = st.sidebar.button(
        "🚀 Run Analysis", 
        type="primary", 
        use_container_width=True
    )
    
    # Export Options
    with st.sidebar.expander("💾 Export Options", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            export_csv = st.button("📊 Export CSV", use_container_width=True)
        with col2:
            export_plots = st.button("📈 Save Plots", use_container_width=True)
    
    # Return all parameters
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
    """Run the flow net analysis with corrected layer handling"""
    
    # Create progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Create domain with proper layer conversion
        status_text.text("Creating domain geometry with soil layers...")
        progress_bar.progress(15)
        
        # Convert depth range format to depth_top/depth_bottom format
        soil_layers_config = []
        for layer in params['soil_layers']:
            soil_layers_config.append({
                "name": layer["name"],
                "depth_top": layer["depth_from"],
                "depth_bottom": layer["depth_to"],
                "hydraulic_conductivity": layer["hydraulic_conductivity"],
                "porosity": layer["porosity"]
            })
        
        domain = create_cofferdam_domain(
            sheet_pile_length=params['sheet_pile_length'],
            excavation_depth=params['excavation_depth'],
            excavation_width=params['excavation_width'],
            domain_width=params['domain_width'],
            domain_depth=params['domain_depth'],
            water_level_outside=params['water_level_outside'],
            water_level_inside=params['water_level_inside'],
            soil_layers_config=soil_layers_config
        )
        
        # Update grid resolution
        domain.nx = int(params['nx'])
        domain.ny = int(params['ny'])
        domain.__post_init__()  # Recalculate derived properties
        
        # Step 2: Validate domain
        status_text.text("Validating configuration...")
        progress_bar.progress(25)
        
        warnings = domain.validate()
        if warnings:
            for warning in warnings:
                st.warning(warning)
        
        # Step 3: Create and run solver with layer-aware formulation
        status_text.text("Solving Laplace equation with layer interfaces...")
        progress_bar.progress(40)
        
        solver = FDMSolver(domain)
        H = solver.solve()
        
        # Step 4: Calculate velocities
        status_text.text("Computing Darcy velocities...")
        progress_bar.progress(60)
        
        qx, qy = solver.calculate_velocities()
        
        # Step 5: Calculate stream function
        status_text.text("Calculating stream function for flow lines...")
        progress_bar.progress(75)
        
        psi = solver.calculate_stream_function()
        
        # Step 6: Calculate results
        status_text.text("Computing seepage quantities and exit gradients...")
        progress_bar.progress(90)
        
        seepage = solver.calculate_seepage_discharge()
        gradients = solver.calculate_exit_gradients()
        
        # Complete
        progress_bar.progress(100)
        status_text.text("Analysis complete! ✓")
        
        # Store results
        st.session_state.domain = domain
        st.session_state.solver = solver
        st.session_state.results = {
            'seepage': seepage,
            'gradients': gradients,
            'H': H,
            'qx': qx,
            'qy': qy,
            'psi': psi
        }
        st.session_state.last_run_params = params
        
        # Clear progress indicators
        import time
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return True
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Analysis failed: {str(e)}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())
        return False


def display_results():
    """Display analysis results with layer-aware visualization"""
    
    if st.session_state.solver is None:
        st.info("👈 Configure parameters and click 'Run Analysis' to start")
        return
    
    # Create visualizer
    viz = FlowNetVisualizer(st.session_state.domain, st.session_state.solver)
    
    # Create tabs for different views
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
                help="Seepage flow rate through excavation bottom"
            )
        
        with col2:
            safety_factor = results['gradients']['safety_factor']
            st.metric(
                "Safety Factor", 
                f"{safety_factor:.2f}",
                delta="Safe" if safety_factor > 1.5 else "Check Required",
                delta_color="normal" if safety_factor > 1.5 else "inverse",
                help="Against piping failure (should be > 1.5)"
            )
        
        with col3:
            st.metric(
                "Max Exit Gradient", 
                f"{results['gradients']['max_exit_gradient']:.3f}",
                help="Maximum hydraulic gradient at exit points"
            )
        
        with col4:
            mass_error = results['seepage']['mass_balance_error']
            st.metric(
                "Mass Balance", 
                f"{mass_error:.1f}%",
                delta="Good" if mass_error < 2 else "Check Grid",
                delta_color="normal" if mass_error < 2 else "inverse",
                help="Numerical accuracy (<2% is good)"
            )
        
        # Dashboard plot
        st.pyplot(viz.plot_summary_dashboard(figsize=(16, 10)))
    
    with tab2:
        st.subheader("Flow Net Visualization (Orthogonal Pattern)")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.write("**Visualization Options**")
            num_equipotentials = st.slider(
                "Equipotential Lines", 
                min_value=5, max_value=30, value=15, step=1,
                help="Number of constant head contours"
            )
            num_flow_lines = st.slider(
                "Flow Lines", 
                min_value=5, max_value=25, value=12, step=1,
                help="Number of stream function contours"
            )
            show_layers = st.checkbox("Show Layer Boundaries", value=True,
                                     help="Display soil layer interfaces")
            show_vectors = st.checkbox("Show Velocity Vectors", value=False,
                                      help="Overlay velocity arrows")
            
            st.divider()
            st.info("Flow net properties:\n"
                   "• Blue: Equipotentials (const. head)\n"
                   "• Red: Flow lines (orthogonal)\n"
                   "• Brown dashed: Layer interfaces")
        
        with col1:
            fig = viz.plot_flow_net(
                num_equipotentials=num_equipotentials,
                num_flow_lines=num_flow_lines,
                show_velocity_vectors=show_vectors,
                show_layers=show_layers,
                figsize=(12, 8)
            )
            st.pyplot(fig)
    
    with tab3:
        st.subheader("Hydraulic Head Distribution")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.write("**Head Statistics**")
            H = results['H']
            stats_df = pd.DataFrame({
                'Statistic': ['Min', 'Max', 'Mean', 'Std Dev'],
                'Value (m)': [f"{np.min(H):.2f}", 
                             f"{np.max(H):.2f}",
                             f"{np.mean(H):.2f}",
                             f"{np.std(H):.3f}"]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
            
            st.divider()
            st.write("**Layer Properties**")
            for layer in st.session_state.soil_layers:
                st.write(f"**{layer['name']}**")
                st.write(f"Depth: {layer['depth_from']:.1f} - {layer['depth_to']:.1f} m")
                st.write(f"K: {layer['hydraulic_conductivity']:.1e} m/s")
        
        with col1:
            st.pyplot(viz.plot_hydraulic_head(figsize=(12, 8)))
    
    with tab4:
        st.subheader("Seepage Velocity Field")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.write("**Velocity Statistics**")
            v_mag = np.sqrt(results['qx']**2 + results['qy']**2)
            v_nonzero = v_mag[v_mag > 1e-15]
            
            stats_df = pd.DataFrame({
                'Statistic': ['Maximum', 'Mean', 'Median'],
                'Velocity (m/s)': [f"{np.max(v_mag):.2e}", 
                                  f"{np.mean(v_nonzero):.2e}",
                                  f"{np.median(v_nonzero):.2e}"]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
            
            st.divider()
            st.write("**Flow Concentration Zones**")
            # Identify high velocity zones
            high_v_threshold = np.percentile(v_nonzero, 90)
            st.info(f"90th percentile: {high_v_threshold:.2e} m/s")
        
        with col1:
            st.pyplot(viz.plot_velocity_field(figsize=(12, 8)))
    
    with tab5:
        st.subheader("Detailed Numerical Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Seepage Flow Rates**")
            seepage_data = []
            for key, value in results['seepage'].items():
                if 'error' not in key:
                    seepage_data.append({
                        'Location': key.replace('_', ' ').title(),
                        'Flow Rate': f"{value:.4e} m³/s/m"
                    })
            seepage_df = pd.DataFrame(seepage_data)
            st.dataframe(seepage_df, hide_index=True, use_container_width=True)
            
            st.write("**Continuity Check**")
            st.info(f"Mass balance error: {results['seepage']['mass_balance_error']:.2f}%")
        
        with col2:
            st.write("**Exit Gradient Analysis**")
            gradient_data = []
            for key, value in results['gradients'].items():
                if not any(x in key for x in ['critical', 'safety']):
                    gradient_data.append({
                        'Location': key.replace('_', ' ').title(),
                        'Gradient': f"{value:.4f}"
                    })
            gradient_df = pd.DataFrame(gradient_data)
            st.dataframe(gradient_df, hide_index=True, use_container_width=True)
            
            st.write("**Safety Assessment**")
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Critical Gradient", 
                         f"{results['gradients']['critical_gradient']:.3f}")
            with col2b:
                st.metric("Safety Factor", 
                         f"{results['gradients']['safety_factor']:.2f}")


def handle_exports(params):
    """Handle export functionality"""
    
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
            os.remove(filename)
            st.success("CSV export ready for download!")
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
    
    if params['export_plots'] and st.session_state.solver is not None:
        st.info("Use right-click → 'Save image as...' on any plot to save it")


def main():
    """Main application function"""
    
    # Page header
    st.title("💧 Flow Net Analysis Tool")
    st.markdown("**Multi-layered soil seepage analysis with accurate flow net generation**")
    
    # Information box
    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        This application performs seepage analysis for excavations with sheet piles in layered soils.
        
        **Key Features:**
        - Handles multiple soil layers with different permeabilities
        - Generates orthogonal flow nets (equipotentials ⊥ flow lines)
        - Calculates safety factors against piping failure
        - Maximum analysis depth: 20 meters
        
        **Theory References:**
        - [Flow Net Construction](https://www.geoengineer.org/education/online-lecture-notes-on-soil-mechanics/33-graphical-generation-of-flow-nets)
        - [Seepage Analysis](https://elementaryengineeringlibrary.com/civil-engineering/soil-mechanics/flow-net/)
        """)
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar with parameters
    params = create_sidebar()
    
    # Run analysis if requested
    if params['run_analysis']:
        with st.spinner("Running seepage analysis..."):
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
        <div style='text-align: center; color: gray; font-size: 0.9em;'>
        Flow Net Analysis Tool v2.0 | Streamlit + Python<br>
        For educational and engineering analysis | Max depth: 20m
        </div>
        """, 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
