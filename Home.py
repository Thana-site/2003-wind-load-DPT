"""
Structural Section Analyzer - Main Application
A Streamlit web app for creating and analyzing structural sections
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import sys
import traceback

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Section Properties Analyzer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import custom modules with error handling
try:
    from modules.section_factory import SectionFactory
    from modules.database_manager import DatabaseManager
    from modules.calculations import SectionAnalyzer
    from modules.ui_components import UIComponents
except ImportError as e:
    st.error(f"""
    ### ❌ Module Import Error
    
    Could not import required modules. Please ensure:
    1. All files are in a `modules/` directory
    2. The `modules/` directory contains `__init__.py`
    3. All required packages are installed: `pip install -r requirements.txt`
    
    **Error Details:**
    ```
    {str(e)}
    ```
    
    **Expected Directory Structure:**
    ```
    project/
    ├── Home.py
    ├── requirements.txt
    └── modules/
        ├── __init__.py
        ├── section_factory.py
        ├── database_manager.py
        ├── calculations.py
        └── ui_components.py
    ```
    """)
    st.stop()

# Initialize session state with safe defaults
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'section_history': [],
        'current_section': None,
        'current_properties': {},
        'section_name': '',
        'polygon_nodes': [(0, 0), (100, 0), (100, 100), (0, 100)],
        'node_count': 4,
        'db_manager': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize database manager if not exists
    if st.session_state.db_manager is None:
        try:
            st.session_state.db_manager = DatabaseManager('data/sections.db')
        except Exception as e:
            st.error(f"Database initialization error: {e}")
            st.session_state.db_manager = None

# Initialize session state
init_session_state()

# Initialize components
try:
    ui = UIComponents()
    factory = SectionFactory()
except Exception as e:
    st.error(f"Component initialization error: {e}")
    st.stop()

# Title and description
st.title("🏗️ Structural Section Properties Analyzer")
st.markdown("Create and analyze structural sections with real-time property calculations")

# Create layout
col1, col2 = st.columns([1, 2])

# Sidebar for inputs
with st.sidebar:
    st.header("📐 Section Configuration")
    
    # Section type selection
    section_type = st.selectbox(
        "Select Section Type",
        ["I-Beam", "Box Section", "Channel", "Circular", "Circular Hollow", 
         "T-Section", "Angle", "Custom Polygon"],
        help="Choose the type of structural section to analyze"
    )
    
    # Dynamic input based on section type
    st.subheader("📏 Geometric Parameters")
    
    try:
        if section_type == "I-Beam":
            params = ui.get_ibeam_inputs()
            
        elif section_type == "Box Section":
            params = ui.get_box_inputs()
            
        elif section_type == "Channel":
            params = ui.get_channel_inputs()
            
        elif section_type == "Circular":
            params = ui.get_circular_inputs()
            
        elif section_type == "Circular Hollow":
            params = ui.get_circular_hollow_inputs()
            
        elif section_type == "T-Section":
            params = ui.get_tsection_inputs()
            
        elif section_type == "Angle":
            params = ui.get_angle_inputs()
            
        else:  # Custom Polygon
            params = ui.get_polygon_inputs()
    except Exception as e:
        st.error(f"Error getting inputs: {e}")
        params = {}
    
    # Action buttons
    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            if not params:
                st.error("❌ Invalid parameters")
            else:
                with st.spinner("Creating section..."):
                    try:
                        # Validate parameters first
                        is_valid, error_msg = factory.validate_parameters(section_type, params)
                        
                        if not is_valid:
                            st.error(f"❌ Validation Error: {error_msg}")
                        else:
                            # Create section
                            section = factory.create_section(section_type, params)
                            st.session_state.current_section = section
                            
                            # Analyze section
                            analyzer = SectionAnalyzer(section)
                            properties = analyzer.calculate_properties()
                            
                            # Store in session state
                            st.session_state.current_properties = properties
                            st.session_state.section_name = f"{section_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            
                            st.success("✅ Analysis complete!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        with st.expander("🐛 Debug Information"):
                            st.code(traceback.format_exc())
    
    with col_btn2:
        if st.button("💾 Save", use_container_width=True):
            if st.session_state.current_section is None:
                st.warning("⚠️ No section to save. Analyze first!")
            elif st.session_state.db_manager is None:
                st.error("❌ Database not initialized")
            else:
                try:
                    # Save to database
                    section_data = {
                        'name': st.session_state.section_name,
                        'type': section_type,
                        'parameters': json.dumps(params),
                        'properties': json.dumps(st.session_state.current_properties)
                    }
                    
                    st.session_state.db_manager.save_section(section_data)
                    st.success("✅ Section saved!")
                except Exception as e:
                    st.error(f"❌ Save error: {e}")

# Main content area
with col1:
    st.header("📊 Section Visualization")
    
    if st.session_state.current_section is not None:
        try:
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot geometry
            ax1.set_title("Section Geometry")
            st.session_state.current_section.plot_mesh(ax=ax1, materials=False)
            ax1.set_aspect('equal')
            ax1.grid(True, alpha=0.3)
            
            # Plot centroids
            ax2.set_title("Centroid & Principal Axes")
            st.session_state.current_section.plot_centroids(ax=ax2)
            ax2.set_aspect('equal')
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Visualization error: {e}")
            with st.expander("Debug"):
                st.code(traceback.format_exc())
    else:
        st.info("👈 Configure section parameters and click 'Analyze' to visualize")

with col2:
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Properties", "📁 Database", "📥 Export", "ℹ️ Help"])
    
    with tab1:
        st.header("Section Properties")
        
        if st.session_state.current_section is not None and st.session_state.current_properties:
            try:
                props = st.session_state.current_properties
                
                # Display properties in organized columns
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Area (A)", f"{props['area']:.2f} mm²")
                    st.metric("Perimeter", f"{props['perimeter']:.2f} mm")
                    
                with col_b:
                    st.metric("Ixx", f"{props['ixx_c']:.2e} mm⁴")
                    st.metric("Iyy", f"{props['iyy_c']:.2e} mm⁴")
                    
                with col_c:
                    st.metric("Zxx", f"{props['zxx_plus']:.2e} mm³")
                    st.metric("Zyy", f"{props['zyy_plus']:.2e} mm³")
                
                # Detailed properties table
                st.subheader("Detailed Properties")
                
                # Create comprehensive properties DataFrame
                detailed_props = pd.DataFrame([
                    ["Cross-sectional Area", props['area'], "mm²"],
                    ["Perimeter", props['perimeter'], "mm"],
                    ["Centroid X", props['cx'], "mm"],
                    ["Centroid Y", props['cy'], "mm"],
                    ["Moment of Inertia Ixx", props['ixx_c'], "mm⁴"],
                    ["Moment of Inertia Iyy", props['iyy_c'], "mm⁴"],
                    ["Product of Inertia Ixy", props['ixy_c'], "mm⁴"],
                    ["Radius of Gyration rx", props['rx'], "mm"],
                    ["Radius of Gyration ry", props['ry'], "mm"],
                    ["Elastic Section Modulus Zxx+", props['zxx_plus'], "mm³"],
                    ["Elastic Section Modulus Zxx-", props['zxx_minus'], "mm³"],
                    ["Elastic Section Modulus Zyy+", props['zyy_plus'], "mm³"],
                    ["Elastic Section Modulus Zyy-", props['zyy_minus'], "mm³"],
                    ["Torsion Constant J", props['j'], "mm⁴"],
                    ["Warping Constant Iw", props['gamma'], "mm⁶"],
                ], columns=["Property", "Value", "Unit"])
                
                # Format the values column
                detailed_props['Value'] = detailed_props['Value'].apply(
                    lambda x: f"{x:.4e}" if abs(x) > 1e6 or (abs(x) < 1e-2 and x != 0) else f"{x:.4f}"
                )
                
                st.dataframe(detailed_props, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error displaying properties: {e}")
        else:
            st.info("No analysis results yet. Configure and analyze a section first.")
    
    with tab2:
        st.header("📁 Saved Sections Database")
        
        if st.session_state.db_manager is None:
            st.error("Database not available")
        else:
            try:
                # Load saved sections
                saved_sections = st.session_state.db_manager.get_all_sections()
                
                if saved_sections:
                    # Display saved sections
                    for idx, section in enumerate(saved_sections):
                        with st.expander(f"📐 {section['name']} ({section['type']})"):
                            col_info, col_action = st.columns([3, 1])
                            
                            with col_info:
                                st.write(f"**Created:** {section['created_at']}")
                                if section['properties']:
                                    props = json.loads(section['properties'])
                                    st.write(f"**Area:** {props.get('area', 'N/A'):.2f} mm²")
                                    st.write(f"**Ixx:** {props.get('ixx_c', 'N/A'):.2e} mm⁴")
                            
                            with col_action:
                                if st.button("Load", key=f"load_{idx}"):
                                    try:
                                        # Load section parameters
                                        params = json.loads(section['parameters'])
                                        loaded_section = factory.create_section(section['type'], params)
                                        st.session_state.current_section = loaded_section
                                        
                                        # Recalculate properties
                                        analyzer = SectionAnalyzer(loaded_section)
                                        st.session_state.current_properties = analyzer.calculate_properties()
                                        st.session_state.section_name = section['name']
                                        st.success(f"✅ Loaded {section['name']}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Load error: {e}")
                                
                                if st.button("Delete", key=f"del_{idx}"):
                                    try:
                                        st.session_state.db_manager.delete_section(section['id'])
                                        st.success("🗑️ Section deleted")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Delete error: {e}")
                else:
                    st.info("No saved sections yet. Create and save your first section!")
            except Exception as e:
                st.error(f"Database error: {e}")
    
    with tab3:
        st.header("📥 Export Results")
        
        if st.session_state.current_section is not None and st.session_state.current_properties:
            st.write("Export current section analysis results:")
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                # CSV Export
                try:
                    df = pd.DataFrame([st.session_state.current_properties])
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv,
                        file_name=f"{st.session_state.section_name}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"CSV export error: {e}")
            
            with col_exp2:
                # JSON Export
                try:
                    json_data = json.dumps({
                        'section_name': st.session_state.section_name,
                        'section_type': section_type,
                        'parameters': params,
                        'properties': st.session_state.current_properties,
                        'timestamp': datetime.now().isoformat()
                    }, indent=2)
                    
                    st.download_button(
                        label="📋 Download JSON",
                        data=json_data,
                        file_name=f"{st.session_state.section_name}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"JSON export error: {e}")
            
            # Batch export
            st.subheader("Batch Export All Saved Sections")
            if st.session_state.db_manager:
                try:
                    all_sections = st.session_state.db_manager.export_all_to_dataframe()
                    if not all_sections.empty:
                        csv_all = all_sections.to_csv(index=False)
                        st.download_button(
                            label="📦 Download All Sections CSV",
                            data=csv_all,
                            file_name=f"all_sections_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No sections to export")
                except Exception as e:
                    st.error(f"Batch export error: {e}")
        else:
            st.info("No results to export. Analyze a section first!")
    
    with tab4:
        st.header("ℹ️ Help & Documentation")
        
        st.markdown("""
        ### How to Use This Application
        
        1. **Select Section Type**: Choose from predefined shapes or create a custom polygon
        2. **Input Parameters**: Enter the geometric dimensions for your section
        3. **Analyze**: Click the Analyze button to calculate section properties
        4. **Save**: Store your section in the database for future use
        5. **Export**: Download results as CSV or JSON files
        
        ### Section Types Available:
        - **I-Beam**: Standard I-shaped sections
        - **Box Section**: Hollow rectangular sections
        - **Channel**: C-shaped sections
        - **Circular**: Solid circular sections
        - **Circular Hollow**: Pipe sections
        - **T-Section**: T-shaped sections
        - **Angle**: L-shaped sections
        - **Custom Polygon**: Define your own shape with nodes
        
        ### Calculated Properties:
        - Cross-sectional area (A)
        - Moments of inertia (Ixx, Iyy, Ixy)
        - Section moduli (Zxx, Zyy)
        - Radii of gyration (rx, ry)
        - Torsion constant (J)
        - Warping constant (Iw)
        - Centroid location (cx, cy)
        
        ### Tips:
        - Use consistent units (mm recommended)
        - Save frequently used sections for quick access
        - Export results for documentation
        - Check validation messages if analysis fails
        """)

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit & sectionproperties | © 2024 Section Analyzer*")
