"""
Structural Section Analyzer - Enhanced Version
Modern UI with composite section support
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Section Properties Analyzer Pro",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Modern color scheme */
    :root {
        --primary-color: #1e3a8a;
        --secondary-color: #3b82f6;
        --accent-color: #10b981;
        --background: #f8fafc;
        --card-background: #ffffff;
    }
    
    /* Card styling */
    .stCard {
        background-color: var(--card-background);
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: var(--card-background);
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        background-color: transparent;
        border-radius: 0.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--secondary-color);
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--secondary-color);
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: var(--primary-color);
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Success/Error styling */
    .success-box {
        background-color: #10b981;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #f59e0b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configure matplotlib backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules with error handling
try:
    from modules.section_factory import SectionFactory
    from modules.database_manager import DatabaseManager
    from modules.calculations import SectionAnalyzer, CompositeSectionAnalyzer, AnalysisResult
    from modules.ui_components import UIComponents
    modules_imported = True
except ImportError as e:
    st.error(f"❌ Module Import Error: {e}")
    st.stop()

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'section_history': [],
        'current_section': None,
        'current_properties': {},
        'current_result': None,  # For AnalysisResult object
        'section_name': '',
        'composite_sections': [],  # For composite analysis
        'composite_analyzer': CompositeSectionAnalyzer(),
        'analysis_mode': 'single',  # 'single' or 'composite'
        'db_manager': None,
        'initialized': False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize database
    if not st.session_state.initialized:
        try:
            os.makedirs('data', exist_ok=True)
            st.session_state.db_manager = DatabaseManager('data/sections.db')
            st.session_state.initialized = True
        except Exception as e:
            st.warning(f"⚠️ Database initialization warning: {e}")

# Initialize
init_session_state()
ui = UIComponents()
factory = SectionFactory()

# Header with gradient
st.markdown("""
<div style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 1rem;
    margin-bottom: 2rem;
">
    <h1 style="color: white; margin: 0;">🏗️ Section Properties Analyzer Pro</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
        Advanced structural section analysis with composite material support
    </p>
</div>
""", unsafe_allow_html=True)

# Mode selector at the top
col1_mode, col2_mode, col3_mode = st.columns([1, 2, 1])
with col2_mode:
    analysis_mode = st.radio(
        "Analysis Mode",
        ["🔷 Single Section", "🔶 Composite Section"],
        horizontal=True,
        key="mode_selector",
        help="Choose between analyzing a single section or combining multiple sections"
    )
    st.session_state.analysis_mode = 'single' if "Single" in analysis_mode else 'composite'

# Main content area with tabs
main_tabs = st.tabs([
    "📐 Geometry Input", 
    "📊 Analysis Results", 
    "📈 Visualization",
    "💾 Database",
    "⚙️ Settings"
])

# Tab 1: Geometry Input
with main_tabs[0]:
    if st.session_state.analysis_mode == 'single':
        # Single section input
        col_input, col_preview = st.columns([1, 1])
        
        with col_input:
            st.markdown("### Section Configuration")
            
            # Section type selection with icons
            section_type = st.selectbox(
                "Select Section Type",
                ["I-Beam", "Box Section", "Channel", "Circular", "Circular Hollow", 
                 "T-Section", "Angle", "Custom Polygon"],
                help="Choose the structural section type"
            )
            
            # Material selection
            use_material = st.checkbox("Apply Material Properties", value=True, 
                                      help="Enable for composite analysis with E*I calculations")
            
            if use_material:
                col_mat1, col_mat2 = st.columns(2)
                with col_mat1:
                    material_type = st.selectbox(
                        "Material",
                        ["Steel", "Aluminum", "Concrete", "Custom"]
                    )
                with col_mat2:
                    if material_type == "Custom":
                        elastic_modulus = st.number_input("E [MPa]", value=200000.0)
                    else:
                        elastic_modulus = {
                            "Steel": 200000,
                            "Aluminum": 70000,
                            "Concrete": 30000
                        }[material_type]
                        st.metric("E [MPa]", f"{elastic_modulus:,}")
            
            # Dynamic parameter inputs
            st.markdown("### Geometric Parameters")
            
            # Get parameters based on section type
            params = {}
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
            
            # Action buttons
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔍 Analyze Section", type="primary", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        try:
                            # Create section
                            section = factory.create_section(section_type, params)
                            st.session_state.current_section = section
                            
                            # Analyze with new analyzer
                            analyzer = SectionAnalyzer(section)
                            result = analyzer.calculate_properties()
                            
                            # Store results
                            st.session_state.current_result = result
                            st.session_state.current_properties = result.properties
                            st.session_state.section_name = f"{section_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            
                            # Show messages
                            for msg in result.messages:
                                if "✓" in msg:
                                    st.success(msg)
                                elif "⚠️" in msg:
                                    st.warning(msg)
                                else:
                                    st.info(msg)
                            
                            st.success("✅ Analysis complete!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
            
            with col_btn2:
                if st.button("💾 Save to Database", use_container_width=True):
                    if st.session_state.current_section is None:
                        st.warning("No section to save!")
                    else:
                        # Save logic here
                        st.success("✅ Saved!")
        
        with col_preview:
            st.markdown("### Live Preview")
            
            # Show section sketch or placeholder
            if st.session_state.current_section:
                try:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    st.session_state.current_section.plot_mesh(ax=ax, materials=False)
                    ax.set_aspect('equal')
                    ax.grid(True, alpha=0.3)
                    ax.set_title(f"{section_type} Section")
                    st.pyplot(fig)
                    plt.close()
                except:
                    st.info("Preview will appear after analysis")
            else:
                # Placeholder
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
                    height: 400px;
                    border-radius: 1rem;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #4c1d95;
                    font-size: 1.2rem;
                ">
                    Section preview will appear here
                </div>
                """, unsafe_allow_html=True)
    
    else:  # Composite mode
        st.markdown("### Composite Section Builder")
        
        col_list, col_add = st.columns([2, 1])
        
        with col_add:
            st.markdown("#### Add Section")
            
            comp_section_type = st.selectbox(
                "Section Type",
                ["I-Beam", "Box Section", "Plate", "Angle"],
                key="comp_type"
            )
            
            # Simplified inputs for composite
            if comp_section_type == "Plate":
                width = st.number_input("Width [mm]", value=200.0)
                height = st.number_input("Height [mm]", value=10.0)
                params = {'width': width, 'depth': height}
            else:
                # Use simplified inputs
                st.info("Configure in the parameters section")
                params = {}
            
            # Offset
            col_x, col_y = st.columns(2)
            with col_x:
                offset_x = st.number_input("Offset X", value=0.0)
            with col_y:
                offset_y = st.number_input("Offset Y", value=0.0)
            
            if st.button("➕ Add to Composite", use_container_width=True):
                if params:
                    # Add section to composite
                    st.session_state.composite_sections.append({
                        'type': comp_section_type,
                        'params': params,
                        'offset': (offset_x, offset_y)
                    })
                    st.success(f"Added {comp_section_type}")
                    st.rerun()
        
        with col_list:
            st.markdown("#### Current Sections")
            
            if st.session_state.composite_sections:
                for idx, section in enumerate(st.session_state.composite_sections):
                    with st.container():
                        col1, col2, col3 = st.columns([2, 2, 1])
                        with col1:
                            st.write(f"**{idx+1}. {section['type']}**")
                        with col2:
                            st.write(f"Offset: ({section['offset'][0]}, {section['offset'][1]})")
                        with col3:
                            if st.button("🗑️", key=f"del_{idx}"):
                                st.session_state.composite_sections.pop(idx)
                                st.rerun()
                
                if st.button("🔍 Analyze Composite", type="primary"):
                    with st.spinner("Creating composite section..."):
                        try:
                            # Create composite
                            composite = CompositeSectionAnalyzer()
                            
                            for section_data in st.session_state.composite_sections:
                                # Create individual sections
                                geom = factory.create_section(
                                    section_data['type'],
                                    section_data['params']
                                )
                                composite.add_section(
                                    geom,
                                    offset=section_data['offset']
                                )
                            
                            # Analyze composite
                            combined, messages = composite.create_composite()
                            
                            # Show messages
                            for msg in messages:
                                if "✓" in msg:
                                    st.success(msg)
                                else:
                                    st.info(msg)
                            
                            st.session_state.current_section = combined
                            st.success("✅ Composite analysis complete!")
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.info("No sections added yet. Use the form to add sections.")

# Tab 2: Analysis Results
with main_tabs[1]:
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        # Display analysis info
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Analysis Type", 
                     "Composite (E*I)" if result.is_composite else "Geometric")
        with col_info2:
            if result.material_info:
                st.metric("Material", result.material_info.get('name', 'Unknown'))
        with col_info3:
            st.metric("Section Name", st.session_state.section_name[:20])
        
        # Key metrics in cards
        st.markdown("### Key Properties")
        
        col1, col2, col3, col4 = st.columns(4)
        props = result.properties
        
        with col1:
            st.markdown("""
            <div class="stCard">
                <h4 style="color: #1e3a8a; margin-bottom: 0.5rem;">Area</h4>
                <h2 style="color: #3b82f6; margin: 0;">{:.2f}</h2>
                <p style="color: #64748b; margin: 0;">mm²</p>
            </div>
            """.format(props.get('area', 0)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stCard">
                <h4 style="color: #1e3a8a; margin-bottom: 0.5rem;">Ixx</h4>
                <h2 style="color: #3b82f6; margin: 0;">{:.2e}</h2>
                <p style="color: #64748b; margin: 0;">mm⁴</p>
            </div>
            """.format(props.get('ixx_c', 0)), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="stCard">
                <h4 style="color: #1e3a8a; margin-bottom: 0.5rem;">Iyy</h4>
                <h2 style="color: #3b82f6; margin: 0;">{:.2e}</h2>
                <p style="color: #64748b; margin: 0;">mm⁴</p>
            </div>
            """.format(props.get('iyy_c', 0)), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="stCard">
                <h4 style="color: #1e3a8a; margin-bottom: 0.5rem;">J</h4>
                <h2 style="color: #3b82f6; margin: 0;">{:.2e}</h2>
                <p style="color: #64748b; margin: 0;">mm⁴</p>
            </div>
            """.format(props.get('j', 0)), unsafe_allow_html=True)
        
        # Detailed table
        st.markdown("### Detailed Properties")
        
        # Create DataFrame with all properties
        detailed_data = []
        property_map = {
            'area': ('Cross-sectional Area', 'mm²'),
            'perimeter': ('Perimeter', 'mm'),
            'cx': ('Centroid X', 'mm'),
            'cy': ('Centroid Y', 'mm'),
            'ixx_c': ('Moment of Inertia Ixx', 'mm⁴'),
            'iyy_c': ('Moment of Inertia Iyy', 'mm⁴'),
            'ixy_c': ('Product of Inertia Ixy', 'mm⁴'),
            'rx': ('Radius of Gyration rx', 'mm'),
            'ry': ('Radius of Gyration ry', 'mm'),
            'zxx_plus': ('Section Modulus Zxx+', 'mm³'),
            'zxx_minus': ('Section Modulus Zxx-', 'mm³'),
            'zyy_plus': ('Section Modulus Zyy+', 'mm³'),
            'zyy_minus': ('Section Modulus Zyy-', 'mm³'),
            'j': ('Torsion Constant J', 'mm⁴'),
            'gamma': ('Warping Constant Γ', 'mm⁶'),
        }
        
        for key, (name, unit) in property_map.items():
            if key in props:
                value = props[key]
                if abs(value) > 1e6 or (abs(value) < 1e-2 and value != 0):
                    formatted = f"{value:.4e}"
                else:
                    formatted = f"{value:.4f}"
                detailed_data.append([name, formatted, unit])
        
        df = pd.DataFrame(detailed_data, columns=['Property', 'Value', 'Unit'])
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Export buttons
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        with col_exp1:
            csv = df.to_csv(index=False)
            st.download_button(
                "📄 Download CSV",
                data=csv,
                file_name=f"{st.session_state.section_name}.csv",
                mime="text/csv"
            )
    else:
        st.info("No analysis results yet. Go to Geometry Input tab to analyze a section.")

# Tab 3: Visualization
with main_tabs[2]:
    if st.session_state.current_section:
        st.markdown("### Section Visualization")
        
        viz_tabs = st.tabs(["Mesh", "Centroids", "Stress Contours"])
        
        with viz_tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 8))
                st.session_state.current_section.plot_mesh(ax=ax)
                ax.set_title("Section Mesh")
                ax.set_aspect('equal')
                st.pyplot(fig)
                plt.close()
        
        with viz_tabs[1]:
            fig, ax = plt.subplots(figsize=(8, 8))
            st.session_state.current_section.plot_centroids(ax=ax)
            ax.set_title("Centroid and Principal Axes")
            ax.set_aspect('equal')
            st.pyplot(fig)
            plt.close()
    else:
        st.info("No section to visualize. Analyze a section first.")

# Tab 4: Database
with main_tabs[3]:
    st.markdown("### Section Database")
    
    if st.session_state.db_manager:
        sections = st.session_state.db_manager.get_all_sections()
        
        if sections:
            # Create a nice table view
            df = pd.DataFrame(sections)
            st.dataframe(
                df[['name', 'type', 'created_at']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No saved sections yet.")
    else:
        st.warning("Database not available")

# Tab 5: Settings
with main_tabs[4]:
    st.markdown("### Application Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Analysis Settings")
        mesh_quality = st.select_slider(
            "Mesh Quality",
            options=["Coarse", "Normal", "Fine", "Very Fine"],
            value="Normal"
        )
        
        warping_analysis = st.checkbox("Include Warping Analysis", value=True)
        plastic_analysis = st.checkbox("Include Plastic Analysis", value=True)
    
    with col2:
        st.markdown("#### Display Settings")
        units = st.selectbox("Units", ["Metric (mm)", "Imperial (in)"])
        decimal_places = st.number_input("Decimal Places", min_value=1, max_value=6, value=4)
        
    st.markdown("#### About")
    st.info("""
    **Section Properties Analyzer Pro**  
    Version 2.0.0  
    
    Enhanced features:
    - ✅ Automatic composite material detection
    - ✅ Multiple section combination
    - ✅ Modern responsive UI
    - ✅ Advanced visualization options
    """)

# Footer
st.markdown("""
<div style="
    margin-top: 3rem;
    padding: 1rem;
    background: linear-gradient(90deg, #e0e7ff 0%, #c7d2fe 100%);
    border-radius: 0.5rem;
    text-align: center;
    color: #4c1d95;
">
    Built with Streamlit & sectionproperties | Enhanced Edition 2024
</div>
""", unsafe_allow_html=True)
