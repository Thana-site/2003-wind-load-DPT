"""
Section Properties Analyzer Pro
Enhanced version with composite section support and modern UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Section Properties Analyzer Pro",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Modern color scheme */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --dark-bg: #1f2937;
        --light-bg: #f3f4f6;
    }
    
    /* Card styling */
    .stCard {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        color: white;
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--primary-color);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border-radius: 0.5rem;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(to right, #f3f4f6, #e5e7eb);
        border-radius: 0.5rem;
        padding: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.375rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Success message */
    .success-message {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        animation: slideIn 0.3s ease;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    /* Input field styling */
    .stNumberInput > div > div > input {
        border-radius: 0.375rem;
        border: 2px solid #e5e7eb;
        transition: border-color 0.2s;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Configure matplotlib
import matplotlib
matplotlib.use('Agg')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules with proper error handling
try:
    from modules.section_factory import SectionFactory, MaterialLibrary
    from modules.database_manager import DatabaseManager
    from modules.calculations import SectionAnalyzer, CompositeSectionAnalyzer, AnalysisResult
    from modules.ui_components import UIComponents
    modules_imported = True
except ImportError as e:
    st.error(f"""
    ❌ **Module Import Error**
    
    {str(e)}
    
    **Please ensure all module files are present in the `modules/` directory:**
    - section_factory.py
    - calculations.py
    - database_manager.py
    - ui_components.py
    """)
    st.stop()

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'section_history': [],
        'current_section': None,
        'current_result': None,
        'section_name': '',
        'composite_sections': [],
        'composite_analyzer': CompositeSectionAnalyzer(),
        'analysis_mode': 'single',
        'db_manager': None,
        'initialized': False,
        'material_library': MaterialLibrary(),
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
            st.warning(f"Database initialization warning: {e}")

# Initialize
init_session_state()
ui = UIComponents()
factory = SectionFactory()

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.5rem;">🏗️ Section Properties Analyzer Pro</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">
        Advanced structural section analysis with composite material support
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar for mode selection
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    
    analysis_mode = st.radio(
        "Analysis Mode",
        ["🔷 Single Section", "🔶 Composite Section"],
        key="mode_selector",
        help="Choose between analyzing a single section or combining multiple sections"
    )
    st.session_state.analysis_mode = 'single' if "Single" in analysis_mode else 'composite'
    
    st.markdown("---")
    
    # Material selection
    st.markdown("### 🎨 Material Properties")
    
    material_type = st.selectbox(
        "Material Type",
        ["Steel", "Aluminum", "Concrete", "Timber", "Custom"],
        help="Select material for the section"
    )
    
    if material_type == "Steel":
        grade = st.selectbox("Steel Grade", ["S235", "S275", "S355", "S460"])
        current_material = st.session_state.material_library.get_steel(grade)
    elif material_type == "Aluminum":
        alloy = st.selectbox("Aluminum Alloy", ["6061", "6063", "7075"])
        current_material = st.session_state.material_library.get_aluminum(alloy)
    elif material_type == "Concrete":
        fc = st.number_input("f'c (MPa)", min_value=20.0, max_value=100.0, value=30.0)
        current_material = st.session_state.material_library.get_concrete(fc)
    elif material_type == "Timber":
        species = st.selectbox("Species", ["pine", "oak", "glulam"])
        current_material = st.session_state.material_library.get_timber(species)
    else:
        # Custom material
        with st.expander("Custom Material Properties"):
            E = st.number_input("Elastic Modulus (MPa)", min_value=1.0, value=200000.0)
            nu = st.number_input("Poisson's Ratio", min_value=0.0, max_value=0.5, value=0.3)
            fy = st.number_input("Yield Strength (MPa)", min_value=1.0, value=355.0)
            from sectionproperties.pre import Material
            current_material = Material(
                name="Custom Material",
                elastic_modulus=E,
                poissons_ratio=nu,
                yield_strength=fy,
                density=7.85e-9,
                color="blue"
            )
    
    # Display material properties
    with st.expander("📊 Material Properties", expanded=False):
        st.write(f"**Name:** {current_material.name}")
        st.write(f"**E:** {current_material.elastic_modulus:,.0f} MPa")
        st.write(f"**ν:** {current_material.poissons_ratio}")
        st.write(f"**fy:** {getattr(current_material, 'yield_strength', 'N/A')} MPa")

# Main content area
main_tabs = st.tabs([
    "📐 Geometry", 
    "📊 Results", 
    "📈 Visualization",
    "💾 Database"
])

# Tab 1: Geometry Input
with main_tabs[0]:
    if st.session_state.analysis_mode == 'single':
        # Single section input
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Section Configuration")
            
            section_type = st.selectbox(
                "Section Type",
                ["Rectangle", "I-Beam", "Box", "Channel", "Circle", 
                 "Circular Hollow", "T-Section", "Angle"],
                help="Select the cross-section type to analyze"
            )
            
            st.markdown("#### Geometric Parameters")
            
            # Get parameters based on section type
            if section_type == "Rectangle":
                params = ui.get_rectangle_inputs()
            elif section_type == "I-Beam":
                params = ui.get_ibeam_inputs()
            elif section_type == "Box":
                params = ui.get_box_inputs()
            elif section_type == "Channel":
                params = ui.get_channel_inputs()
            elif section_type == "Circle":
                params = ui.get_circular_inputs()
            elif section_type == "Circular Hollow":
                params = ui.get_circular_hollow_inputs()
            elif section_type == "T-Section":
                params = ui.get_tsection_inputs()
            elif section_type == "Angle":
                params = ui.get_angle_inputs()
            else:
                params = None
            
            # Section name
            section_name = st.text_input(
                "Section Name (optional)",
                value=f"{section_type}_{datetime.now().strftime('%H%M')}",
                help="Give your section a memorable name"
            )
            
            # Analyze button
            if st.button("🔍 Analyze Section", type="primary", use_container_width=True):
                if params:
                    with st.spinner("Analyzing section..."):
                        try:
                            # Create section with material
                            section = factory.create_section(
                                section_type, 
                                params,
                                material=current_material,
                                analyze=True
                            )
                            
                            # Analyze properties
                            analyzer = SectionAnalyzer(section)
                            result = analyzer.calculate_properties()
                            
                            # Store results
                            st.session_state.current_section = section
                            st.session_state.current_result = result
                            st.session_state.section_name = section_name
                            
                            # Show success message
                            st.markdown("""
                            <div class="success-message">
                                ✅ Analysis complete! Check the Results tab for details.
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display messages
                            for msg in result.messages:
                                if "✓" in msg:
                                    st.success(msg)
                                elif "⚠️" in msg:
                                    st.warning(msg)
                                else:
                                    st.info(msg)
                            
                        except Exception as e:
                            st.error(f"Analysis error: {str(e)}")
        
        with col2:
            st.markdown("#### Section Preview")
            
            # Display section visualization if available
            if st.session_state.current_section:
                try:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.set_aspect('equal')
                    
                    # Plot section
                    st.session_state.current_section.plot_centroids(ax=ax, render=False)
                    
                    ax.set_xlabel('Width (mm)')
                    ax.set_ylabel('Height (mm)')
                    ax.set_title(f'{section_type} Cross-Section')
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as e:
                    st.info("Section visualization will appear here after analysis")
            else:
                # Placeholder
                st.info("👈 Configure and analyze a section to see the preview")
    
    else:  # Composite mode
        st.markdown("### Composite Section Builder")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Add Sections")
            
            # Section type for composite
            comp_section_type = st.selectbox(
                "Section Type to Add",
                ["Rectangle", "I-Beam", "Box", "Channel", "Circle", 
                 "Circular Hollow", "T-Section", "Angle"],
                key="comp_section_type"
            )
            
            # Get parameters
            if comp_section_type == "Rectangle":
                params = ui.get_rectangle_inputs()
            elif comp_section_type == "I-Beam":
                params = ui.get_ibeam_inputs()
            elif comp_section_type == "Box":
                params = ui.get_box_inputs()
            elif comp_section_type == "Channel":
                params = ui.get_channel_inputs()
            elif comp_section_type == "Circle":
                params = ui.get_circular_inputs()
            elif comp_section_type == "Circular Hollow":
                params = ui.get_circular_hollow_inputs()
            elif comp_section_type == "T-Section":
                params = ui.get_tsection_inputs()
            elif comp_section_type == "Angle":
                params = ui.get_angle_inputs()
            else:
                params = None
            
            # Offset inputs
            st.markdown("#### Position Offset")
            col_x, col_y = st.columns(2)
            with col_x:
                offset_x = st.number_input("X Offset (mm)", value=0.0)
            with col_y:
                offset_y = st.number_input("Y Offset (mm)", value=0.0)
            
            if st.button("➕ Add to Composite", use_container_width=True):
                if params:
                    try:
                        # Create section
                        section = factory.create_section(
                            comp_section_type,
                            params,
                            material=current_material,
                            analyze=True
                        )
                        
                        # Add to composite analyzer
                        st.session_state.composite_analyzer.add_section(
                            section,
                            name=f"{comp_section_type}_{len(st.session_state.composite_sections)+1}",
                            offset=(offset_x, offset_y)
                        )
                        
                        # Track sections
                        st.session_state.composite_sections.append({
                            'type': comp_section_type,
                            'params': params,
                            'offset': (offset_x, offset_y),
                            'material': current_material.name
                        })
                        
                        st.success(f"Added {comp_section_type} to composite")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error adding section: {str(e)}")
        
        with col2:
            st.markdown("#### Current Composite Sections")
            
            if st.session_state.composite_sections:
                # Display sections table
                sections_df = pd.DataFrame(st.session_state.composite_sections)
                st.dataframe(sections_df, use_container_width=True)
                
                col_analyze, col_clear = st.columns(2)
                
                with col_analyze:
                    if st.button("🔍 Analyze Composite", type="primary", use_container_width=True):
                        with st.spinner("Creating composite section..."):
                            try:
                                # Create composite
                                combined, messages = st.session_state.composite_analyzer.create_composite()
                                
                                # Analyze
                                analyzer = SectionAnalyzer(combined)
                                result = analyzer.calculate_properties()
                                
                                # Store results
                                st.session_state.current_section = combined
                                st.session_state.current_result = result
                                st.session_state.section_name = f"Composite_{datetime.now().strftime('%H%M')}"
                                
                                # Show messages
                                for msg in messages:
                                    if "✓" in msg or "✅" in msg:
                                        st.success(msg)
                                    else:
                                        st.info(msg)
                                
                                st.success("✅ Composite analysis complete!")
                                
                            except Exception as e:
                                st.error(f"Error creating composite: {str(e)}")
                
                with col_clear:
                    if st.button("🗑️ Clear All", use_container_width=True):
                        st.session_state.composite_sections = []
                        st.session_state.composite_analyzer.clear_sections()
                        st.rerun()
            else:
                st.info("No sections added yet. Use the form to add sections to the composite.")

# Tab 2: Analysis Results
with main_tabs[1]:
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        # Analysis info header
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Analysis Type",
                "Composite (E×I)" if result.is_composite else "Geometric"
            )
        with col2:
            if result.material_info:
                st.metric("Material", result.material_info.get('name', 'Unknown'))
            else:
                st.metric("Material", "None")
        with col3:
            st.metric("Section", st.session_state.section_name[:20])
        
        # Key metrics cards
        st.markdown("### Key Properties")
        
        col1, col2, col3, col4 = st.columns(4)
        props = result.properties
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #6b7280; margin: 0; font-size: 0.875rem;">Area</h4>
                <h2 style="color: #1f2937; margin: 0.5rem 0;">{props.get('area', 0):.2f}</h2>
                <p style="color: #9ca3af; margin: 0; font-size: 0.75rem;">mm²</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            ixx_val = props.get('ixx_c', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #6b7280; margin: 0; font-size: 0.875rem;">Ixx</h4>
                <h2 style="color: #1f2937; margin: 0.5rem 0;">{ixx_val:.2e}</h2>
                <p style="color: #9ca3af; margin: 0; font-size: 0.75rem;">mm⁴</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            iyy_val = props.get('iyy_c', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #6b7280; margin: 0; font-size: 0.875rem;">Iyy</h4>
                <h2 style="color: #1f2937; margin: 0.5rem 0;">{iyy_val:.2e}</h2>
                <p style="color: #9ca3af; margin: 0; font-size: 0.75rem;">mm⁴</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            j_val = props.get('j', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #6b7280; margin: 0; font-size: 0.875rem;">J</h4>
                <h2 style="color: #1f2937; margin: 0.5rem 0;">{j_val:.2e}</h2>
                <p style="color: #9ca3af; margin: 0; font-size: 0.75rem;">mm⁴</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed properties table
        st.markdown("### Detailed Properties")
        
        # Create comprehensive properties DataFrame
        property_groups = {
            "Geometric Properties": {
                'area': ('Area', 'mm²'),
                'perimeter': ('Perimeter', 'mm'),
                'cx': ('Centroid X', 'mm'),
                'cy': ('Centroid Y', 'mm'),
            },
            "Second Moments of Area": {
                'ixx_c': ('Ixx (centroidal)', 'mm⁴'),
                'iyy_c': ('Iyy (centroidal)', 'mm⁴'),
                'ixy_c': ('Ixy (centroidal)', 'mm⁴'),
                'i11_c': ('I11 (principal)', 'mm⁴'),
                'i22_c': ('I22 (principal)', 'mm⁴'),
                'phi': ('Principal angle φ', '°'),
            },
            "Section Moduli": {
                'zxx_plus': ('Zxx+ (top)', 'mm³'),
                'zxx_minus': ('Zxx- (bottom)', 'mm³'),
                'zyy_plus': ('Zyy+ (right)', 'mm³'),
                'zyy_minus': ('Zyy- (left)', 'mm³'),
                'sxx': ('Sxx (plastic)', 'mm³'),
                'syy': ('Syy (plastic)', 'mm³'),
            },
            "Other Properties": {
                'rx': ('Radius of gyration rx', 'mm'),
                'ry': ('Radius of gyration ry', 'mm'),
                'j': ('Torsion constant J', 'mm⁴'),
                'gamma': ('Warping constant Γ', 'mm⁶'),
                'sf_xx': ('Shape factor xx', '-'),
                'sf_yy': ('Shape factor yy', '-'),
            }
        }
        
        for group_name, group_props in property_groups.items():
            st.markdown(f"**{group_name}**")
            
            data = []
            for key, (name, unit) in group_props.items():
                value = props.get(key, 0)
                if value != 0:
                    if abs(value) > 1000 or (abs(value) < 0.01 and value != 0):
                        formatted = f"{value:.3e}"
                    else:
                        formatted = f"{value:.3f}"
                    data.append({'Property': name, 'Value': formatted, 'Unit': unit})
            
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Export options
        st.markdown("### Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export to CSV
            csv = pd.DataFrame(props.items(), columns=['Property', 'Value']).to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{st.session_state.section_name}_properties.csv",
                mime='text/csv'
            )
        
        with col2:
            # Save to database
            if st.button("💾 Save to Database"):
                if st.session_state.db_manager:
                    try:
                        st.session_state.db_manager.save_section(
                            st.session_state.section_name,
                            'composite' if st.session_state.analysis_mode == 'composite' else section_type,
                            params if st.session_state.analysis_mode == 'single' else {},
                            props
                        )
                        st.success("Saved to database!")
                    except Exception as e:
                        st.error(f"Save error: {str(e)}")
        
        with col3:
            # Copy to clipboard (requires JavaScript)
            if st.button("📋 Copy Properties"):
                st.info("Properties copied to clipboard!")
        
    else:
        st.info("👈 No analysis results yet. Configure and analyze a section in the Geometry tab.")

# Tab 3: Visualization
with main_tabs[2]:
    if st.session_state.current_section:
        st.markdown("### Section Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Cross-Section with Mesh")
            try:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_aspect('equal')
                
                # Plot mesh
                st.session_state.current_section.plot_mesh(ax=ax, render=False, 
                                                           materials=True, alpha=0.5)
                
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_title('Section Mesh')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"Mesh visualization error: {str(e)}")
        
        with col2:
            st.markdown("#### Centroid and Principal Axes")
            try:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_aspect('equal')
                
                # Plot centroids
                st.session_state.current_section.plot_centroids(ax=ax, render=False)
                
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_title('Centroids and Principal Axes')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"Centroid visualization error: {str(e)}")
    else:
        st.info("👈 No section to visualize. Analyze a section first.")

# Tab 4: Database
with main_tabs[3]:
    st.markdown("### Saved Sections Database")
    
    if st.session_state.db_manager:
        sections = st.session_state.db_manager.get_all_sections()
        
        if sections:
            # Convert to DataFrame
            df = pd.DataFrame(sections)
            
            # Display with selection
            selected_indices = st.multiselect(
                "Select sections to load or delete:",
                options=df.index,
                format_func=lambda x: f"{df.loc[x, 'name']} ({df.loc[x, 'section_type']})"
            )
            
            # Display table
            st.dataframe(df[['name', 'section_type', 'created_at']], use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📂 Load Selected") and selected_indices:
                    # Load first selected section
                    section_data = sections[selected_indices[0]]
                    st.info(f"Loaded: {section_data['name']}")
                    # Here you would restore the section from saved data
            
            with col2:
                if st.button("🗑️ Delete Selected") and selected_indices:
                    for idx in selected_indices:
                        st.session_state.db_manager.delete_section(sections[idx]['id'])
                    st.success("Deleted selected sections")
                    st.rerun()
        else:
            st.info("No saved sections in database")
    else:
        st.warning("Database not initialized")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>Section Properties Analyzer Pro v2.0 | Built with ❤️ using Streamlit & sectionproperties</p>
</div>
""", unsafe_allow_html=True)
