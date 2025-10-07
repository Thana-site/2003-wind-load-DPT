import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sectionproperties.pre.library import rectangular_section, i_section
from sectionproperties.pre import Material, Geometry
from sectionproperties.pre.geometry import CompoundGeometry
from sectionproperties.analysis import Section
from shapely.geometry import Polygon
import pandas as pd

# Page configuration
st.set_page_config(page_title="Steel Section Properties Analyzer", layout="wide")

# Title and description
st.title("🏗️ Steel Built-Up Section Properties Analyzer")
st.markdown("Analyze geometric and mechanical properties of steel built-up sections")

# Sidebar for section type selection
st.sidebar.header("Section Configuration")
input_method = st.sidebar.radio(
    "Input Method",
    ["Simple Shapes", "Node-Based Input", "Multiple Components"]
)

# Material properties
st.sidebar.header("Material Properties")
steel_grade = st.sidebar.selectbox("Steel Grade", ["A36", "A572 Gr50", "A992", "Custom"])

if steel_grade == "Custom":
    E = st.sidebar.number_input("Elastic Modulus (MPa)", value=200000.0)
    fy = st.sidebar.number_input("Yield Strength (MPa)", value=250.0)
else:
    steel_properties = {
        "A36": {"E": 200000, "fy": 250},
        "A572 Gr50": {"E": 200000, "fy": 345},
        "A992": {"E": 200000, "fy": 345}
    }
    E = steel_properties[steel_grade]["E"]
    fy = steel_properties[steel_grade]["fy"]
    st.sidebar.write(f"E = {E} MPa")
    st.sidebar.write(f"fy = {fy} MPa")

# Create material
steel = Material(
    name="Steel",
    elastic_modulus=E,
    poissons_ratio=0.3,
    yield_strength=fy,
    density=7850e-9,
    color="lightgrey"
)

# Initialize session state for storing geometries
if 'geometries' not in st.session_state:
    st.session_state.geometries = []
if 'node_data' not in st.session_state:
    st.session_state.node_data = pd.DataFrame({
        'Node': [1, 2, 3, 4],
        'X (mm)': [0.0, 200.0, 200.0, 0.0],
        'Y (mm)': [0.0, 0.0, 10.0, 10.0]
    })

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Section Definition")
    
    # ============ SIMPLE SHAPES METHOD ============
    if input_method == "Simple Shapes":
        section_type = st.selectbox(
            "Select Section Type",
            ["I-Section", "Box Section", "Channel Section", "T-Section"]
        )
        
        if section_type == "I-Section":
            st.subheader("I-Section Parameters")
            d = st.number_input("Overall Depth (mm)", value=400.0, min_value=0.1)
            bf = st.number_input("Flange Width (mm)", value=200.0, min_value=0.1)
            tf = st.number_input("Flange Thickness (mm)", value=15.0, min_value=0.1)
            tw = st.number_input("Web Thickness (mm)", value=10.0, min_value=0.1)
            
            geom = i_section(d=d, b=bf, t_f=tf, t_w=tw, r=0, n_r=1, material=steel)
            
        elif section_type == "Box Section":
            st.subheader("Box Section Parameters")
            b = st.number_input("Width (mm)", value=300.0, min_value=0.1)
            d = st.number_input("Depth (mm)", value=400.0, min_value=0.1)
            tf = st.number_input("Flange Thickness (mm)", value=12.0, min_value=0.1)
            tw = st.number_input("Web Thickness (mm)", value=10.0, min_value=0.1)
            
            outer = rectangular_section(b=b, d=d, material=steel)
            inner = rectangular_section(b=b-2*tw, d=d-2*tf, material=steel).shift_section(tw, tf)
            geom = outer - inner
            
        elif section_type == "Channel Section":
            st.subheader("Channel Section Parameters")
            d = st.number_input("Overall Depth (mm)", value=300.0, min_value=0.1)
            b = st.number_input("Flange Width (mm)", value=100.0, min_value=0.1)
            tf = st.number_input("Flange Thickness (mm)", value=12.0, min_value=0.1)
            tw = st.number_input("Web Thickness (mm)", value=8.0, min_value=0.1)
            
            web = rectangular_section(b=tw, d=d, material=steel)
            top_flange = rectangular_section(b=b, d=tf, material=steel).shift_section(0, d-tf)
            bottom_flange = rectangular_section(b=b, d=tf, material=steel)
            geom = web + top_flange + bottom_flange
            
        else:  # T-Section
            st.subheader("T-Section Parameters")
            bf = st.number_input("Flange Width (mm)", value=200.0, min_value=0.1)
            tf = st.number_input("Flange Thickness (mm)", value=15.0, min_value=0.1)
            d = st.number_input("Overall Depth (mm)", value=300.0, min_value=0.1)
            tw = st.number_input("Web Thickness (mm)", value=10.0, min_value=0.1)
            
            flange = rectangular_section(b=bf, d=tf, material=steel).shift_section(0, d-tf)
            web = rectangular_section(b=tw, d=d-tf, material=steel).shift_section((bf-tw)/2, 0)
            geom = flange + web
    
    # ============ NODE-BASED INPUT METHOD ============
    elif input_method == "Node-Based Input":
        st.subheader("Define Section by Nodes")
        st.write("Create custom sections by defining corner points (nodes)")
        
        input_type = st.radio("Input Type", ["Manual Table", "Upload CSV"])
        
        if input_type == "Manual Table":
            st.write("**Edit Node Coordinates:**")
            st.info("💡 Define nodes in order (clockwise or counter-clockwise). The section will be closed automatically.")
            
            # Data editor for nodes
            edited_df = st.data_editor(
                st.session_state.node_data,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=False
            )
            st.session_state.node_data = edited_df
            
            # Add preset shapes
            st.write("**Quick Presets:**")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("Rectangle"):
                    st.session_state.node_data = pd.DataFrame({
                        'Node': [1, 2, 3, 4],
                        'X (mm)': [0.0, 200.0, 200.0, 0.0],
                        'Y (mm)': [0.0, 0.0, 300.0, 300.0]
                    })
                    st.rerun()
            
            with col_b:
                if st.button("L-Shape"):
                    st.session_state.node_data = pd.DataFrame({
                        'Node': [1, 2, 3, 4, 5, 6],
                        'X (mm)': [0.0, 100.0, 100.0, 15.0, 15.0, 0.0],
                        'Y (mm)': [0.0, 0.0, 15.0, 15.0, 200.0, 200.0]
                    })
                    st.rerun()
            
            with col_c:
                if st.button("Z-Shape"):
                    st.session_state.node_data = pd.DataFrame({
                        'Node': [1, 2, 3, 4, 5, 6],
                        'X (mm)': [0.0, 150.0, 150.0, 15.0, 15.0, 0.0],
                        'Y (mm)': [0.0, 0.0, 15.0, 15.0, 150.0, 150.0]
                    })
                    st.rerun()
            
        else:  # Upload CSV
            st.write("**Upload CSV file with columns: Node, X (mm), Y (mm)**")
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
            
            if uploaded_file is not None:
                st.session_state.node_data = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
            
            # Download template
            template_csv = st.session_state.node_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Template CSV",
                data=template_csv,
                file_name="node_template.csv",
                mime="text/csv"
            )
        
        # Preview nodes
        if not st.session_state.node_data.empty:
            st.write(f"**Total Nodes: {len(st.session_state.node_data)}**")
            
            # Create geometry from nodes
            try:
                coords = st.session_state.node_data[['X (mm)', 'Y (mm)']].values
                if len(coords) >= 3:
                    polygon = Polygon(coords)
                    geom = Geometry(geom=polygon, material=steel)
                else:
                    st.error("❌ Need at least 3 nodes to create a section")
                    geom = None
            except Exception as e:
                st.error(f"❌ Error creating geometry: {str(e)}")
                geom = None
    
    # ============ MULTIPLE COMPONENTS METHOD ============
    else:  # Multiple Components
        st.subheader("Build Section from Multiple Components")
        st.write("Add and position multiple shapes to create complex built-up sections")
        
        # Component management
        col_add, col_clear = st.columns(2)
        
        with col_add:
            component_type = st.selectbox(
                "Component Type",
                ["Plate", "I-Section", "Channel", "Angle"]
            )
        
        with st.expander("➕ Add New Component", expanded=True):
            if component_type == "Plate":
                c_width = st.number_input("Width (mm)", value=200.0, key="c_width")
                c_thick = st.number_input("Thickness (mm)", value=10.0, key="c_thick")
                c_x = st.number_input("X Position (mm)", value=0.0, key="c_x")
                c_y = st.number_input("Y Position (mm)", value=0.0, key="c_y")
                c_rotation = st.number_input("Rotation (degrees)", value=0.0, key="c_rot")
                
                if st.button("Add Plate"):
                    comp = rectangular_section(b=c_width, d=c_thick, material=steel)
                    if c_rotation != 0:
                        comp = comp.rotate_section(angle=c_rotation)
                    comp = comp.shift_section(x_offset=c_x, y_offset=c_y)
                    st.session_state.geometries.append(("Plate", comp))
                    st.success(f"✅ Added plate {len(st.session_state.geometries)}")
            
            elif component_type == "I-Section":
                c_d = st.number_input("Depth (mm)", value=400.0, key="i_d")
                c_bf = st.number_input("Flange Width (mm)", value=200.0, key="i_bf")
                c_tf = st.number_input("Flange Thickness (mm)", value=15.0, key="i_tf")
                c_tw = st.number_input("Web Thickness (mm)", value=10.0, key="i_tw")
                c_x = st.number_input("X Position (mm)", value=0.0, key="i_x")
                c_y = st.number_input("Y Position (mm)", value=0.0, key="i_y")
                c_rotation = st.number_input("Rotation (degrees)", value=0.0, key="i_rot")
                
                if st.button("Add I-Section"):
                    comp = i_section(d=c_d, b=c_bf, t_f=c_tf, t_w=c_tw, r=0, n_r=1, material=steel)
                    if c_rotation != 0:
                        comp = comp.rotate_section(angle=c_rotation)
                    comp = comp.shift_section(x_offset=c_x, y_offset=c_y)
                    st.session_state.geometries.append(("I-Section", comp))
                    st.success(f"✅ Added I-section {len(st.session_state.geometries)}")
            
            elif component_type == "Channel":
                c_d = st.number_input("Depth (mm)", value=300.0, key="ch_d")
                c_b = st.number_input("Width (mm)", value=100.0, key="ch_b")
                c_tf = st.number_input("Flange Thickness (mm)", value=12.0, key="ch_tf")
                c_tw = st.number_input("Web Thickness (mm)", value=8.0, key="ch_tw")
                c_x = st.number_input("X Position (mm)", value=0.0, key="ch_x")
                c_y = st.number_input("Y Position (mm)", value=0.0, key="ch_y")
                c_rotation = st.number_input("Rotation (degrees)", value=0.0, key="ch_rot")
                
                if st.button("Add Channel"):
                    web = rectangular_section(b=c_tw, d=c_d, material=steel)
                    top_fl = rectangular_section(b=c_b, d=c_tf, material=steel).shift_section(0, c_d-c_tf)
                    bot_fl = rectangular_section(b=c_b, d=c_tf, material=steel)
                    comp = web + top_fl + bot_fl
                    if c_rotation != 0:
                        comp = comp.rotate_section(angle=c_rotation)
                    comp = comp.shift_section(x_offset=c_x, y_offset=c_y)
                    st.session_state.geometries.append(("Channel", comp))
                    st.success(f"✅ Added channel {len(st.session_state.geometries)}")
            
            else:  # Angle
                c_d = st.number_input("Leg 1 Length (mm)", value=100.0, key="a_d")
                c_b = st.number_input("Leg 2 Length (mm)", value=100.0, key="a_b")
                c_t = st.number_input("Thickness (mm)", value=10.0, key="a_t")
                c_x = st.number_input("X Position (mm)", value=0.0, key="a_x")
                c_y = st.number_input("Y Position (mm)", value=0.0, key="a_y")
                c_rotation = st.number_input("Rotation (degrees)", value=0.0, key="a_rot")
                
                if st.button("Add Angle"):
                    leg1 = rectangular_section(b=c_t, d=c_d, material=steel)
                    leg2 = rectangular_section(b=c_b, d=c_t, material=steel)
                    comp = leg1 + leg2
                    if c_rotation != 0:
                        comp = comp.rotate_section(angle=c_rotation)
                    comp = comp.shift_section(x_offset=c_x, y_offset=c_y)
                    st.session_state.geometries.append(("Angle", comp))
                    st.success(f"✅ Added angle {len(st.session_state.geometries)}")
        
        # Display added components
        st.write(f"**Components Added: {len(st.session_state.geometries)}**")
        if st.session_state.geometries:
            for idx, (comp_type, _) in enumerate(st.session_state.geometries):
                col_info, col_del = st.columns([3, 1])
                with col_info:
                    st.write(f"{idx+1}. {comp_type}")
                with col_del:
                    if st.button("🗑️", key=f"del_{idx}"):
                        st.session_state.geometries.pop(idx)
                        st.rerun()
        
        with col_clear:
            if st.button("🗑️ Clear All", type="secondary"):
                st.session_state.geometries = []
                st.rerun()
        
        # Combine all geometries
        if st.session_state.geometries:
            geom = st.session_state.geometries[0][1]
            for _, comp_geom in st.session_state.geometries[1:]:
                geom = geom + comp_geom
        else:
            geom = None
            st.warning("⚠️ No components added yet")

# Analyze button
analyze_button = st.sidebar.button("🔍 Analyze Section", type="primary")

# Analysis section
if analyze_button and geom is not None:
    with st.spinner("Calculating section properties..."):
        try:
            # Create mesh
            geom.create_mesh(mesh_sizes=[50])
            
            # Create Section object and calculate properties
            section = Section(geometry=geom)
            section.calculate_geometric_properties()
            section.calculate_warping_properties()
            section.calculate_plastic_properties()
            
            # Get properties
            area = section.get_area()
            Ixx = section.get_ig()[0]
            Iyy = section.get_ig()[1]
            Ixy = section.get_ig()[2]
            cx, cy = section.get_c()
            Zxx_plus = section.get_z()[0]
            Zxx_minus = section.get_z()[1]
            Zyy_plus = section.get_z()[2]
            Zyy_minus = section.get_z()[3]
            rx = section.get_rc()[0]
            ry = section.get_rc()[1]
            Sxx = section.get_s()[0]
            Syy = section.get_s()[1]
            J = section.get_j()
            
            with col2:
                st.header("Analysis Results")
                
                # Display section plot
                st.subheader("Section Geometry")
                fig, ax = plt.subplots(figsize=(8, 6))
                section.plot_mesh(ax=ax, materials=False, mask=None, alpha=0.7)
                
                # Plot centroid
                ax.plot(cx, cy, 'r*', markersize=15, label='Centroid', zorder=5)
                ax.legend()
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                st.pyplot(fig)
                plt.close()
            
            # Create detailed results
            st.header("📊 Section Properties Report")
            
            # Geometric Properties
            st.subheader("1. Geometric Properties")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Cross-sectional Area", f"{area:.2f} mm²")
                st.metric("Centroid X", f"{cx:.2f} mm")
            
            with col_b:
                st.metric("Moment of Inertia Ixx", f"{Ixx:.2e} mm⁴")
                st.metric("Moment of Inertia Iyy", f"{Iyy:.2e} mm⁴")
            
            with col_c:
                st.metric("Product of Inertia Ixy", f"{Ixy:.2e} mm⁴")
                st.metric("Centroid Y", f"{cy:.2f} mm")
            
            # Section Moduli
            st.subheader("2. Section Moduli")
            col_d, col_e = st.columns(2)
            
            with col_d:
                st.write("**Elastic Section Modulus**")
                st.write(f"- Sxx = {Sxx:.2e} mm³")
                st.write(f"- Syy = {Syy:.2e} mm³")
            
            with col_e:
                st.write("**Plastic Section Modulus**")
                st.write(f"- Zxx+ = {Zxx_plus:.2e} mm³")
                st.write(f"- Zxx- = {Zxx_minus:.2e} mm³")
                st.write(f"- Zyy+ = {Zyy_plus:.2e} mm³")
                st.write(f"- Zyy- = {Zyy_minus:.2e} mm³")
            
            # Additional Properties
            st.subheader("3. Additional Properties")
            col_f, col_g = st.columns(2)
            
            with col_f:
                st.metric("Radius of Gyration rx", f"{rx:.2f} mm")
                st.metric("Radius of Gyration ry", f"{ry:.2f} mm")
            
            with col_g:
                st.metric("Torsion Constant J", f"{J:.2e} mm⁴")
                st.metric("Weight per meter", f"{area * 7850e-9 * 1000:.2f} kg/m")
            
            # Capacity Calculations
            st.subheader("4. Design Capacities (Simplified)")
            st.info("Note: These are simplified calculations. Consult relevant design codes for actual design.")
            
            # Plastic moment capacity
            Mp_xx = Zxx_plus * fy / 1e6  # kN·m
            Mp_yy = Zyy_plus * fy / 1e6  # kN·m
            
            col_h, col_i = st.columns(2)
            with col_h:
                st.metric("Plastic Moment Capacity Mpx", f"{Mp_xx:.2f} kN·m")
            with col_i:
                st.metric("Plastic Moment Capacity Mpy", f"{Mp_yy:.2f} kN·m")
            
            # Generate downloadable report
            st.subheader("5. Download Report")
            
            report_data = {
                "Property": [
                    "Cross-sectional Area",
                    "Centroid X",
                    "Centroid Y",
                    "Moment of Inertia Ixx",
                    "Moment of Inertia Iyy",
                    "Product of Inertia Ixy",
                    "Elastic Section Modulus Sxx",
                    "Elastic Section Modulus Syy",
                    "Plastic Section Modulus Zxx+",
                    "Plastic Section Modulus Zxx-",
                    "Plastic Section Modulus Zyy+",
                    "Plastic Section Modulus Zyy-",
                    "Radius of Gyration rx",
                    "Radius of Gyration ry",
                    "Torsion Constant J",
                    "Weight per meter",
                    "Plastic Moment Capacity Mpx",
                    "Plastic Moment Capacity Mpy"
                ],
                "Value": [
                    f"{area:.2f}",
                    f"{cx:.2f}",
                    f"{cy:.2f}",
                    f"{Ixx:.2e}",
                    f"{Iyy:.2e}",
                    f"{Ixy:.2e}",
                    f"{Sxx:.2e}",
                    f"{Syy:.2e}",
                    f"{Zxx_plus:.2e}",
                    f"{Zxx_minus:.2e}",
                    f"{Zyy_plus:.2e}",
                    f"{Zyy_minus:.2e}",
                    f"{rx:.2f}",
                    f"{ry:.2f}",
                    f"{J:.2e}",
                    f"{area * 7850e-9 * 1000:.2f}",
                    f"{Mp_xx:.2f}",
                    f"{Mp_yy:.2f}"
                ],
                "Unit": [
                    "mm²", "mm", "mm", "mm⁴", "mm⁴", "mm⁴",
                    "mm³", "mm³", "mm³", "mm³", "mm³", "mm³",
                    "mm", "mm", "mm⁴", "kg/m", "kN·m", "kN·m"
                ]
            }
            
            df_report = pd.DataFrame(report_data)
            
            csv = df_report.to_csv(index=False)
            st.download_button(
                label="📥 Download Report as CSV",
                data=csv,
                file_name="section_properties_report.csv",
                mime="text/csv"
            )
            
            # Display table
            st.dataframe(df_report, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.write("Please check your section definition and try again.")
            
else:
    if not analyze_button:
        with col2:
            st.info("👈 Configure your section and click 'Analyze Section' to see results")
    elif geom is None:
        with col2:
            st.warning("⚠️ Please define a valid section geometry first")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit and sectionproperties library | For educational purposes*")
