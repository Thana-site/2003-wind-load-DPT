import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sectionproperties.pre.library import rectangular_section, i_section
from sectionproperties.pre import Material, Geometry
from sectionproperties.analysis import Section
from io import BytesIO
import pandas as pd

# Page configuration
st.set_page_config(page_title="Steel Section Properties Analyzer", layout="wide")

# Title and description
st.title("🏗️ Steel Built-Up Section Properties Analyzer")
st.markdown("Analyze geometric and mechanical properties of steel built-up sections")

# Sidebar for section type selection
st.sidebar.header("Section Configuration")
section_type = st.sidebar.selectbox(
    "Select Section Type",
    ["I-Section (Welded)", "Box Section", "Channel Section", "T-Section", "Custom Plates"]
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

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Section Dimensions")
    
    if section_type == "I-Section (Welded)":
        st.subheader("I-Section Parameters")
        d = st.number_input("Overall Depth (mm)", value=400.0, min_value=0.1)
        bf = st.number_input("Flange Width (mm)", value=200.0, min_value=0.1)
        tf = st.number_input("Flange Thickness (mm)", value=15.0, min_value=0.1)
        tw = st.number_input("Web Thickness (mm)", value=10.0, min_value=0.1)
        
        # Create I-section geometry
        geom = i_section(d=d, b=bf, t_f=tf, t_w=tw, r=0, n_r=1, material=steel)
        
    elif section_type == "Box Section":
        st.subheader("Box Section Parameters")
        b = st.number_input("Width (mm)", value=300.0, min_value=0.1)
        d = st.number_input("Depth (mm)", value=400.0, min_value=0.1)
        tf = st.number_input("Flange Thickness (mm)", value=12.0, min_value=0.1)
        tw = st.number_input("Web Thickness (mm)", value=10.0, min_value=0.1)
        
        # Create box section using rectangular sections
        outer = rectangular_section(b=b, d=d, material=steel)
        inner = rectangular_section(b=b-2*tw, d=d-2*tf, material=steel).shift_section(tw, tf)
        geom = outer - inner
        
    elif section_type == "Channel Section":
        st.subheader("Channel Section Parameters")
        d = st.number_input("Overall Depth (mm)", value=300.0, min_value=0.1)
        b = st.number_input("Flange Width (mm)", value=100.0, min_value=0.1)
        tf = st.number_input("Flange Thickness (mm)", value=12.0, min_value=0.1)
        tw = st.number_input("Web Thickness (mm)", value=8.0, min_value=0.1)
        
        # Create channel section
        web = rectangular_section(b=tw, d=d, material=steel)
        top_flange = rectangular_section(b=b, d=tf, material=steel).shift_section(0, d-tf)
        bottom_flange = rectangular_section(b=b, d=tf, material=steel)
        geom = web + top_flange + bottom_flange
        
    elif section_type == "T-Section":
        st.subheader("T-Section Parameters")
        bf = st.number_input("Flange Width (mm)", value=200.0, min_value=0.1)
        tf = st.number_input("Flange Thickness (mm)", value=15.0, min_value=0.1)
        d = st.number_input("Overall Depth (mm)", value=300.0, min_value=0.1)
        tw = st.number_input("Web Thickness (mm)", value=10.0, min_value=0.1)
        
        # Create T-section
        flange = rectangular_section(b=bf, d=tf, material=steel).shift_section(0, d-tf)
        web = rectangular_section(b=tw, d=d-tf, material=steel).shift_section((bf-tw)/2, 0)
        geom = flange + web
        
    else:  # Custom Plates
        st.subheader("Custom Built-Up Section")
        st.write("Define individual plates")
        num_plates = st.number_input("Number of Plates", min_value=1, max_value=10, value=3)
        
        plates = []
        for i in range(num_plates):
            with st.expander(f"Plate {i+1}"):
                width = st.number_input(f"Width (mm)", value=200.0, key=f"w{i}")
                thickness = st.number_input(f"Thickness (mm)", value=10.0, key=f"t{i}")
                x_pos = st.number_input(f"X Position (mm)", value=0.0, key=f"x{i}")
                y_pos = st.number_input(f"Y Position (mm)", value=float(i*100), key=f"y{i}")
                
                plate = rectangular_section(b=width, d=thickness, material=steel).shift_section(x_pos, y_pos)
                plates.append(plate)
        
        geom = plates[0]
        for plate in plates[1:]:
            geom = geom + plate

# Calculate section properties
if st.sidebar.button("🔍 Analyze Section", type="primary"):
    with st.spinner("Calculating section properties..."):
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
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
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
        
else:
    with col2:
        st.info("👈 Configure your section and click 'Analyze Section' to see results")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit and sectionproperties library | For educational purposes*")
