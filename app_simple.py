"""
Simplified Section Analyzer - Fallback Version
Use this if the main app fails to run
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Section Analyzer (Simple)",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Section Properties Calculator - Simple Version")
st.info("This is a simplified version. If you see this, the full app had loading issues.")

# Basic section calculations
def calculate_rectangle(width, height):
    """Calculate properties for a rectangular section"""
    area = width * height
    ixx = (width * height**3) / 12
    iyy = (height * width**3) / 12
    rx = np.sqrt(ixx / area)
    ry = np.sqrt(iyy / area)
    zxx = ixx / (height / 2)
    zyy = iyy / (width / 2)
    
    return {
        'Area': area,
        'Ixx': ixx,
        'Iyy': iyy,
        'rx': rx,
        'ry': ry,
        'Zxx': zxx,
        'Zyy': zyy
    }

def calculate_circle(diameter):
    """Calculate properties for a circular section"""
    radius = diameter / 2
    area = np.pi * radius**2
    i_moment = (np.pi * diameter**4) / 64
    r_gyration = diameter / 4
    z_modulus = (np.pi * diameter**3) / 32
    
    return {
        'Area': area,
        'Ixx': i_moment,
        'Iyy': i_moment,
        'rx': r_gyration,
        'ry': r_gyration,
        'Zxx': z_modulus,
        'Zyy': z_modulus
    }

def calculate_i_beam(h, b, tw, tf):
    """Calculate approximate properties for an I-beam"""
    # Simplified calculation (approximate)
    area = b * tf * 2 + tw * (h - 2 * tf)
    
    # Moment of inertia about x-axis (simplified)
    ixx = (b * h**3 - (b - tw) * (h - 2*tf)**3) / 12
    
    # Moment of inertia about y-axis
    iyy = (2 * tf * b**3 + (h - 2*tf) * tw**3) / 12
    
    zxx = ixx / (h / 2)
    zyy = iyy / (b / 2)
    
    rx = np.sqrt(ixx / area)
    ry = np.sqrt(iyy / area)
    
    return {
        'Area': area,
        'Ixx': ixx,
        'Iyy': iyy,
        'rx': rx,
        'ry': ry,
        'Zxx': zxx,
        'Zyy': zyy
    }

# Sidebar inputs
with st.sidebar:
    st.header("Section Configuration")
    
    section_type = st.selectbox(
        "Select Section Type",
        ["Rectangle", "Circle", "I-Beam (Simplified)"]
    )
    
    st.subheader("Dimensions")
    
    if section_type == "Rectangle":
        width = st.number_input("Width [mm]", value=100.0, min_value=1.0)
        height = st.number_input("Height [mm]", value=200.0, min_value=1.0)
        
        if st.button("Calculate", type="primary"):
            results = calculate_rectangle(width, height)
            st.session_state['results'] = results
            st.session_state['type'] = "Rectangle"
            st.session_state['params'] = f"Width: {width}mm, Height: {height}mm"
    
    elif section_type == "Circle":
        diameter = st.number_input("Diameter [mm]", value=100.0, min_value=1.0)
        
        if st.button("Calculate", type="primary"):
            results = calculate_circle(diameter)
            st.session_state['results'] = results
            st.session_state['type'] = "Circle"
            st.session_state['params'] = f"Diameter: {diameter}mm"
    
    else:  # I-Beam
        h = st.number_input("Height [mm]", value=300.0, min_value=10.0)
        b = st.number_input("Flange Width [mm]", value=150.0, min_value=10.0)
        tw = st.number_input("Web Thickness [mm]", value=6.0, min_value=1.0)
        tf = st.number_input("Flange Thickness [mm]", value=10.0, min_value=1.0)
        
        if st.button("Calculate", type="primary"):
            results = calculate_i_beam(h, b, tw, tf)
            st.session_state['results'] = results
            st.session_state['type'] = "I-Beam"
            st.session_state['params'] = f"H:{h}mm, B:{b}mm, tw:{tw}mm, tf:{tf}mm"

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📊 Results")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Display metrics
        st.metric("Area", f"{results['Area']:.2f} mm²")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Ixx", f"{results['Ixx']:.2e} mm⁴")
            st.metric("Zxx", f"{results['Zxx']:.2e} mm³")
            st.metric("rx", f"{results['rx']:.2f} mm")
        
        with col_b:
            st.metric("Iyy", f"{results['Iyy']:.2e} mm⁴")
            st.metric("Zyy", f"{results['Zyy']:.2e} mm³")
            st.metric("ry", f"{results['ry']:.2f} mm")
    else:
        st.info("Configure section and click Calculate")

with col2:
    st.header("📋 Properties Table")
    
    if 'results' in st.session_state:
        # Create DataFrame
        df = pd.DataFrame([
            ["Section Type", st.session_state.get('type', 'N/A'), ""],
            ["Parameters", st.session_state.get('params', 'N/A'), ""],
            ["", "", ""],
            ["Area", f"{results['Area']:.2f}", "mm²"],
            ["Moment of Inertia Ixx", f"{results['Ixx']:.2e}", "mm⁴"],
            ["Moment of Inertia Iyy", f"{results['Iyy']:.2e}", "mm⁴"],
            ["Section Modulus Zxx", f"{results['Zxx']:.2e}", "mm³"],
            ["Section Modulus Zyy", f"{results['Zyy']:.2e}", "mm³"],
            ["Radius of Gyration rx", f"{results['rx']:.2f}", "mm"],
            ["Radius of Gyration ry", f"{results['ry']:.2f}", "mm"],
        ], columns=["Property", "Value", "Unit"])
        
        st.dataframe(df, hide_index=True, use_container_width=True)
        
        # Export button
        csv = df.to_csv(index=False)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"section_properties_{timestamp}.csv",
            mime="text/csv"
        )
    else:
        st.info("No results to display")

# Help section
with st.expander("ℹ️ Help & Formulas"):
    st.markdown("""
    ### Basic Formulas Used:
    
    **Rectangle:**
    - Area = width × height
    - Ixx = (width × height³) / 12
    - Iyy = (height × width³) / 12
    
    **Circle:**
    - Area = π × radius²
    - Ixx = Iyy = (π × diameter⁴) / 64
    - Zxx = Zyy = (π × diameter³) / 32
    
    **I-Beam (Simplified):**
    - Approximate formulas for quick estimation
    - For accurate results, use the full version with sectionproperties
    
    ### Troubleshooting:
    
    If you're seeing this simplified version, the full app couldn't load.
    
    **Common fixes:**
    1. Install missing packages:
       ```bash
       pip install sectionproperties shapely
       ```
    
    2. Run the setup script:
       ```bash
       python setup_and_run.py
       ```
    
    3. Check Python version (3.7+ required):
       ```bash
       python --version
       ```
    """)

st.markdown("---")
st.caption("Simplified Section Analyzer - Use for basic calculations only")
