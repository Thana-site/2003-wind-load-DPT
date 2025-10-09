"""
UI Components Module
Reusable UI components for the Streamlit application
"""

import streamlit as st
import pandas as pd

class UIComponents:
    """Collection of reusable UI components for section input"""
    
    @staticmethod
    def get_rectangle_inputs():
        """Get rectangle parameters from user"""
        col1, col2 = st.columns(2)
        
        with col1:
            width = st.number_input(
                "Width [mm]", 
                min_value=10.0, 
                max_value=2000.0, 
                value=200.0,
                step=10.0,
                help="Width of the rectangle"
            )
        
        with col2:
            depth = st.number_input(
                "Depth [mm]", 
                min_value=10.0, 
                max_value=2000.0, 
                value=300.0,
                step=10.0,
                help="Depth/height of the rectangle"
            )
        
        return {
            'width': width,
            'depth': depth,
            'mesh_size': 10
        }
    
    @staticmethod
    def get_ibeam_inputs():
        """Get I-beam parameters from user"""
        col1, col2 = st.columns(2)
        
        with col1:
            depth = st.number_input(
                "Depth (d) [mm]", 
                min_value=10.0, 
                max_value=2000.0, 
                value=300.0,
                step=10.0,
                help="Overall depth of the I-beam"
            )
            width = st.number_input(
                "Width (b) [mm]", 
                min_value=10.0, 
                max_value=1000.0, 
                value=150.0,
                step=10.0,
                help="Width of the flanges"
            )
        
        with col2:
            flange_thickness = st.number_input(
                "Flange Thickness (tf) [mm]", 
                min_value=1.0, 
                max_value=100.0, 
                value=10.0,
                step=1.0,
                help="Thickness of the flanges"
            )
            web_thickness = st.number_input(
                "Web Thickness (tw) [mm]", 
                min_value=1.0, 
                max_value=50.0, 
                value=6.0,
                step=1.0,
                help="Thickness of the web"
            )
        
        # Optional parameters
        with st.expander("Advanced Options"):
            root_radius = st.number_input(
                "Root Radius [mm]", 
                min_value=0.0, 
                max_value=50.0, 
                value=0.0,
                help="Fillet radius at flange-web junction"
            )
            mesh_size = st.slider(
                "Mesh Size [mm]", 
                min_value=1, 
                max_value=50, 
                value=10,
                help="Mesh size for analysis"
            )
        else:
            root_radius = 0.0
            mesh_size = 10
        
        return {
            'depth': depth,
            'width': width,
            'flange_thickness': flange_thickness,
            'web_thickness': web_thickness,
            'root_radius': root_radius,
            'mesh_size': mesh_size
        }
    
    @staticmethod
    def get_box_inputs():
        """Get box section parameters from user"""
        col1, col2 = st.columns(2)
        
        with col1:
            depth = st.number_input(
                "Depth [mm]", 
                min_value=10.0, 
                max_value=2000.0, 
                value=300.0,
                step=10.0,
                help="Overall depth of the box section"
            )
            width = st.number_input(
                "Width [mm]", 
                min_value=10.0, 
                max_value=1000.0, 
                value=200.0,
                step=10.0,
                help="Overall width of the box section"
            )
        
        with col2:
            uniform = st.checkbox("Uniform wall thickness", value=True)
            
            if uniform:
                wall_thickness = st.number_input(
                    "Wall Thickness [mm]", 
                    min_value=1.0, 
                    max_value=50.0, 
                    value=8.0,
                    step=1.0,
                    help="Uniform wall thickness"
                )
                params = {
                    'depth': depth,
                    'width': width,
                    'wall_thickness': wall_thickness,
                    'mesh_size': 10
                }
            else:
                web_thickness = st.number_input(
                    "Web Thickness [mm]", 
                    min_value=1.0, 
                    max_value=50.0, 
                    value=6.0,
                    step=1.0,
                    help="Thickness of vertical walls"
                )
                flange_thickness = st.number_input(
                    "Flange Thickness [mm]", 
                    min_value=1.0, 
                    max_value=50.0, 
                    value=8.0,
                    step=1.0,
                    help="Thickness of horizontal walls"
                )
                params = {
                    'depth': depth,
                    'width': width,
                    'web_thickness': web_thickness,
                    'flange_thickness': flange_thickness,
                    'mesh_size': 10
                }
        
        return params
    
    @staticmethod
    def get_channel_inputs():
        """Get channel section parameters from user"""
        col1, col2 = st.columns(2)
        
        with col1:
            depth = st.number_input(
                "Depth [mm]", 
                min_value=10.0, 
                max_value=2000.0, 
                value=250.0,
                step=10.0,
                help="Overall depth of the channel"
            )
            width = st.number_input(
                "Width [mm]", 
                min_value=10.0, 
                max_value=1000.0, 
                value=100.0,
                step=10.0,
                help="Width of the flanges"
            )
        
        with col2:
            flange_thickness = st.number_input(
                "Flange Thickness [mm]", 
                min_value=1.0, 
                max_value=100.0, 
                value=8.0,
                step=1.0,
                help="Thickness of the flanges"
            )
            web_thickness = st.number_input(
                "Web Thickness [mm]", 
                min_value=1.0, 
                max_value=50.0, 
                value=6.0,
                step=1.0,
                help="Thickness of the web"
            )
        
        return {
            'depth': depth,
            'width': width,
            'flange_thickness': flange_thickness,
            'web_thickness': web_thickness,
            'mesh_size': 10
        }
    
    @staticmethod
    def get_circular_inputs():
        """Get circular section parameters from user"""
        diameter = st.number_input(
            "Diameter [mm]", 
            min_value=10.0, 
            max_value=2000.0, 
            value=200.0,
            step=10.0,
            help="Diameter of the circle"
        )
        
        n_circle = st.slider(
            "Number of segments", 
            min_value=16, 
            max_value=128, 
            value=64,
            step=8,
            help="Number of segments to approximate the circle"
        )
        
        return {
            'diameter': diameter,
            'n_circle': n_circle,
            'mesh_size': 10
        }
    
    @staticmethod
    def get_circular_hollow_inputs():
        """Get circular hollow section parameters from user"""
        col1, col2 = st.columns(2)
        
        with col1:
            outer_diameter = st.number_input(
                "Outer Diameter [mm]", 
                min_value=10.0, 
                max_value=2000.0, 
                value=200.0,
                step=10.0,
                help="Outer diameter of the tube"
            )
        
        with col2:
            thickness = st.number_input(
                "Wall Thickness [mm]", 
                min_value=1.0, 
                max_value=100.0, 
                value=10.0,
                step=1.0,
                help="Wall thickness of the tube"
            )
        
        n_circle = st.slider(
            "Number of segments", 
            min_value=16, 
            max_value=128, 
            value=64,
            step=8,
            help="Number of segments to approximate the circle"
        )
        
        return {
            'outer_diameter': outer_diameter,
            'thickness': thickness,
            'n_circle': n_circle,
            'mesh_size': 10
        }
    
    @staticmethod
    def get_tsection_inputs():
        """Get T-section parameters from user"""
        col1, col2 = st.columns(2)
        
        with col1:
            depth = st.number_input(
                "Depth [mm]", 
                min_value=10.0, 
                max_value=2000.0, 
                value=200.0,
                step=10.0,
                help="Overall depth of the T-section"
            )
            width = st.number_input(
                "Width [mm]", 
                min_value=10.0, 
                max_value=1000.0, 
                value=150.0,
                step=10.0,
                help="Width of the flange"
            )
        
        with col2:
            flange_thickness = st.number_input(
                "Flange Thickness [mm]", 
                min_value=1.0, 
                max_value=100.0, 
                value=10.0,
                step=1.0,
                help="Thickness of the flange"
            )
            web_thickness = st.number_input(
                "Web Thickness [mm]", 
                min_value=1.0, 
                max_value=50.0, 
                value=8.0,
                step=1.0,
                help="Thickness of the web"
            )
        
        return {
            'depth': depth,
            'width': width,
            'flange_thickness': flange_thickness,
            'web_thickness': web_thickness,
            'mesh_size': 10
        }
    
    @staticmethod
    def get_angle_inputs():
        """Get angle section parameters from user"""
        col1, col2 = st.columns(2)
        
        with col1:
            leg1_length = st.number_input(
                "Leg 1 Length [mm]", 
                min_value=10.0, 
                max_value=500.0, 
                value=100.0,
                step=10.0,
                help="Length of the first leg"
            )
            leg2_length = st.number_input(
                "Leg 2 Length [mm]", 
                min_value=10.0, 
                max_value=500.0, 
                value=100.0,
                step=10.0,
                help="Length of the second leg"
            )
        
        with col2:
            thickness = st.number_input(
                "Thickness [mm]", 
                min_value=1.0, 
                max_value=50.0, 
                value=8.0,
                step=1.0,
                help="Thickness of both legs"
            )
        
        return {
            'leg1_length': leg1_length,
            'leg2_length': leg2_length,
            'thickness': thickness,
            'mesh_size': 10
        }
    
    @staticmethod
    def display_results_table(properties: dict, title: str = "Section Properties"):
        """Display properties in a formatted table"""
        st.markdown(f"### {title}")
        
        # Group properties by category
        geometric = {
            'Area': properties.get('area', 0),
            'Perimeter': properties.get('perimeter', 0),
            'Centroid X': properties.get('cx', 0),
            'Centroid Y': properties.get('cy', 0)
        }
        
        moments = {
            'Ixx': properties.get('ixx_c', 0),
            'Iyy': properties.get('iyy_c', 0),
            'Ixy': properties.get('ixy_c', 0),
            'I11': properties.get('i11_c', 0),
            'I22': properties.get('i22_c', 0)
        }
        
        moduli = {
            'Zxx+': properties.get('zxx_plus', 0),
            'Zxx-': properties.get('zxx_minus', 0),
            'Zyy+': properties.get('zyy_plus', 0),
            'Zyy-': properties.get('zyy_minus', 0),
            'Sxx': properties.get('sxx', 0),
            'Syy': properties.get('syy', 0)
        }
        
        other = {
            'rx': properties.get('rx', 0),
            'ry': properties.get('ry', 0),
            'J': properties.get('j', 0),
            'Γ': properties.get('gamma', 0),
            'φ': properties.get('phi', 0)
        }
        
        # Create tabs for different property groups
        tabs = st.tabs(["Geometric", "Moments", "Moduli", "Other"])
        
        with tabs[0]:
            df = pd.DataFrame(geometric.items(), columns=['Property', 'Value'])
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        with tabs[1]:
            df = pd.DataFrame(moments.items(), columns=['Property', 'Value'])
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        with tabs[2]:
            df = pd.DataFrame(moduli.items(), columns=['Property', 'Value'])
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        with tabs[3]:
            df = pd.DataFrame(other.items(), columns=['Property', 'Value'])
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    @staticmethod
    def format_value(value: float, decimals: int = 3) -> str:
        """Format numerical values for display"""
        if abs(value) > 1e6 or (abs(value) < 1e-3 and value != 0):
            return f"{value:.{decimals}e}"
        else:
            return f"{value:.{decimals}f}"
    
    @staticmethod
    def create_property_card(title: str, value: float, unit: str, color: str = "#6366f1"):
        """Create a styled property card"""
        formatted_value = UIComponents.format_value(value)
        
        st.markdown(f"""
        <div style="
            background: white;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid {color};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="color: #6b7280; font-size: 0.875rem;">{title}</div>
            <div style="color: #1f2937; font-size: 1.5rem; font-weight: 600; margin: 0.25rem 0;">
                {formatted_value}
            </div>
            <div style="color: #9ca3af; font-size: 0.75rem;">{unit}</div>
        </div>
        """, unsafe_allow_html=True)
