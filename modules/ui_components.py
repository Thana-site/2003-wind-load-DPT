"""
UI Components Module
Reusable UI components for the Streamlit application
"""

import streamlit as st
import pandas as pd

class UIComponents:
    """Collection of reusable UI components"""
    
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
                    'wall_thickness': wall_thickness
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
                    'flange_thickness': flange_thickness
                }
        
        # Mesh size
        params['mesh_size'] = st.slider(
            "Mesh Size [mm]", 
            min_value=1, 
            max_value=50, 
            value=10,
            help="Mesh size for analysis"
        )
        
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
                max_value=500.0, 
                value=100.0,
                step=10.0,
                help="Width of the flanges"
            )
        
        with col2:
            flange_thickness = st.number_input(
                "Flange Thickness [mm]", 
                min_value=1.0, 
                max_value=50.0, 
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
            value=100.0,
            step=10.0,
            help="Diameter of the circular section"
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
        """Get circular hollow section (pipe) parameters from user"""
        col1, col2 = st.columns(2)
        
        with col1:
            outer_diameter = st.number_input(
                "Outer Diameter [mm]", 
                min_value=10.0, 
                max_value=2000.0, 
                value=150.0,
                step=10.0,
                help="Outer diameter of the pipe"
            )
        
        with col2:
            thickness = st.number_input(
                "Wall Thickness [mm]", 
                min_value=1.0, 
                max_value=100.0, 
                value=5.0,
                step=1.0,
                help="Wall thickness of the pipe"
            )
        
        # Validation
        if thickness >= outer_diameter / 2:
            st.error("Wall thickness must be less than radius!")
        
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
    def get_polygon_inputs():
        """Get custom polygon parameters from user"""
        st.write("Define polygon nodes (vertices)")
        
        # Node input method
        input_method = st.radio(
            "Input Method",
            ["Interactive Table", "Paste Coordinates", "Upload CSV"],
            horizontal=True
        )
        
        if input_method == "Interactive Table":
            # Dynamic node table
            if 'node_count' not in st.session_state:
                st.session_state.node_count = 4
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("➕ Add Node"):
                    st.session_state.node_count += 1
                if st.button("➖ Remove Node") and st.session_state.node_count > 3:
                    st.session_state.node_count -= 1
            
            # Create node input table
            nodes = []
            cols = st.columns(2)
            
            for i in range(st.session_state.node_count):
                with cols[i % 2]:
                    x = st.number_input(
                        f"Node {i+1} X [mm]",
                        value=float(i * 100) if i < 4 else 0.0,
                        key=f"node_x_{i}"
                    )
                    y = st.number_input(
                        f"Node {i+1} Y [mm]",
                        value=float((i % 2) * 100) if i < 4 else 0.0,
                        key=f"node_y_{i}"
                    )
                    nodes.append((x, y))
            
            st.session_state.polygon_nodes = nodes
            
        elif input_method == "Paste Coordinates":
            coords_text = st.text_area(
                "Paste coordinates (x,y pairs, one per line)",
                value="0,0\n100,0\n100,100\n0,100",
                height=150
            )
            
            try:
                nodes = []
                for line in coords_text.strip().split('\n'):
                    if line:
                        x, y = map(float, line.split(','))
                        nodes.append((x, y))
                st.session_state.polygon_nodes = nodes
            except:
                st.error("Invalid coordinate format. Use: x,y")
                nodes = st.session_state.polygon_nodes
        
        else:  # Upload CSV
            uploaded_file = st.file_uploader(
                "Upload CSV with X,Y columns",
                type=['csv']
            )
            
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                if 'X' in df.columns and 'Y' in df.columns:
                    nodes = list(zip(df['X'], df['Y']))
                    st.session_state.polygon_nodes = nodes
                else:
                    st.error("CSV must have 'X' and 'Y' columns")
                    nodes = st.session_state.polygon_nodes
            else:
                nodes = st.session_state.polygon_nodes
        
        # Display preview
        if len(nodes) >= 3:
            st.success(f"✅ {len(nodes)} nodes defined")
            
            # Option to add holes
            add_holes = st.checkbox("Add holes to the section")
            holes = []
            
            if add_holes:
                st.write("Define hole polygon (must be inside main polygon)")
                hole_nodes = []
                for i in range(4):
                    col1, col2 = st.columns(2)
                    with col1:
                        hx = st.number_input(
                            f"Hole Node {i+1} X",
                            value=25.0 + i*10,
                            key=f"hole_x_{i}"
                        )
                    with col2:
                        hy = st.number_input(
                            f"Hole Node {i+1} Y",
                            value=25.0 + (i%2)*10,
                            key=f"hole_y_{i}"
                        )
                    hole_nodes.append((hx, hy))
                holes = [hole_nodes]
        else:
            st.error("Need at least 3 nodes to define a polygon")
            nodes = [(0,0), (100,0), (100,100), (0,100)]
        
        params = {
            'nodes': nodes,
            'mesh_size': 10
        }
        
        if holes:
            params['holes'] = holes
        
        return params
    
    @staticmethod
    def display_quick_stats(properties):
        """Display quick statistics in metrics format"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Area", f"{properties['area']:.2f} mm²")
        with col2:
            st.metric("Ixx", f"{properties['ixx_c']:.2e} mm⁴")
        with col3:
            st.metric("Iyy", f"{properties['iyy_c']:.2e} mm⁴")
        with col4:
            st.metric("J", f"{properties['j']:.2e} mm⁴")
