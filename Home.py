import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import re

# ===================== PAGE CONFIGURATION =====================
st.set_page_config(
    page_title="Advanced Pile Foundation Analysis",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .foundation-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .formula-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .tension-warning {
        background-color: #ffe6e6;
        border: 2px solid #ff0000;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .properties-table {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .critical-load {
        background-color: #fffacd;
        padding: 0.5rem;
        border-left: 4px solid #ffa500;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===================== TITLE =====================
st.markdown('<h1 class="main-header">🏗️ Advanced Pile Foundation Analysis Tool</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Theory-Based Design with Custom Foundation Support</p>', unsafe_allow_html=True)

# ===================== SESSION STATE INITIALIZATION =====================
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'final_results' not in st.session_state:
    st.session_state.final_results = None
if 'custom_foundations' not in st.session_state:
    st.session_state.custom_foundations = {}
if 'node_assignments' not in st.session_state:
    st.session_state.node_assignments = {}
if 'allowed_foundations' not in st.session_state:
    st.session_state.allowed_foundations = []
if 'tension_nodes' not in st.session_state:
    st.session_state.tension_nodes = []
if 'foundation_properties' not in st.session_state:
    st.session_state.foundation_properties = {}

# ===================== DEFAULT CONFIGURATIONS =====================
DEFAULT_FOUNDATIONS = {
    'F1': {
        'name': '1-Pile Single',
        'piles': 1,
        'color': '#FF6B6B',
        'coords': [(0, 0)]
    },
    
    'F2': {
        'name': '2-Pile Linear',
        'piles': 2,
        'color': '#FF8C42',
        'coords': [(0.9, 0), (-0.9, 0)]
    },
    
    'F3': {
        'name': '3-Pile Triangle',
        'piles': 3,
        'color': '#FFA726',
        'coords': [(0, 1.2), (-0.9, -0.6), (0.9, -0.6)]
    },
    
    'F4': {
        'name': '4-Pile Square',
        'piles': 4,
        'color': '#4ECDC4',
        'coords': [(-0.9, 0.9), (0.9, 0.9), (-0.9, -0.9), (0.9, -0.9)]
    },
    
    'F5': {
        'name': '5-Pile Square+Center',
        'piles': 5,
        'color': '#45B7D1',
        'coords': [(0, 0),(-0.9, 0.9), (0.9, 0.9), (-0.9, -0.9), (0.9, -0.9)]
    },
    
    'F6': {
        'name': '6-Pile Rectangle 2×3',
        'piles': 6,
        'color': '#96CEB4',
        'coords': [(-0.9, 0), (0.9, 0), (-0.9, 1.8), (0.9, 1.8), (-0.9, -1.8), (0.9, -1.8)]
    },
    
    'F7': {
        'name': '7-Pile Hexagon+Center',
        'piles': 7,
        'color': '#FFEAA7',
        'coords': [
            (0, 0),  # Center
            (1.3, 1.3),  # Right
            (1.3, -1.3),  # Top-right
            (-1.3, 1.3),  # Top-left
            (-1.3, -1.3),  # Left
            (0, -2.6),  # Bottom-left
            (0, 2.6)  # Bottom-right
        ]
    },
    
    'F8': {
        'name': '8-Pile Rectangle 2×4',
        'piles': 8,
        'color': '#DDA0DD',
        'coords': [(-1.8, 0.9), (0, 0.9), (1.8, 0.9), (-1.8, -0.9), (0, -0.9), (1.8, -0.9), (0, -1.8), (0, 1.8)]
    },
    
    'F9': {
        'name': '9-Pile Square 3×3',
        'piles': 9,
        'color': '#98D8C8',
        'coords': [(-1.8, 0), (0, 0), (1.8, 0), (-1.8, 1.8), (0, 1.8), (1.8, 1.8), (-1.8, -1.8), (0, -1.8), (1.8, -1.8)]
    },
    
    'F10': {
        'name': '9-Pile Square 3×3',
        'piles': 9,
        'color': '#98D8C8',
        'coords': [(1.8, 0), (-1.8, 0), (0, 0.9), (0, -0.9), (1.8, -1.8), (-1.8, -1.8), (-1.8, 1.8), (1.8, 1.8), (0,2.7), (0, -2.7)]
    },
}

DEFAULT_NODES = [29]

# Load combination patterns - Serviceability only (no MRSA)
LOAD_COMBINATIONS = {
    "cLCB70": "SERV :D + (L)",
    "cLCB71": "SERV :D + (L) + Wx",
    "cLCB72": "SERV :D + (L) + Wy",
    "cLCB73": "SERV :D + (L) + Wz",
    "cLCB74": "SERV :D + (L) - Wx",
    "cLCB75": "SERV :D + (L) - Wy",
    "cLCB76": "SERV :D + (L) - Wz",
    "cLCB77": "SERV :D + Wx",
    "cLCB78": "SERV :D + Wy",
    "cLCB79": "SERV :D + Wz",
    "cLCB80": "SERV :D - Wx",
    "cLCB81": "SERV :D - Wy",
    "cLCB82": "SERV :D - Wz",
    "cLCB83": "SERV :D + (L) + Ex",
    "cLCB84": "SERV :D + (L) + Ey",
    "cLCB85": "SERV :D + (L) - Ex",
    "cLCB86": "SERV :D + (L) - Ey",
    "cLCB87": "SERV :D + Ex",
    "cLCB88": "SERV :D + Ey",
    "cLCB89": "SERV :D - Ex",
    "cLCB90": "SERV :D - Ey",
    "cLCB91": "SERV :D + (L) + (1.0)(RSA_Ex(RS)+RSA_Ex(ES))",
    "cLCB92": "SERV :D + (L) + (1.0)(RSA_Ex(RS)-RSA_Ex(ES))",
    "cLCB93": "SERV :D + (L) + (1.0)(RSA_Ey(RS)+RSA_Ey(ES))",
    "cLCB94": "SERV :D + (L) + (1.0)(RSA_Ey(RS)-RSA_Ey(ES))",
    "cLCB103": "SERV :D + (L) - (1.0)(RSA_Ex(RS)+RSA_Ex(ES))",
    "cLCB104": "SERV :D + (L) - (1.0)(RSA_Ex(RS)-RSA_Ex(ES))",
    "cLCB105": "SERV :D + (L) - (1.0)(RSA_Ey(RS)+RSA_Ey(ES))",
    "cLCB106": "SERV :D + (L) - (1.0)(RSA_Ey(RS)-RSA_Ey(ES))",
    "cLCB115": "SERV :D + (1.0)(RSA_Ex(RS)+RSA_Ex(ES))",
    "cLCB116": "SERV :D + (1.0)(RSA_Ex(RS)-RSA_Ex(ES))",
    "cLCB117": "SERV :D + (1.0)(RSA_Ey(RS)+RSA_Ey(ES))",
    "cLCB118": "SERV :D + (1.0)(RSA_Ey(RS)-RSA_Ey(ES))",
    "cLCB127": "SERV :D - (1.0)(RSA_Ex(RS)+RSA_Ex(ES))",
    "cLCB128": "SERV :D - (1.0)(RSA_Ex(RS)-RSA_Ex(ES))",
    "cLCB129": "SERV :D - (1.0)(RSA_Ey(RS)+RSA_Ey(ES))",
    "cLCB130": "SERV :D - (1.0)(RSA_Ey(RS)-RSA_Ey(ES))",
}

# ===================== ANALYZER CLASS =====================
class PileAnalyzer:
    """Pile foundation analysis using proper structural engineering theory"""
    
    def __init__(self, pile_diameter=0.6, pile_capacity=120):
        self.pile_diameter = pile_diameter
        self.pile_capacity = pile_capacity
        self.min_spacing = 2.5 * pile_diameter
    
    def get_foundation_config(self, foundation_id):
        """Get foundation configuration from default or custom"""
        all_foundations = {**DEFAULT_FOUNDATIONS, **st.session_state.custom_foundations}
        config = all_foundations.get(foundation_id, None)
        
        if config is None:
            st.warning(f"Foundation {foundation_id} not found. Using default F4.")
            config = DEFAULT_FOUNDATIONS.get('F4')
        
        return config
    
    def calculate_properties(self, foundation_id):
        """Calculate section properties for a foundation"""
        config = self.get_foundation_config(foundation_id)
        if not config:
            return None
        
        # Get coordinates
        coords = np.array(config.get('coords', []))
        if len(coords) == 0:
            st.error(f"No coordinates found for foundation {foundation_id}")
            return None
        
        # Coordinates are already properly scaled in the foundation definitions
        n_piles = len(coords)
        
        # Calculate centroid
        centroid_x = np.mean(coords[:, 0])
        centroid_y = np.mean(coords[:, 1])
        
        # Adjust to centroid
        x_coords = coords[:, 0] - centroid_x
        y_coords = coords[:, 1] - centroid_y
        
        # Calculate moment of inertia
        Ixx = np.sum(y_coords**2)
        Iyy = np.sum(x_coords**2)
        
        # Maximum distances (c values)
        cx_max = np.max(np.abs(x_coords)) if len(x_coords) > 0 else 1.0
        cy_max = np.max(np.abs(y_coords)) if len(y_coords) > 0 else 1.0
        
        # Prevent division by zero
        cx_max = max(cx_max, 0.1)
        cy_max = max(cy_max, 0.1)
        
        # Section modulus
        Zx = Ixx / cy_max if cy_max > 0 else 1
        Zy = Iyy / cx_max if cx_max > 0 else 1
        
        # Store in session state for display
        st.session_state.foundation_properties[foundation_id] = {
            'n_piles': n_piles,
            'Ixx': Ixx,
            'Iyy': Iyy,
            'cx_max': cx_max,
            'cy_max': cy_max,
            'Zx': Zx,
            'Zy': Zy,
            'coords': coords.tolist(),
            'centroid': (centroid_x, centroid_y)
        }
        
        return {
            'n_piles': n_piles,
            'Ixx': Ixx,
            'Iyy': Iyy,
            'Zx': Zx,
            'Zy': Zy,
            'cx_max': cx_max,
            'cy_max': cy_max,
            'coords': coords.tolist()
        }
    
    def calculate_pile_loads(self, Fz, Mx, My, foundation_id):
        """Calculate maximum pile load using P = P/A + Mx*c/Ix + My*c/Iy"""
        try:
            props = self.calculate_properties(foundation_id)
            if not props:
                return None
            
            config = self.get_foundation_config(foundation_id)
            
            # Calculate stress components
            axial_stress = Fz / props['n_piles'] if props['n_piles'] > 0 else 0
            stress_from_My = abs(My) * props['cx_max'] / props['Iyy'] if props['Iyy'] > 0 else 0
            stress_from_Mx = abs(Mx) * props['cy_max'] / props['Ixx'] if props['Ixx'] > 0 else 0
            
            # Maximum pile load
            max_pile_load = axial_stress + stress_from_My + stress_from_Mx
            min_pile_load = axial_stress - stress_from_My - stress_from_Mx
            
            # Utilization ratio
            utilization = max_pile_load / self.pile_capacity if self.pile_capacity > 0 else 0
            
            # Category
            if utilization > 1.0:
                category = "Over-Capacity"
            elif utilization > 0.95:
                category = "Near-Capacity"
            elif utilization > 0.80:
                category = "Optimal"
            elif utilization > 0.60:
                category = "Conservative"
            else:
                category = "Over-Conservative"
            
            return {
                'foundation_id': foundation_id,
                'foundation_name': config.get('name', 'Unknown'),
                'n_piles': props['n_piles'],
                'axial_stress': axial_stress,
                'moment_stress_mx': stress_from_Mx,
                'moment_stress_my': stress_from_My,
                'max_pile_load': max_pile_load,
                'min_pile_load': min_pile_load,
                'utilization_ratio': utilization,
                'is_safe': utilization <= 1.0,
                'has_tension': min_pile_load < 0,
                'category': category,
                'color': config.get('color', '#808080'),
                'Ixx': props['Ixx'],
                'Iyy': props['Iyy'],
                'cx_max': props['cx_max'],
                'cy_max': props['cy_max'],
                'Zx': props['Zx'],
                'Zy': props['Zy']
            }
        except Exception as e:
            st.error(f"Error calculating loads for {foundation_id}: {str(e)}")
            return None
    
    def optimize_foundation(self, Fz, Mx, My, target_utilization=0.85, allowed_foundations=None):
        """Find optimal foundation for given loads"""
        if allowed_foundations is None:
            allowed_foundations = list(DEFAULT_FOUNDATIONS.keys())
        
        if len(allowed_foundations) == 0:
            allowed_foundations = ['F4', 'F6', 'F9']  # Default fallback
        
        results = []
        for foundation_id in allowed_foundations:
            result = self.calculate_pile_loads(Fz, Mx, My, foundation_id)
            if result:
                result['target_diff'] = abs(result['utilization_ratio'] - target_utilization)
                results.append(result)
        
        if not results:
            # Try with default F4 as last resort
            result = self.calculate_pile_loads(Fz, Mx, My, 'F4')
            if result:
                return result
            return None
        
        # Sort by target difference
        results.sort(key=lambda x: x['target_diff'])
        
        # Find best safe design
        for result in results:
            if result['is_safe']:
                return result
        
        # If no safe design, return lowest utilization
        return min(results, key=lambda x: x['utilization_ratio'])

# ===================== HELPER FUNCTIONS =====================
def format_load_combination(load_case):
    """Convert load case code to readable combination"""
    if isinstance(load_case, str):
        # Check if it matches a known pattern
        for pattern, combination in LOAD_COMBINATIONS.items():
            if pattern.lower() in load_case.lower():
                return combination
        
        # Try to extract number from load case
        match = re.search(r'(\d+)', load_case)
        if match:
            case_num = f"cLCB{match.group(1)}"
            if case_num in LOAD_COMBINATIONS:
                return LOAD_COMBINATIONS[case_num]
    
    return load_case  # Return original if no match

def load_csv_file(uploaded_file):
    """Load CSV file with multiple encoding attempts"""
    if uploaded_file is None:
        return None, "No file uploaded"
    
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            df = pd.read_csv(uploaded_file, encoding=encoding)
            uploaded_file.seek(0)  # Reset file pointer
            return df, f"Successfully loaded with {encoding} encoding"
        except:
            uploaded_file.seek(0)
            continue
    
    return None, "Could not decode file with any standard encoding"

def check_tension_nodes(df):
    """Check for nodes with tensile forces (Fz < 0)"""
    tension_nodes = []
    
    if 'FZ (tonf)' in df.columns:
        tension_df = df[df['FZ (tonf)'] < 0]
        fz_col = 'FZ (tonf)'
    elif 'Fz' in df.columns:
        tension_df = df[df['Fz'] < 0]
        fz_col = 'Fz'
    else:
        return tension_nodes
    
    for _, row in tension_df.iterrows():
        tension_nodes.append({
            'Node': row['Node'],
            'Fz': row[fz_col],
            'Load_Case': row.get('Load Combination', row.get('Load Case', 'Unknown'))
        })
    
    return tension_nodes

def process_analysis(df, selected_nodes, analyzer, target_utilization=0.85):
    """Process pile analysis for selected nodes"""
    if df is None:
        return None
    
    # Filter for selected nodes
    df_filtered = df[df['Node'].isin(selected_nodes)]
    
    if len(df_filtered) == 0:
        return None
    
    # Check for tension nodes
    tension_nodes = check_tension_nodes(df_filtered)
    st.session_state.tension_nodes = tension_nodes
    
    results = []
    
    for idx, row in df_filtered.iterrows():
        # Extract loads - preserve original values
        Fz_raw = row.get('FZ (tonf)', row.get('Fz', 0))
        Mx_raw = row.get('MX (tonf·m)', row.get('Mx', 0))  # Original signed value
        My_raw = row.get('MY (tonf·m)', row.get('My', 0))  # Original signed value
        
        # Use absolute values for calculation
        Fz = abs(Fz_raw)
        Mx = abs(Mx_raw)
        My = abs(My_raw)
        
        # Get load combination
        load_case_raw = row.get('Load Combination', row.get('Load Case', f'LC_{idx}'))
        load_combination = format_load_combination(load_case_raw)
        
        # Check for manual assignment
        node = row['Node']
        if node in st.session_state.node_assignments:
            foundation_id = st.session_state.node_assignments[node]
            result = analyzer.calculate_pile_loads(Fz, Mx, My, foundation_id)
            result['assignment_method'] = 'Manual'
        else:
            # Automatic optimization
            allowed = st.session_state.allowed_foundations if st.session_state.allowed_foundations else None
            result = analyzer.optimize_foundation(Fz, Mx, My, target_utilization, allowed)
            result['assignment_method'] = 'Optimized'
        
        if result:
            # Add node information
            result['Node'] = node
            result['X'] = row.get('X', 0)
            result['Y'] = row.get('Y', 0)
            result['Z'] = row.get('Z', 0)
            result['Fz'] = Fz
            result['Fz_raw'] = Fz_raw  # Store raw value for tension check
            result['Mx'] = Mx_raw      # Store ORIGINAL signed value for display
            result['My'] = My_raw      # Store ORIGINAL signed value for display
            result['Mx_abs'] = Mx      # Store absolute value used in calculation
            result['My_abs'] = My      # Store absolute value used in calculation
            result['Load_Case'] = load_case_raw
            result['Load_Combination'] = load_combination
            result['Has_Tension'] = Fz_raw < 0
            
            results.append(result)
    
    return pd.DataFrame(results)

def create_foundation_properties_display(foundation_id):
    """Create a detailed display of foundation properties"""
    if foundation_id in st.session_state.foundation_properties:
        props = st.session_state.foundation_properties[foundation_id]
        
        # Create properties table
        properties_df = pd.DataFrame([
            {'Property': 'Number of Piles (n)', 'Value': f"{props['n_piles']}", 'Unit': 'piles'},
            {'Property': 'Moment of Inertia Ixx', 'Value': f"{props['Ixx']:.3f}", 'Unit': 'm²'},
            {'Property': 'Moment of Inertia Iyy', 'Value': f"{props['Iyy']:.3f}", 'Unit': 'm²'},
            {'Property': 'Distance cx (max)', 'Value': f"{props['cx_max']:.3f}", 'Unit': 'm'},
            {'Property': 'Distance cy (max)', 'Value': f"{props['cy_max']:.3f}", 'Unit': 'm'},
            {'Property': 'Section Modulus Zx', 'Value': f"{props['Zx']:.3f}", 'Unit': 'm²'},
            {'Property': 'Section Modulus Zy', 'Value': f"{props['Zy']:.3f}", 'Unit': 'm²'},
            {'Property': 'Centroid Location', 'Value': f"({props['centroid'][0]:.2f}, {props['centroid'][1]:.2f})", 'Unit': 'm'}
        ])
        
        return properties_df
    return None

def create_custom_foundation_ui():
    """UI for creating custom foundations"""
    st.markdown("### 🛠️ Custom Foundation Designer")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Foundation Details")
        
        custom_id = st.text_input("Foundation ID", value="CUSTOM1", key="custom_id")
        custom_name = st.text_input("Foundation Name", value="Custom Foundation", key="custom_name")
        custom_color = st.color_picker("Color", value="#FF00FF", key="custom_color")
        
        st.markdown("#### Pile Coordinates")
        
        input_method = st.radio("Input Method", ["Table", "Text", "Template"], key="input_method")
        
        if input_method == "Table":
            n_piles = st.number_input("Number of Piles", 2, 50, 4, key="n_piles")
            
            # Create coordinate table
            coords_data = []
            for i in range(n_piles):
                coords_data.append({'Pile': f'P{i+1}', 'X': 0.0, 'Y': 0.0})
            
            df_coords = pd.DataFrame(coords_data)
            edited_df = st.data_editor(df_coords, use_container_width=True, key="coords_table")
            
            coordinates = [(row['X'], row['Y']) for _, row in edited_df.iterrows()]
            
        elif input_method == "Text":
            coord_text = st.text_area(
                "Enter coordinates (x,y per line)",
                value="0,2\n1.73,1\n1.73,-1\n0,-2\n-1.73,-1\n-1.73,1",
                height=200,
                key="coord_text"
            )
            
            coordinates = []
            try:
                for line in coord_text.strip().split('\n'):
                    if line.strip():
                        x, y = map(float, line.split(','))
                        coordinates.append((x, y))
            except:
                st.error("Invalid coordinate format")
                coordinates = []
        
        else:  # Template
            template = st.selectbox("Select Template", 
                                   ["Square", "Circle", "Rectangle", "Hexagon"],
                                   key="template")
            
            if template == "Square":
                n = st.slider("Grid Size", 2, 5, 3, key="square_n")
                coordinates = []
                for i in range(n):
                    for j in range(n):
                        x = (j - (n-1)/2) * spacing
                        y = (i - (n-1)/2) * spacing
                        coordinates.append((x, y))
            
            elif template == "Circle":
                n = st.slider("Number of Piles", 3, 20, 8, key="circle_n")
                radius = st.slider("Radius", 1.0, 5.0, 2.0, 0.1, key="circle_radius")
                center = st.checkbox("Include Center", key="circle_center")             
                coordinates = []
                if center:
                    coordinates.append((0, 0))
                for i in range(n):
                    angle = 2 * np.pi * i / n
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    coordinates.append((round(x, 2), round(y, 2)))
            
            elif template == "Rectangle":
                rows = st.slider("Rows", 2, 6, 3, key="rect_rows")
                cols = st.slider("Columns", 2, 6, 3, key="rect_cols")
                
                coordinates = []
                for i in range(rows):
                    for j in range(cols):
                        x = (j - (cols-1)/2) * spacing
                        y = (i - (rows-1)/2) * spacing
                        coordinates.append((x, y))
            
            else:  # Hexagon
                coordinates = [(0, 2), (1.73, 1), (1.73, -1), 
                              (0, -2), (-1.73, -1), (-1.73, 1)]
    
    with col2:
        st.markdown("#### Preview")
        
        if coordinates:
            # Create visualization
            coords_array = np.array(coordinates)
            
            fig = go.Figure()
            
            # Add piles
            fig.add_trace(go.Scatter(
                x=coords_array[:, 0],
                y=coords_array[:, 1],
                mode='markers+text',
                marker=dict(size=25, color=custom_color, 
                           symbol='circle', line=dict(color='darkblue', width=2)),
                text=[f'P{i+1}' for i in range(len(coords_array))],
                textposition='top center',
                name='Piles'
            ))
            
            # Add centroid
            centroid_x = np.mean(coords_array[:, 0])
            centroid_y = np.mean(coords_array[:, 1])
            
            fig.add_trace(go.Scatter(
                x=[centroid_x],
                y=[centroid_y],
                mode='markers',
                marker=dict(size=10, color='red', symbol='x'),
                name='Centroid'
            ))
            
            # Add grid
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            fig.update_layout(
                title=f'{custom_id}: {custom_name}',
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                height=400,
                showlegend=True,
                xaxis=dict(scaleanchor="y"),
                yaxis=dict(scaleanchor="x")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate properties
            st.markdown("#### Properties")
            
            x_coords = coords_array[:, 0] - centroid_x
            y_coords = coords_array[:, 1] - centroid_y
            
            Ixx = np.sum(y_coords**2)
            Iyy = np.sum(x_coords**2)
            
            col1_prop, col2_prop = st.columns(2)
            with col1_prop:
                st.metric("Number of Piles", len(coordinates))
                st.metric("Ixx (m²)", f"{Ixx:.3f}")
            
            with col2_prop:
                st.metric("Iyy (m²)", f"{Iyy:.3f}")
                cx_max = np.max(np.abs(x_coords)) if len(x_coords) > 0 else 0
                cy_max = np.max(np.abs(y_coords)) if len(y_coords) > 0 else 0
                st.metric("Max Distance (m)", f"{max(cx_max, cy_max):.2f}")
            
            # Save button
            if st.button("💾 Save Custom Foundation", type="primary", use_container_width=True):
                custom_config = {
                    'name': custom_name,
                    'piles': len(coordinates),
                    'color': custom_color,
                    'coords': coordinates
                }
                st.session_state.custom_foundations[custom_id] = custom_config
                st.success(f"✅ {custom_id} saved successfully!")
                st.rerun()

def create_visualizations(results_df):
    """Create analysis visualizations"""
    if results_df is None or len(results_df) == 0:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Foundation Distribution', 'Utilization by Type',
                       'Load vs Utilization', 'Site Plan'),
        specs=[[{'type': 'pie'}, {'type': 'box'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # 1. Pie chart
    foundation_counts = results_df['foundation_id'].value_counts()
    colors = [results_df[results_df['foundation_id'] == f]['color'].iloc[0] 
             for f in foundation_counts.index]
    
    fig.add_trace(
        go.Pie(labels=foundation_counts.index, 
               values=foundation_counts.values,
               marker=dict(colors=colors)),
        row=1, col=1
    )
    
    # 2. Box plot
    for foundation in results_df['foundation_id'].unique():
        data = results_df[results_df['foundation_id'] == foundation]
        fig.add_trace(
            go.Box(y=data['utilization_ratio'], name=foundation,
                  marker_color=data['color'].iloc[0]),
            row=1, col=2
        )
    
    # 3. Load vs Utilization
    fig.add_trace(
        go.Scatter(x=results_df['Fz'], y=results_df['utilization_ratio'],
                  mode='markers', marker=dict(size=10, color=results_df['n_piles'],
                                             colorscale='Viridis'),
                  text=results_df['Node'], name='Nodes'),
        row=2, col=1
    )
    
    # 4. Site plan (if coordinates exist)
    if 'X' in results_df.columns and 'Y' in results_df.columns:
        for foundation in results_df['foundation_id'].unique():
            data = results_df[results_df['foundation_id'] == foundation]
            fig.add_trace(
                go.Scatter(x=data['X'], y=data['Y'],
                          mode='markers+text',
                          marker=dict(size=15, color=data['color'].iloc[0]),
                          text=data['Node'], textposition='top center',
                          name=foundation),
                row=2, col=2
            )
    
    fig.update_layout(height=800, showlegend=True, title_text="Pile Foundation Analysis Results")
    
    return fig

# ===================== SIDEBAR =====================
st.sidebar.title("📋 Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])

# Pile parameters
st.sidebar.subheader("🔧 Pile Parameters")
pile_diameter = st.sidebar.number_input("Pile Diameter (m)", 0.3, 2.0, 0.6, 0.1)
pile_capacity = st.sidebar.number_input("Pile Capacity (tonf)", 30, 500, 120, 10)
target_utilization = st.sidebar.slider("Target Utilization", 0.7, 0.95, 0.85, 0.05)

# Node selection
st.sidebar.subheader("🎯 Node Selection")
use_default = st.sidebar.checkbox("Use Default Nodes", value=True)

if use_default:
    selected_nodes = DEFAULT_NODES
else:
    node_input = st.sidebar.text_area("Enter nodes (comma-separated)", 
                                      value=",".join(map(str, DEFAULT_NODES[:5])))
    try:
        selected_nodes = [int(x.strip()) for x in node_input.split(",")]
    except:
        selected_nodes = DEFAULT_NODES[:5]
        st.sidebar.error("Invalid node format")

st.sidebar.info(f"Selected {len(selected_nodes)} nodes")

# ===================== MAIN CONTENT - TABS =====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📚 Theory & Setup",
    "🛠️ Custom Foundations",
    "📊 Data & Analysis",
    "🔍 Foundation Properties",
    "📈 Results & Visualization",
    "🗺️ Site Plan & Foundation Map",  # NEW TAB
    "💾 Export"
])

with tab1:
    st.markdown('<h2 class="section-header">📚 Theory & Foundation Setup</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Pile Group Design Formula
        
        <div class="formula-box">
        <strong>Maximum Pile Load:</strong><br>
        P<sub>max</sub> = P/n + |M<sub>x</sub>|·c<sub>y</sub>/I<sub>xx</sub> + |M<sub>y</sub>|·c<sub>x</sub>/I<sub>yy</sub>
        <br><br>
        Where:<br>
        • P = Vertical load (tonf)<br>
        • n = Number of piles<br>
        • M<sub>x</sub>, M<sub>y</sub> = Moments (tonf·m)<br>
        • c<sub>x</sub>, c<sub>y</sub> = Distance to extreme fiber (m)<br>
        • I<sub>xx</sub>, I<sub>yy</sub> = Moment of inertia (m²)<br>
        • Z<sub>x</sub> = I<sub>xx</sub>/c<sub>y</sub> (Section modulus)<br>
        • Z<sub>y</sub> = I<sub>yy</sub>/c<sub>x</sub> (Section modulus)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Foundation Assignment")
        
        assignment_method = st.radio(
            "Assignment Method",
            ["Automatic Optimization", "Manual Assignment", "Selected Foundations Only"]
        )
        
        if assignment_method == "Automatic Optimization":
            st.info("System will automatically select optimal foundation for each node")
            
        elif assignment_method == "Manual Assignment":
            st.warning("Manually assign foundations to specific nodes")
            
            manual_nodes = st.multiselect("Select Nodes", selected_nodes)
            all_foundations = {**DEFAULT_FOUNDATIONS, **st.session_state.custom_foundations}
            manual_foundation = st.selectbox("Assign Foundation", list(all_foundations.keys()))
            
            if st.button("Apply Assignment"):
                for node in manual_nodes:
                    st.session_state.node_assignments[node] = manual_foundation
                st.success(f"Assigned {manual_foundation} to {len(manual_nodes)} nodes")
        
        else:  # Selected Foundations Only
            all_foundations = {**DEFAULT_FOUNDATIONS, **st.session_state.custom_foundations}
            selected_foundations = st.multiselect(
                "Select Allowed Foundations",
                list(all_foundations.keys()),
                default=list(DEFAULT_FOUNDATIONS.keys())[:5]
            )
            st.session_state.allowed_foundations = selected_foundations
    
    with col2:
        st.markdown("### Available Foundations")
        
        # Default foundations
        st.markdown("**Default Foundations:**")
        for fid, config in DEFAULT_FOUNDATIONS.items():
            st.markdown(f"""
            <div class="foundation-card" style="border-left-color: {config['color']}">
            <strong>{fid}</strong>: {config['name']}<br>
            <small>{config['piles']} piles</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Custom foundations
        if st.session_state.custom_foundations:
            st.markdown("**Custom Foundations:**")
            for fid, config in st.session_state.custom_foundations.items():
                st.markdown(f"""
                <div class="foundation-card" style="border-left-color: {config['color']}">
                <strong>{fid}</strong>: {config['name']}<br>
                <small>{config['piles']} piles</small>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown('<h2 class="section-header">🛠️ Custom Foundation Designer</h2>', unsafe_allow_html=True)
    
    create_custom_foundation_ui()
    
    # Display saved custom foundations
    if st.session_state.custom_foundations:
        st.markdown("### 📚 Saved Custom Foundations")
        
        for fid, config in st.session_state.custom_foundations.items():
            with st.expander(f"{fid}: {config['name']} ({config['piles']} piles)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Piles:** {config['piles']}")
                    st.write(f"**Color:** {config['color']}")
                with col2:
                    if st.button(f"Delete {fid}", key=f"del_{fid}"):
                        del st.session_state.custom_foundations[fid]
                        st.rerun()

with tab3:
    st.markdown('<h2 class="section-header">📊 Data Input & Analysis</h2>', unsafe_allow_html=True)
    
    if uploaded_file:
        df, message = load_csv_file(uploaded_file)
        
        if df is not None:
            st.success(message)
            
            # Data overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Unique Nodes", df['Node'].nunique() if 'Node' in df.columns else 0)
            with col4:
                total_foundations = len(DEFAULT_FOUNDATIONS) + len(st.session_state.custom_foundations)
                st.metric("Available Foundations", total_foundations)
            
            # Preview data
            with st.expander("Preview Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Check for tension nodes before analysis
            tension_check = check_tension_nodes(df[df['Node'].isin(selected_nodes)])
            if tension_check:
                st.markdown("""
                <div class="tension-warning">
                <h4>⚠️ TENSION WARNING!</h4>
                <p>The following nodes have negative Fz values (tension):</p>
                </div>
                """, unsafe_allow_html=True)
                
                tension_df = pd.DataFrame(tension_check)
                st.dataframe(tension_df, use_container_width=True)
                
                st.warning("Proceeding with analysis using absolute values of Fz. Please review foundation design for tension conditions.")
            
            # Run analysis
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    analyzer = PileAnalyzer(pile_diameter, pile_capacity)
                    results = process_analysis(df, selected_nodes, analyzer, target_utilization)
                    
                    if results is not None and len(results) > 0:
                        st.session_state.final_results = results
                        st.success(f"✅ Analysis completed for {len(results)} nodes!")
                        st.balloons()
                    else:
                        st.error("No valid results generated")
        else:
            st.error(message)
    else:
        st.info("Please upload a CSV file to begin analysis")
        
        # Show expected format
        st.subheader("Expected Data Format")
        example_df = pd.DataFrame({
            'Node': [789, 790, 791],
            'X': [0, 10, 20],
            'Y': [0, 0, 0],
            'Load Combination': ['cLCB5', 'cLCB6', 'cLCB9'],
            'FZ (tonf)': [400, -50, 450],  # Include negative value for example
            'MX (tonf·m)': [80, 70, 90],
            'MY (tonf·m)': [60, 50, 70]
        })
        st.dataframe(example_df, use_container_width=True)

with tab4:
    st.markdown('<h2 class="section-header">🔍 Foundation Properties Summary</h2>', unsafe_allow_html=True)
    
    if st.session_state.final_results is not None:
        results = st.session_state.final_results
        
        # Select a foundation to display properties
        unique_foundations = results['foundation_id'].unique()
        selected_foundation = st.selectbox("Select Foundation to View Properties", unique_foundations)
        
        if selected_foundation:
            # Display formula
            st.markdown("""
            <div class="formula-box">
            <strong>Stress Calculation Formula:</strong><br>
            σ = P/A ± M<sub>x</sub>·c<sub>y</sub>/I<sub>xx</sub> ± M<sub>y</sub>·c<sub>x</sub>/I<sub>yy</sub>
            </div>
            """, unsafe_allow_html=True)
            
            # Display properties
            props_df = create_foundation_properties_display(selected_foundation)
            
            if props_df is not None:
                st.markdown("### Foundation Geometric Properties")
                st.markdown('<div class="properties-table">', unsafe_allow_html=True)
                st.dataframe(props_df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show example calculation
                example_node = results[results['foundation_id'] == selected_foundation].iloc[0]
                
                st.markdown("### Example Calculation (Node {})".format(example_node['Node']))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Input Loads:**")
                    st.write(f"• P (Fz) = {example_node['Fz']:.2f} tonf")
                    st.write(f"• Mx = {example_node['Mx']:.2f} tonf·m")
                    st.write(f"• My = {example_node['My']:.2f} tonf·m")
                    if 'Load_Combination' in example_node:
                        st.write(f"• Load Combination: **{example_node['Load_Combination']}**")
                    else:
                        st.write(f"• Load Case: **{example_node.get('Load_Case', 'N/A')}**")
                
                with col2:
                    st.markdown("**Stress Components:**")
                    st.write(f"• P/n = {example_node['axial_stress']:.2f} tonf")
                    st.write(f"• Mx·cy/Ixx = {example_node['moment_stress_mx']:.2f} tonf")
                    st.write(f"• My·cx/Iyy = {example_node['moment_stress_my']:.2f} tonf")
                    st.write(f"• **Total = {example_node['max_pile_load']:.2f} tonf**")
                
                # Utilization
                st.markdown(f"""
                <div class="critical-load">
                <strong>Utilization Ratio:</strong> {example_node['utilization_ratio']:.2%}<br>
                <strong>Category:</strong> {example_node['category']}<br>
                <strong>Safety Status:</strong> {'✅ Safe' if example_node['is_safe'] else '❌ Over-capacity'}
                </div>
                """, unsafe_allow_html=True)
                
                # Visual representation
                if selected_foundation in st.session_state.foundation_properties:
                    props = st.session_state.foundation_properties[selected_foundation]
                    coords = np.array(props['coords'])
                    
                    fig = go.Figure()
                    
                    # Add piles
                    fig.add_trace(go.Scatter(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        mode='markers+text',
                        marker=dict(size=30, color='blue', symbol='circle', 
                                  line=dict(color='darkblue', width=2)),
                        text=[f'P{i+1}' for i in range(len(coords))],
                        textposition='top center',
                        name='Piles'
                    ))
                    
                    # Add centroid
                    fig.add_trace(go.Scatter(
                        x=[props['centroid'][0]],
                        y=[props['centroid'][1]],
                        mode='markers',
                        marker=dict(size=15, color='red', symbol='x'),
                        name='Centroid'
                    ))
                    
                    # Add extreme fibers
                    fig.add_shape(type="rect",
                        x0=min(coords[:, 0]), y0=min(coords[:, 1]),
                        x1=max(coords[:, 0]), y1=max(coords[:, 1]),
                        line=dict(color="gray", dash="dash")
                    )
                    
                    fig.update_layout(
                        title=f"Foundation Layout: {selected_foundation}",
                        xaxis_title="X (m)",
                        yaxis_title="Y (m)",
                        height=500,
                        showlegend=True,
                        xaxis=dict(scaleanchor="y", showgrid=True),
                        yaxis=dict(scaleanchor="x", showgrid=True)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run analysis to view foundation properties")

with tab5:
    st.markdown('<h2 class="section-header">📈 Results & Visualization</h2>', unsafe_allow_html=True)
    
    if st.session_state.final_results is not None:
        results = st.session_state.final_results
        
        # Display tension warning if applicable
        if st.session_state.tension_nodes:
            st.markdown("""
            <div class="tension-warning">
            <h4>⚠️ TENSION NODES IN RESULTS</h4>
            <p>Nodes with tension forces are highlighted in the table below.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes Analyzed", len(results))
        with col2:
            st.metric("Avg Utilization", f"{results['utilization_ratio'].mean():.1%}")
        with col3:
            safe_count = len(results[results['is_safe']])
            st.metric("Safe Designs", f"{safe_count}/{len(results)}")
        with col4:
            total_piles = results['n_piles'].sum()
            st.metric("Total Piles", int(total_piles))
        
        # Results table with critical load combination
        st.subheader("Analysis Results - Critical Load Combinations")
        
        # Prepare display dataframe
        display_results = results.copy()
        
        # Check if Load_Combination column exists
        if 'Load_Combination' in display_results.columns:
            display_results['Display_Load'] = display_results['Load_Combination']
        else:
            display_results['Display_Load'] = display_results.get('Load_Case', 'N/A')
        
        display_results['Tension_Flag'] = display_results['Has_Tension'].apply(lambda x: '⚠️ TENSION' if x else '✓')
        
        # Updated columns to include Mx and My
        display_cols = ['Node', 'foundation_id', 'n_piles', 'Display_Load',
                       'Fz', 'Mx', 'My', 'Mx_abs', 'My_abs', 
                       'max_pile_load', 'utilization_ratio', 
                       'category', 'is_safe', 'Tension_Flag']
        
        # Format the dataframe for better readability
        display_results_formatted = display_results[display_cols].copy()
        display_results_formatted['Fz'] = display_results_formatted['Fz'].apply(lambda x: f"{x:.2f}")
        display_results_formatted['Mx'] = display_results_formatted['Mx'].apply(lambda x: f"{x:.2f}")
        display_results_formatted['My'] = display_results_formatted['My'].apply(lambda x: f"{x:.2f}")
        display_results_formatted['max_pile_load'] = display_results_formatted['max_pile_load'].apply(lambda x: f"{x:.2f}")
        display_results_formatted['utilization_ratio'] = display_results_formatted['utilization_ratio'].apply(lambda x: f"{x:.1%}")
        
        # Rename columns for display
        display_results_formatted.columns = ['Node', 'Foundation', 'Piles', 'Load Combination',
                                             'Fz (tonf)', 'Mx (tonf·m)', 'My (tonf·m)', 
                                             'Mx Used (tonf·m)', 'My Used (tonf·m)',
                                             'Max Pile Load (tonf)', 'Utilization', 
                                             'Category', 'Safe', 'Status']
        
        # Style the dataframe to highlight tension nodes
        def highlight_tension(row):
            idx = row.name
            has_tension = display_results.loc[idx, 'Has_Tension']
            is_safe = display_results.loc[idx, 'is_safe']
            category = display_results.loc[idx, 'category']
            
            if has_tension:
                return ['background-color: #ffe6e6'] * len(row)
            elif not is_safe:
                return ['background-color: #ffcccc'] * len(row)
            elif category == 'Optimal':
                return ['background-color: #ccffcc'] * len(row)
            else:
                return [''] * len(row)
        
        styled_df = display_results_formatted.style.apply(highlight_tension, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Legend
        st.markdown("""
        **Legend:**
        - 🟩 Green: Optimal utilization (80-95%)
        - 🟥 Light Red: Over-capacity design
        - 🟨 Pink: Tension condition detected
        """)
        
        # Visualizations
        st.subheader("Visualizations")
        fig = create_visualizations(results)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No results to display. Please run analysis first.")

with tab6:
    st.markdown('<h2 class="section-header">🗺️ Site Plan & Foundation Map</h2>', unsafe_allow_html=True)
    
    if st.session_state.final_results is not None:
        results = st.session_state.final_results
        
        # Check if coordinate data exists
        if 'X' not in results.columns or 'Y' not in results.columns:
            st.warning("⚠️ Coordinate data (X, Y) not found in results. Cannot generate site plan.")
        else:
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                show_labels = st.checkbox("Show Node Labels", value=True)
            with col2:
                show_foundation_icons = st.checkbox("Show Foundation Type Icons", value=True)
            with col3:
                marker_size = st.slider("Marker Size", 10, 50, 25)
            
            # Create main site plan
            fig = go.Figure()
            
            # Group by foundation type for better visualization
            for foundation_id in results['foundation_id'].unique():
                foundation_data = results[results['foundation_id'] == foundation_id]
                
                # Get foundation configuration
                all_foundations = {**DEFAULT_FOUNDATIONS, **st.session_state.custom_foundations}
                config = all_foundations.get(foundation_id, {})
                
                # Plot nodes with this foundation type
                fig.add_trace(go.Scatter(
                    x=foundation_data['X'],
                    y=foundation_data['Y'],
                    mode='markers+text' if show_labels else 'markers',
                    name=f"{foundation_id} ({config.get('name', 'Unknown')})",
                    marker=dict(
                        size=marker_size,
                        color=config.get('color', '#808080'),
                        line=dict(color='white', width=2),
                        symbol='circle'
                    ),
                    text=foundation_data['Node'].astype(str),
                    textposition='top center',
                    textfont=dict(size=10, color='black'),
                    hovertemplate='<b>Node %{text}</b><br>' +
                                  'Foundation: ' + foundation_id + '<br>' +
                                  'Piles: %{customdata[0]}<br>' +
                                  'X: %{x:.2f} m<br>' +
                                  'Y: %{y:.2f} m<br>' +
                                  'Utilization: %{customdata[1]:.1%}<br>' +
                                  '<extra></extra>',
                    customdata=foundation_data[['n_piles', 'utilization_ratio']].values
                ))
            
            # Update layout
            fig.update_layout(
                title='Foundation Distribution Site Plan',
                xaxis_title='X Coordinate (m)',
                yaxis_title='Y Coordinate (m)',
                height=700,
                hovermode='closest',
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray', scaleanchor="x", scaleratio=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics by location
            st.markdown("### Foundation Distribution Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Foundation type counts
                foundation_summary = results.groupby('foundation_id').agg({
                    'Node': 'count',
                    'n_piles': 'first',
                    'utilization_ratio': 'mean',
                    'foundation_name': 'first'
                }).reset_index()
                
                foundation_summary.columns = ['Foundation ID', 'Count', 'Piles/Foundation', 
                                             'Avg Utilization', 'Name']
                foundation_summary['Total Piles'] = foundation_summary['Count'] * foundation_summary['Piles/Foundation']
                foundation_summary['Avg Utilization'] = foundation_summary['Avg Utilization'].apply(lambda x: f"{x:.1%}")
                
                st.markdown("#### Foundation Type Distribution")
                st.dataframe(foundation_summary, use_container_width=True, hide_index=True)
            
            with col2:
                # Create pie chart of foundation distribution
                fig_pie = go.Figure(data=[go.Pie(
                    labels=foundation_summary['Foundation ID'],
                    values=foundation_summary['Count'],
                    hole=0.3,
                    marker=dict(colors=[
                        results[results['foundation_id'] == fid]['color'].iloc[0] 
                        for fid in foundation_summary['Foundation ID']
                    ])
                )])
                
                fig_pie.update_layout(
                    title='Foundation Type Distribution',
                    height=400
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Detailed foundation layout viewer
            st.markdown("### Detailed Foundation Layout Viewer")
            
            selected_node = st.selectbox(
                "Select Node to View Foundation Details",
                options=results['Node'].unique(),
                format_func=lambda x: f"Node {x}"
            )
            
            if selected_node:
                node_data = results[results['Node'] == selected_node].iloc[0]
                foundation_id = node_data['foundation_id']
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"#### Node {selected_node} Details")
                    st.markdown(f"""
                    - **Foundation Type:** {foundation_id}
                    - **Foundation Name:** {node_data['foundation_name']}
                    - **Number of Piles:** {int(node_data['n_piles'])}
                    - **Location:** ({node_data['X']:.2f}, {node_data['Y']:.2f})
                    - **Utilization:** {node_data['utilization_ratio']:.1%}
                    - **Load Combination:** {node_data.get('Display_Load', node_data.get('Load_Combination', 'N/A'))}
                    - **Max Pile Load:** {node_data['max_pile_load']:.2f} tonf
                    - **Category:** {node_data['category']}
                    - **Status:** {'✅ Safe' if node_data['is_safe'] else '❌ Over-capacity'}
                    """)
                    
                    if node_data.get('Has_Tension', False):
                        st.markdown("""
                        <div class="tension-warning">
                        ⚠️ This node has tension forces
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Show foundation layout
                    if foundation_id in st.session_state.foundation_properties:
                        props = st.session_state.foundation_properties[foundation_id]
                        coords = np.array(props['coords'])
                        
                        fig_layout = go.Figure()
                        
                        # Add piles
                        fig_layout.add_trace(go.Scatter(
                            x=coords[:, 0],
                            y=coords[:, 1],
                            mode='markers+text',
                            marker=dict(
                                size=40,
                                color=node_data['color'],
                                symbol='circle',
                                line=dict(color='darkblue', width=2)
                            ),
                            text=[f'P{i+1}' for i in range(len(coords))],
                            textposition='middle center',
                            textfont=dict(size=12, color='white', family='Arial Black'),
                            name='Piles'
                        ))
                        
                        # Add centroid
                        fig_layout.add_trace(go.Scatter(
                            x=[props['centroid'][0]],
                            y=[props['centroid'][1]],
                            mode='markers',
                            marker=dict(size=20, color='red', symbol='x'),
                            name='Centroid'
                        ))
                        
                        # Add bounding box
                        fig_layout.add_shape(
                            type="rect",
                            x0=min(coords[:, 0]) - 0.3,
                            y0=min(coords[:, 1]) - 0.3,
                            x1=max(coords[:, 0]) + 0.3,
                            y1=max(coords[:, 1]) + 0.3,
                            line=dict(color="gray", dash="dash", width=2)
                        )
                        
                        fig_layout.update_layout(
                            title=f"{foundation_id} Layout - {node_data['foundation_name']}",
                            xaxis_title='X (m)',
                            yaxis_title='Y (m)',
                            height=500,
                            showlegend=True,
                            xaxis=dict(scaleanchor="y", showgrid=True, zeroline=True),
                            yaxis=dict(scaleanchor="x", showgrid=True, zeroline=True)
                        )
                        
                        st.plotly_chart(fig_layout, use_container_width=True)
            
            # Export site plan data
            st.markdown("### Export Site Plan Data")
            
            site_plan_export = results[['Node', 'X', 'Y', 'foundation_id', 'foundation_name', 
                                       'n_piles', 'utilization_ratio', 'category']].copy()
            site_plan_export.columns = ['Node', 'X (m)', 'Y (m)', 'Foundation ID', 
                                       'Foundation Name', 'Piles', 'Utilization', 'Category']
            
            csv_site_plan = site_plan_export.to_csv(index=False)
            st.download_button(
                "📥 Download Site Plan Data (CSV)",
                data=csv_site_plan,
                file_name=f"site_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    else:
        st.info("Run analysis first to view site plan and foundation locations")


with tab7:
    st.markdown('<h2 class="section-header">💾 Export Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.final_results is not None:
        results = st.session_state.final_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export CSV with all details
            export_df = results.copy()
            if 'Load_Combination' in export_df.columns:
                export_df['Critical_Load_Combination'] = export_df['Load_Combination']
            else:
                export_df['Critical_Load_Combination'] = export_df.get('Load_Case', 'N/A')
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                "📥 Download Complete Results (CSV)",
                data=csv,
                file_name=f"pile_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export foundation properties
            if st.session_state.foundation_properties:
                props_json = json.dumps(st.session_state.foundation_properties, indent=2)
                st.download_button(
                    "📥 Export Foundation Properties (JSON)",
                    data=props_json,
                    file_name=f"foundation_properties_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        # Generate comprehensive report
        tension_count = len(st.session_state.tension_nodes)
        report = f"""# Pile Foundation Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- Pile Diameter: {pile_diameter} m
- Pile Capacity: {pile_capacity} tonf
- Target Utilization: {target_utilization:.0%}

## Results Summary
- Total Nodes: {len(results)}
- Average Utilization: {results['utilization_ratio'].mean():.1%}
- Safe Designs: {len(results[results['is_safe']])} / {len(results)}
- Total Piles Required: {int(results['n_piles'].sum())}
- **Nodes with Tension: {tension_count}**

## Foundation Distribution
{results['foundation_id'].value_counts().to_string()}

## Critical Load Combinations Used
"""
        
        # Build load combination table
        if 'Load_Combination' in results.columns:
            load_combo_df = results[['Node', 'Load_Combination', 'utilization_ratio']]
        else:
            load_combo_df = results[['Node', 'Load_Case', 'utilization_ratio']]
            load_combo_df.rename(columns={'Load_Case': 'Load_Combination'}, inplace=True)
        
        report += load_combo_df.to_string()
        
        report += "\n\n## Warnings\n"
        
        if tension_count > 0:
            report += f"\n### ⚠️ TENSION WARNING\n"
            report += f"{tension_count} nodes have tensile forces (negative Fz).\n"
            report += "These require special foundation design considerations.\n\n"
            for tension_node in st.session_state.tension_nodes:
                report += f"- Node {tension_node['Node']}: Fz = {tension_node['Fz']:.2f} tonf\n"
        
        report += """
## Foundation Properties Summary
"""
        for fid in results['foundation_id'].unique():
            if fid in st.session_state.foundation_properties:
                props = st.session_state.foundation_properties[fid]
                report += f"""
### {fid}
- Number of Piles: {props['n_piles']}
- Ixx: {props['Ixx']:.3f} m²
- Iyy: {props['Iyy']:.3f} m²
- cx_max: {props['cx_max']:.3f} m
- cy_max: {props['cy_max']:.3f} m
- Zx: {props['Zx']:.3f} m²
- Zy: {props['Zy']:.3f} m²
"""
        
        st.download_button(
            "📄 Download Comprehensive Report (MD)",
            data=report,
            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )
        
        st.success("✅ Export options ready!")
    else:
        st.info("No results to export. Run analysis first.")
