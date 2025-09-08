import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io

# Page configuration
st.set_page_config(
    page_title="Advanced Pile Foundation Analysis Tool",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .formula-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
    .property-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #b0d4f1;
        margin: 0.5rem 0;
    }
    .custom-config {
        background-color: #fff5ee;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #ff8c00;
        margin: 1rem 0;
    }
    .safe-design {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .unsafe-design {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .optimal-design {
        background-color: #cff4fc;
        border-left: 5px solid #0dcaf0;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏗️ Advanced Pile Foundation Analysis Tool</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">With Custom Foundation Properties Configuration</p>', unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'final_results' not in st.session_state:
    st.session_state.final_results = None
if 'custom_foundations' not in st.session_state:
    st.session_state.custom_foundations = {}
if 'foundation_database' not in st.session_state:
    st.session_state.foundation_database = {}

# ===================== DEFAULT PILE CONFIGURATIONS =====================
DEFAULT_PILE_CONFIGURATIONS = {
    'F3': {
        'name': '3-Pile Triangle',
        'num_piles': 3,
        'arrangement': 'Triangular',
        'coordinates': [
            (0, 1.155),
            (-1.0, -0.577),
            (1.0, -0.577)
        ]
    },
    'F4': {
        'name': '4-Pile Square (2×2)',
        'num_piles': 4,
        'arrangement': 'Square',
        'coordinates': [
            (-1.0, 1.0), (1.0, 1.0),
            (-1.0, -1.0), (1.0, -1.0)
        ]
    },
    'F5': {
        'name': '5-Pile Pentagon',
        'num_piles': 5,
        'arrangement': 'Pentagon',
        'coordinates': [
            (0, 1.2),
            (1.14, 0.37),
            (0.71, -0.97),
            (-0.71, -0.97),
            (-1.14, 0.37)
        ]
    },
    'F6': {
        'name': '6-Pile Rectangle (2×3)',
        'num_piles': 6,
        'arrangement': 'Rectangular 2×3',
        'coordinates': [
            (-1.0, 2.0), (1.0, 2.0),
            (-1.0, 0), (1.0, 0),
            (-1.0, -2.0), (1.0, -2.0)
        ]
    },
    'F7': {
        'name': '7-Pile (Hexagon + Center)',
        'num_piles': 7,
        'arrangement': 'Hexagonal with center',
        'coordinates': [
            (0, 0),
            (1.2, 0), (-1.2, 0),
            (0.6, 1.04), (-0.6, 1.04),
            (0.6, -1.04), (-0.6, -1.04)
        ]
    },
    'F8': {
        'name': '8-Pile Rectangle (2×4)',
        'num_piles': 8,
        'arrangement': 'Rectangular 2×4',
        'coordinates': [
            (-1.0, 3.0), (1.0, 3.0),
            (-1.0, 1.0), (1.0, 1.0),
            (-1.0, -1.0), (1.0, -1.0),
            (-1.0, -3.0), (1.0, -3.0)
        ]
    },
    'F9': {
        'name': '9-Pile Square (3×3)',
        'num_piles': 9,
        'arrangement': 'Square 3×3',
        'coordinates': [
            (-2.0, 2.0), (0, 2.0), (2.0, 2.0),
            (-2.0, 0), (0, 0), (2.0, 0),
            (-2.0, -2.0), (0, -2.0), (2.0, -2.0)
        ]
    },
    'F10': {
        'name': '10-Pile Rectangle (2×5)',
        'num_piles': 10,
        'arrangement': 'Rectangular 2×5',
        'coordinates': [
            (-1.0, 4.0), (1.0, 4.0),
            (-1.0, 2.0), (1.0, 2.0),
            (-1.0, 0), (1.0, 0),
            (-1.0, -2.0), (1.0, -2.0),
            (-1.0, -4.0), (1.0, -4.0)
        ]
    },
    'F12': {
        'name': '12-Pile Rectangle (3×4)',
        'num_piles': 12,
        'arrangement': 'Rectangular 3×4',
        'coordinates': [
            (-2.0, 3.0), (0, 3.0), (2.0, 3.0),
            (-2.0, 1.0), (0, 1.0), (2.0, 1.0),
            (-2.0, -1.0), (0, -1.0), (2.0, -1.0),
            (-2.0, -3.0), (0, -3.0), (2.0, -3.0)
        ]
    },
    'F15': {
        'name': '15-Pile Rectangle (3×5)',
        'num_piles': 15,
        'arrangement': 'Rectangular 3×5',
        'coordinates': [
            (-2.0, 4.0), (0, 4.0), (2.0, 4.0),
            (-2.0, 2.0), (0, 2.0), (2.0, 2.0),
            (-2.0, 0), (0, 0), (2.0, 0),
            (-2.0, -2.0), (0, -2.0), (2.0, -2.0),
            (-2.0, -4.0), (0, -4.0), (2.0, -4.0)
        ]
    },
    'F18': {
        'name': '18-Pile Rectangle (3×6)',
        'num_piles': 18,
        'arrangement': 'Rectangular 3×6',
        'coordinates': [
            (-2.0, 5.0), (0, 5.0), (2.0, 5.0),
            (-2.0, 3.0), (0, 3.0), (2.0, 3.0),
            (-2.0, 1.0), (0, 1.0), (2.0, 1.0),
            (-2.0, -1.0), (0, -1.0), (2.0, -1.0),
            (-2.0, -3.0), (0, -3.0), (2.0, -3.0),
            (-2.0, -5.0), (0, -5.0), (2.0, -5.0)
        ]
    },
    'F20': {
        'name': '20-Pile Rectangle (4×5)',
        'num_piles': 20,
        'arrangement': 'Rectangular 4×5',
        'coordinates': [
            (-3.0, 4.0), (-1.0, 4.0), (1.0, 4.0), (3.0, 4.0),
            (-3.0, 2.0), (-1.0, 2.0), (1.0, 2.0), (3.0, 2.0),
            (-3.0, 0), (-1.0, 0), (1.0, 0), (3.0, 0),
            (-3.0, -2.0), (-1.0, -2.0), (1.0, -2.0), (3.0, -2.0),
            (-3.0, -4.0), (-1.0, -4.0), (1.0, -4.0), (3.0, -4.0)
        ]
    }
}

# Default nodes list
DEFAULT_NODES = [789, 790, 791,
                4561, 4572, 4576, 4581, 4586,
                4627, 4632, 4637,
                4657, 4663,
                4748, 4749, 4752,
                4827, 4831,
                5568, 5569,
                5782, 5784,
                7446, 7447, 7448, 7453, 7461, 7464]

class PileGroupAnalyzer:
    """
    Pile group analysis using proper structural engineering theory
    Based on: Bowles (1996), Das (2016), AASHTO LRFD (2020)
    """
    
    def __init__(self, pile_diameter=0.6, pile_capacity=120, pile_spacing=1.5, 
                 pile_material='Concrete', pile_type='Driven', 
                 cap_thickness=1.0, cap_material='Concrete',
                 safety_factor=1.5):
        """
        Initialize pile group analyzer with comprehensive properties
        """
        self.pile_diameter = pile_diameter
        self.pile_capacity = pile_capacity
        self.pile_spacing = pile_spacing
        self.pile_material = pile_material
        self.pile_type = pile_type
        self.cap_thickness = cap_thickness
        self.cap_material = cap_material
        self.safety_factor = safety_factor
        self.min_spacing = 2.5 * pile_diameter
        
        # Material properties
        self.material_properties = {
            'Concrete': {'E': 30000, 'density': 2400},  # MPa, kg/m³
            'Steel': {'E': 200000, 'density': 7850},
            'Composite': {'E': 50000, 'density': 2000}
        }
        
    def calculate_section_properties(self, footing_config, custom_spacing=None):
        """
        Calculate geometric properties of pile group
        """
        if isinstance(footing_config, dict):
            config = footing_config
        else:
            config = DEFAULT_PILE_CONFIGURATIONS.get(footing_config, None)
            if config is None:
                raise ValueError(f"Unknown footing type: {footing_config}")
        
        # Use custom spacing if provided
        spacing = custom_spacing if custom_spacing else self.pile_spacing
        
        # Scale coordinates by spacing
        coords = np.array(config['coordinates']) * spacing
        
        # Calculate centroid
        centroid_x = np.mean(coords[:, 0])
        centroid_y = np.mean(coords[:, 1])
        
        # Adjust coordinates to centroid
        x_coords = coords[:, 0] - centroid_x
        y_coords = coords[:, 1] - centroid_y
        
        # Calculate moment of inertia
        Ixx = np.sum(x_coords**2)
        Iyy = np.sum(y_coords**2)
        
        # Maximum distances from centroid
        xmax = np.max(np.abs(x_coords)) if len(x_coords) > 0 else 1.0
        ymax = np.max(np.abs(y_coords)) if len(y_coords) > 0 else 1.0
        
        # Section modulus
        Zx = Ixx / xmax if xmax > 0 else float('inf')
        Zy = Iyy / ymax if ymax > 0 else float('inf')
        
        # Group efficiency (Converse-Labarre formula)
        n_piles = len(coords)
        if n_piles > 1:
            theta = np.arctan(self.pile_diameter / spacing)
            m = int(np.sqrt(n_piles))  # Approximate rows
            n = int(np.ceil(n_piles / m))  # Approximate columns
            efficiency = 1 - (theta * (n - 1) * m + (m - 1) * n) / (90 * m * n)
            efficiency = max(0.7, min(1.0, efficiency))  # Limit between 0.7 and 1.0
        else:
            efficiency = 1.0
        
        return {
            'Ixx': Ixx,
            'Iyy': Iyy,
            'xmax': xmax,
            'ymax': ymax,
            'Zx': Zx,
            'Zy': Zy,
            'n_piles': n_piles,
            'name': config.get('name', 'Custom'),
            'arrangement': config.get('arrangement', 'Custom'),
            'coordinates': coords.tolist(),
            'efficiency': efficiency,
            'cap_area': self.calculate_cap_area(coords),
            'cap_volume': self.calculate_cap_area(coords) * self.cap_thickness
        }
    
    def calculate_cap_area(self, coords):
        """Calculate pile cap area using convex hull"""
        if len(coords) < 3:
            return self.pile_diameter ** 2
        
        # Simple bounding box for now
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        # Add edge distance (typically 0.5m beyond piles)
        edge_distance = 0.5
        width = (x_max - x_min) + 2 * edge_distance + self.pile_diameter
        length = (y_max - y_min) + 2 * edge_distance + self.pile_diameter
        
        return width * length
    
    def calculate_pile_loads(self, Fz, Mx, My, footing_config, include_cap_weight=True):
        """
        Calculate maximum pile load with comprehensive analysis
        """
        # Get section properties
        props = self.calculate_section_properties(footing_config)
        
        # Add pile cap self-weight if requested
        if include_cap_weight:
            cap_weight = props['cap_volume'] * self.material_properties[self.cap_material]['density'] * 9.81 / 1000  # tonf
            Fz_total = abs(Fz) + cap_weight
        else:
            Fz_total = abs(Fz)
        
        # Apply group efficiency
        effective_capacity = self.pile_capacity * props['efficiency']
        
        # Component stresses
        axial_stress = Fz_total / props['n_piles']
        
        # Moment-induced stresses
        if props['Zx'] != float('inf'):
            stress_from_My = abs(My) / props['Zx']
        else:
            stress_from_My = 0
            
        if props['Zy'] != float('inf'):
            stress_from_Mx = abs(Mx) / props['Zy']
        else:
            stress_from_Mx = 0
        
        # Maximum and minimum pile loads
        max_pile_load = axial_stress + stress_from_My + stress_from_Mx
        min_pile_load = axial_stress - stress_from_My - stress_from_Mx
        
        # Design checks
        utilization = max_pile_load / effective_capacity
        design_capacity = effective_capacity / self.safety_factor
        design_utilization = max_pile_load / design_capacity
        
        # Lateral load check (simplified)
        Fx = 0  # Placeholder for lateral loads
        Fy = 0
        lateral_load = np.sqrt(Fx**2 + Fy**2) / props['n_piles']
        lateral_capacity = 0.1 * self.pile_capacity  # Simplified: 10% of axial capacity
        lateral_utilization = lateral_load / lateral_capacity if lateral_capacity > 0 else 0
        
        # Settlement estimation (simplified Vesic method)
        pile_length = 15.0  # Default pile length in meters
        pile_area = np.pi * (self.pile_diameter/2)**2
        E_pile = self.material_properties[self.pile_material]['E'] * 1000  # kPa
        settlement = (max_pile_load * 1000 * pile_length) / (pile_area * E_pile) * 1000  # mm
        
        # Category determination
        if utilization > 1.0:
            category = "Over-Capacity"
            category_color = "🔴"
        elif utilization > 0.95:
            category = "Near-Capacity"
            category_color = "🟠"
        elif utilization > 0.80:
            category = "Optimal"
            category_color = "🟢"
        elif utilization > 0.60:
            category = "Conservative"
            category_color = "🟡"
        else:
            category = "Over-Conservative"
            category_color = "⚪"
        
        return {
            'footing_type': footing_config if isinstance(footing_config, str) else 'Custom',
            'footing_name': props['name'],
            'arrangement': props['arrangement'],
            'n_piles': props['n_piles'],
            'axial_stress': axial_stress,
            'moment_stress_mx': stress_from_Mx,
            'moment_stress_my': stress_from_My,
            'max_pile_load': max_pile_load,
            'min_pile_load': min_pile_load,
            'pile_capacity': self.pile_capacity,
            'effective_capacity': effective_capacity,
            'design_capacity': design_capacity,
            'utilization_ratio': utilization,
            'design_utilization': design_utilization,
            'is_safe': utilization <= 1.0,
            'has_tension': min_pile_load < 0,
            'category': category,
            'category_color': category_color,
            'Ixx': props['Ixx'],
            'Iyy': props['Iyy'],
            'Zx': props['Zx'],
            'Zy': props['Zy'],
            'xmax': props['xmax'],
            'ymax': props['ymax'],
            'coordinates': props['coordinates'],
            'efficiency': props['efficiency'],
            'cap_area': props['cap_area'],
            'cap_volume': props['cap_volume'],
            'cap_weight': cap_weight if include_cap_weight else 0,
            'lateral_utilization': lateral_utilization,
            'estimated_settlement': settlement,
            'safety_factor': self.safety_factor
        }
    
    def optimize_footing(self, Fz, Mx, My, target_utilization=0.85, available_footings=None):
        """
        Find optimal footing type for target utilization
        """
        if available_footings is None:
            available_footings = DEFAULT_PILE_CONFIGURATIONS
        
        results = []
        
        # Try each footing configuration
        for footing_key, footing_config in available_footings.items():
            analysis = self.calculate_pile_loads(Fz, Mx, My, footing_config)
            analysis['footing_key'] = footing_key
            analysis['target_diff'] = abs(analysis['utilization_ratio'] - target_utilization)
            results.append(analysis)
        
        # Convert to DataFrame for easier analysis
        df_results = pd.DataFrame(results)
        
        # Filter safe designs
        safe_designs = df_results[df_results['is_safe']]
        
        if len(safe_designs) > 0:
            # Among safe designs, find closest to target utilization
            optimal_idx = safe_designs['target_diff'].idxmin()
            optimal_design = safe_designs.loc[optimal_idx].to_dict()
        else:
            # If no safe design, use the one with lowest utilization
            optimal_idx = df_results['utilization_ratio'].idxmin()
            optimal_design = df_results.loc[optimal_idx].to_dict()
        
        return optimal_design

def create_custom_foundation_layout():
    """Create interface for custom foundation input"""
    st.markdown('<h3>🏗️ Custom Foundation Configuration</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("### Quick Templates")
        
        template = st.selectbox(
            "Select Template",
            ["Custom", "Rectangular Grid", "Circular", "Staggered", "Triangular Grid"]
        )
        
        if template == "Rectangular Grid":
            rows = st.number_input("Number of Rows", 2, 10, 3)
            cols = st.number_input("Number of Columns", 2, 10, 3)
            spacing_x = st.number_input("X Spacing (m)", 1.0, 5.0, 1.5, 0.1)
            spacing_y = st.number_input("Y Spacing (m)", 1.0, 5.0, 1.5, 0.1)
            
            if st.button("Generate Grid"):
                coordinates = []
                for i in range(rows):
                    for j in range(cols):
                        x = (j - (cols-1)/2) * spacing_x
                        y = (i - (rows-1)/2) * spacing_y
                        coordinates.append([x, y])
                st.session_state['temp_coords'] = coordinates
                
        elif template == "Circular":
            n_piles = st.number_input("Number of Piles", 6, 20, 8)
            radius = st.number_input("Radius (m)", 1.0, 10.0, 3.0, 0.1)
            include_center = st.checkbox("Include Center Pile")
            
            if st.button("Generate Circle"):
                coordinates = []
                if include_center:
                    coordinates.append([0, 0])
                for i in range(n_piles):
                    angle = 2 * np.pi * i / n_piles
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    coordinates.append([x, y])
                st.session_state['temp_coords'] = coordinates
    
    with col2:
        st.markdown("### Manual Coordinate Entry")
        
        # Option 1: Text area for coordinates
        coord_input_method = st.radio(
            "Input Method",
            ["Text Entry", "Interactive Table", "Upload CSV"]
        )
        
        if coord_input_method == "Text Entry":
            coord_text = st.text_area(
                "Enter coordinates (one per line, format: x,y)",
                value="0,0\n1.5,0\n0,1.5\n1.5,1.5",
                height=200
            )
            
            if st.button("Parse Coordinates"):
                try:
                    coordinates = []
                    for line in coord_text.strip().split('\n'):
                        if line.strip():
                            x, y = map(float, line.split(','))
                            coordinates.append([x, y])
                    st.session_state['temp_coords'] = coordinates
                    st.success(f"Parsed {len(coordinates)} pile locations")
                except Exception as e:
                    st.error(f"Error parsing coordinates: {e}")
        
        elif coord_input_method == "Interactive Table":
            n_piles = st.number_input("Number of Piles", 3, 50, 4)
            
            # Create editable dataframe
            if 'temp_coords' not in st.session_state:
                st.session_state['temp_coords'] = [[0, 0] for _ in range(n_piles)]
            
            df_coords = pd.DataFrame(st.session_state['temp_coords'], columns=['X (m)', 'Y (m)'])
            edited_df = st.data_editor(df_coords, num_rows="dynamic")
            st.session_state['temp_coords'] = edited_df.values.tolist()
        
        elif coord_input_method == "Upload CSV":
            uploaded_coords = st.file_uploader(
                "Upload coordinate file (CSV with X,Y columns)",
                type=['csv']
            )
            if uploaded_coords:
                try:
                    df_coords = pd.read_csv(uploaded_coords)
                    if 'X' in df_coords.columns and 'Y' in df_coords.columns:
                        coordinates = df_coords[['X', 'Y']].values.tolist()
                    elif 'x' in df_coords.columns and 'y' in df_coords.columns:
                        coordinates = df_coords[['x', 'y']].values.tolist()
                    else:
                        coordinates = df_coords.iloc[:, :2].values.tolist()
                    st.session_state['temp_coords'] = coordinates
                    st.success(f"Loaded {len(coordinates)} pile locations")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
    
    with col3:
        st.markdown("### Foundation Properties")
        
        foundation_name = st.text_input("Foundation Name", "Custom_F1")
        foundation_desc = st.text_input("Description", "Custom Configuration")
        
        # Pile cap properties
        st.markdown("**Pile Cap Properties**")
        cap_thickness = st.number_input("Cap Thickness (m)", 0.5, 3.0, 1.0, 0.1)
        cap_material = st.selectbox("Cap Material", ["Concrete", "Steel", "Composite"])
        edge_distance = st.number_input("Edge Distance (m)", 0.3, 1.0, 0.5, 0.1)
        
        # Save configuration
        if st.button("💾 Save Foundation Configuration", type="primary"):
            if 'temp_coords' in st.session_state and len(st.session_state['temp_coords']) > 0:
                config = {
                    'name': foundation_desc,
                    'num_piles': len(st.session_state['temp_coords']),
                    'arrangement': 'Custom',
                    'coordinates': st.session_state['temp_coords'],
                    'cap_thickness': cap_thickness,
                    'cap_material': cap_material,
                    'edge_distance': edge_distance
                }
                st.session_state.custom_foundations[foundation_name] = config
                st.success(f"Saved foundation configuration: {foundation_name}")
            else:
                st.error("No coordinates defined")
    
    # Visualization of current configuration
    if 'temp_coords' in st.session_state and len(st.session_state['temp_coords']) > 0:
        st.markdown("### 📊 Foundation Layout Preview")
        
        coords = np.array(st.session_state['temp_coords'])
        
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
        centroid_x = np.mean(coords[:, 0])
        centroid_y = np.mean(coords[:, 1])
        fig.add_trace(go.Scatter(
            x=[centroid_x],
            y=[centroid_y],
            mode='markers',
            marker=dict(size=10, color='red', symbol='x'),
            name='Centroid'
        ))
        
        # Add cap outline (simplified rectangle)
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        edge = 0.5
        
        fig.add_shape(type="rect",
            x0=x_min - edge, y0=y_min - edge,
            x1=x_max + edge, y1=y_max + edge,
            line=dict(color="gray", width=2, dash="dash"),
            fillcolor="rgba(200,200,200,0.2)"
        )
        
        fig.update_layout(
            title=f'Foundation Layout - {len(coords)} Piles',
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            showlegend=True,
            hovermode='closest',
            xaxis=dict(scaleanchor="y", scaleratio=1, gridcolor='lightgray'),
            yaxis=dict(scaleanchor="x", scaleratio=1, gridcolor='lightgray'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display properties
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Piles", len(coords))
            Ixx = np.sum((coords[:, 0] - centroid_x)**2)
            st.metric("Ixx", f"{Ixx:.2f} m²")
        
        with col2:
            cap_area = (x_max - x_min + 2*edge) * (y_max - y_min + 2*edge)
            st.metric("Cap Area", f"{cap_area:.2f} m²")
            Iyy = np.sum((coords[:, 1] - centroid_y)**2)
            st.metric("Iyy", f"{Iyy:.2f} m²")
        
        with col3:
            if cap_thickness:
                cap_volume = cap_area * cap_thickness
                st.metric("Cap Volume", f"{cap_volume:.2f} m³")
            xmax = np.max(np.abs(coords[:, 0] - centroid_x))
            ymax = np.max(np.abs(coords[:, 1] - centroid_y))
            st.metric("Max Distance", f"{max(xmax, ymax):.2f} m")

def load_data(uploaded_file):
    """Load and process uploaded CSV file"""
    try:
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(uploaded_file, encoding=encoding)
                return df, f"Successfully loaded with {encoding} encoding"
            except UnicodeDecodeError:
                continue
        return None, "Could not decode file with any encoding"
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

def process_pile_analysis(df, nodes, analyzer, target_utilization=0.85, use_custom_foundations=False):
    """Process pile analysis with custom or default foundations"""
    
    # Determine which foundations to use
    if use_custom_foundations and len(st.session_state.custom_foundations) > 0:
        available_footings = {**DEFAULT_PILE_CONFIGURATIONS, **st.session_state.custom_foundations}
    else:
        available_footings = DEFAULT_PILE_CONFIGURATIONS
    
    # Filter for selected nodes
    df_filtered = df[df['Node'].isin(nodes)].copy()
    
    if len(df_filtered) == 0:
        return None
    
    # Process all load combinations
    all_results = []
    
    for idx, row in df_filtered.iterrows():
        # Extract loads and moments
        Fz = abs(row.get('FZ (tonf)', row.get('Fz', 0)))
        Mx = abs(row.get('MX (tonf·m)', row.get('Mx', 0)))
        My = abs(row.get('MY (tonf·m)', row.get('My', 0)))
        
        # Find optimal footing
        optimal = analyzer.optimize_footing(Fz, Mx, My, target_utilization, available_footings)
        
        # Add row information
        optimal['Node'] = row['Node']
        optimal['Load_Case'] = row.get('Load Combination', row.get('Load Case', f'LC_{idx}'))
        optimal['X'] = row.get('X', 0)
        optimal['Y'] = row.get('Y', 0)
        optimal['Z'] = row.get('Z', 0)
        optimal['Fz'] = Fz
        optimal['Mx'] = Mx
        optimal['My'] = My
        optimal['Target_Utilization'] = target_utilization
        
        all_results.append(optimal)
    
    return pd.DataFrame(all_results)

def get_critical_design_per_node(df_all_cases):
    """For each node, select the critical (maximum) footing requirement"""
    
    if df_all_cases is None or len(df_all_cases) == 0:
        return None
    
    def get_critical_case(group):
        """Get the critical case for a node (max piles required)"""
        max_pile_idx = group['n_piles'].idxmax()
        critical_case = group.loc[max_pile_idx].copy()
        
        critical_case['Total_Load_Cases'] = len(group)
        critical_case['Max_Fz'] = group['Fz'].max()
        critical_case['Max_Mx'] = group['Mx'].max()
        critical_case['Max_My'] = group['My'].max()
        critical_case['Max_Utilization'] = group['utilization_ratio'].max()
        critical_case['Avg_Utilization'] = group['utilization_ratio'].mean()
        
        return critical_case
    
    critical_results = df_all_cases.groupby('Node').apply(get_critical_case).reset_index(drop=True)
    
    return critical_results

def create_visualizations(results_df):
    """Create comprehensive visualizations"""
    plots = {}
    
    # 1. Utilization Distribution
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=results_df['utilization_ratio'],
        nbinsx=20,
        name='Utilization Distribution',
        marker_color='lightblue',
        opacity=0.7
    ))
    fig_dist.add_vline(x=0.85, line_dash="dash", line_color="red", 
                      annotation_text="Target (85%)")
    fig_dist.add_vline(x=1.0, line_dash="solid", line_color="darkred", 
                      annotation_text="Capacity Limit")
    fig_dist.update_layout(
        title='Utilization Ratio Distribution',
        xaxis_title='Utilization Ratio',
        yaxis_title='Count',
        showlegend=False,
        height=400
    )
    plots['utilization_dist'] = fig_dist
    
    # 2. Efficiency vs Settlement
    if 'efficiency' in results_df.columns and 'estimated_settlement' in results_df.columns:
        fig_eff = px.scatter(
            results_df,
            x='efficiency',
            y='estimated_settlement',
            color='utilization_ratio',
            size='n_piles',
            hover_data=['Node', 'footing_type'],
            title='Group Efficiency vs Settlement',
            labels={'efficiency': 'Group Efficiency', 
                   'estimated_settlement': 'Estimated Settlement (mm)'},
            color_continuous_scale='RdYlGn_r'
        )
        fig_eff.update_layout(height=400)
        plots['efficiency_settlement'] = fig_eff
    
    return plots

# ===================== SIDEBAR CONFIGURATION =====================
st.sidebar.title("📋 Configuration")

# Configuration mode
config_mode = st.sidebar.radio(
    "Configuration Mode",
    ["Quick Setup", "Advanced Properties", "Foundation Database"]
)

if config_mode == "Quick Setup":
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload CSV File",
        type=['csv'],
        help="Upload structural analysis data with loads and moments"
    )
    
    # Basic pile parameters
    st.sidebar.subheader("🔧 Basic Parameters")
    
    pile_capacity = st.sidebar.number_input(
        "Single Pile Capacity (tonf)",
        min_value=50,
        max_value=500,
        value=120,
        step=10
    )
    
    pile_spacing = st.sidebar.number_input(
        "Pile Spacing (m)",
        min_value=1.0,
        max_value=5.0,
        value=1.5,
        step=0.1
    )
    
    target_utilization = st.sidebar.slider(
        "Target Utilization Ratio",
        min_value=0.70,
        max_value=0.95,
        value=0.85,
        step=0.05
    )
    
    # Set default values for advanced properties
    pile_diameter = 0.6
    pile_material = "Concrete"
    pile_type = "Driven"
    cap_thickness = 1.0
    cap_material = "Concrete"
    safety_factor = 1.5
    include_cap_weight = True
    use_custom_foundations = False

elif config_mode == "Advanced Properties":
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload CSV File",
        type=['csv']
    )
    
    # Pile Properties
    st.sidebar.subheader("🏗️ Pile Properties")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        pile_diameter = st.number_input(
            "Diameter (m)",
            0.3, 2.0, 0.6, 0.1
        )
        pile_capacity = st.number_input(
            "Capacity (tonf)",
            50, 500, 120, 10
        )
    
    with col2:
        pile_length = st.number_input(
            "Length (m)",
            5.0, 50.0, 15.0, 1.0
        )
        pile_spacing = st.number_input(
            "Spacing (m)",
            1.0, 5.0, 1.5, 0.1
        )
    
    pile_material = st.sidebar.selectbox(
        "Pile Material",
        ["Concrete", "Steel", "Composite"]
    )
    
    pile_type = st.sidebar.selectbox(
        "Pile Type",
        ["Driven", "Bored", "CFA", "Micropile"]
    )
    
    # Pile Cap Properties
    st.sidebar.subheader("🔲 Pile Cap Properties")
    
    cap_thickness = st.sidebar.number_input(
        "Cap Thickness (m)",
        0.5, 3.0, 1.0, 0.1
    )
    
    cap_material = st.sidebar.selectbox(
        "Cap Material",
        ["Concrete", "Steel", "Composite"]
    )
    
    include_cap_weight = st.sidebar.checkbox(
        "Include Cap Self-Weight",
        value=True
    )
    
    # Design Parameters
    st.sidebar.subheader("📐 Design Parameters")
    
    safety_factor = st.sidebar.number_input(
        "Safety Factor",
        1.0, 3.0, 1.5, 0.1
    )
    
    target_utilization = st.sidebar.slider(
        "Target Utilization",
        0.70, 0.95, 0.85, 0.05
    )
    
    use_custom_foundations = st.sidebar.checkbox(
        "Use Custom Foundations",
        value=False
    )
    
    # Check minimum spacing
    min_spacing = 2.5 * pile_diameter
    if pile_spacing < min_spacing:
        st.sidebar.warning(f"⚠️ Spacing should be ≥ {min_spacing:.1f}m (2.5D)")

else:  # Foundation Database
    st.sidebar.markdown("### 📚 Foundation Library")
    
    # Display saved foundations
    if len(st.session_state.custom_foundations) > 0:
        st.sidebar.success(f"💾 {len(st.session_state.custom_foundations)} custom foundations saved")
        
        for name, config in st.session_state.custom_foundations.items():
            with st.sidebar.expander(f"{name} ({config['num_piles']} piles)"):
                st.write(f"**Type:** {config['arrangement']}")
                st.write(f"**Piles:** {config['num_piles']}")
                if st.button(f"Delete {name}", key=f"del_{name}"):
                    del st.session_state.custom_foundations[name]
                    st.rerun()
    else:
        st.sidebar.info("No custom foundations saved yet")
    
    # Export/Import options
    st.sidebar.markdown("### 💾 Export/Import")
    
    if st.sidebar.button("Export Foundations"):
        if len(st.session_state.custom_foundations) > 0:
            json_str = json.dumps(st.session_state.custom_foundations, indent=2)
            st.sidebar.download_button(
                "Download JSON",
                data=json_str,
                file_name="custom_foundations.json",
                mime="application/json"
            )
    
    uploaded_json = st.sidebar.file_uploader(
        "Import Foundations (JSON)",
        type=['json']
    )
    
    if uploaded_json:
        try:
            imported = json.load(uploaded_json)
            st.session_state.custom_foundations.update(imported)
            st.sidebar.success(f"Imported {len(imported)} foundations")
        except Exception as e:
            st.sidebar.error(f"Error importing: {e}")
    
    # Set defaults for this mode
    uploaded_file = None
    pile_diameter = 0.6
    pile_capacity = 120
    pile_spacing = 1.5
    pile_material = "Concrete"
    pile_type = "Driven"
    cap_thickness = 1.0
    cap_material = "Concrete"
    safety_factor = 1.5
    target_utilization = 0.85
    include_cap_weight = True
    use_custom_foundations = True

# Node selection (common for all modes)
st.sidebar.subheader("🎯 Node Selection")
use_default_nodes = st.sidebar.checkbox("Use Default Nodes", value=True)

if use_default_nodes:
    selected_nodes = DEFAULT_NODES
    st.sidebar.info(f"Using {len(DEFAULT_NODES)} default nodes")
else:
    nodes_input = st.sidebar.text_area(
        "Enter Node Numbers (comma-separated)",
        value=",".join(map(str, DEFAULT_NODES[:5]))
    )
    try:
        selected_nodes = [int(x.strip()) for x in nodes_input.split(",") if x.strip()]
        st.sidebar.success(f"Selected {len(selected_nodes)} nodes")
    except:
        st.sidebar.error("Invalid format")
        selected_nodes = DEFAULT_NODES[:5]

# ===================== MAIN CONTENT =====================

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📚 Theory",
    "🏗️ Foundation Properties",
    "📊 Data Input",
    "🔬 Analysis Results",
    "📈 Visualizations",
    "🧮 Calculations",
    "💾 Export"
])

with tab1:
    st.markdown('<h2 class="section-header">📚 Pile Group Design Theory</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Fundamental Formula
        
        The distribution of load to individual piles in a group:
        """)
        
        st.markdown("""
        <div class="formula-box">
        <strong>Maximum Pile Load:</strong><br>
        P<sub>max</sub> = P/n + |M<sub>y</sub>|/Z<sub>x</sub> + |M<sub>x</sub>|/Z<sub>y</sub>
        <br><br>
        <strong>Section Properties:</strong><br>
        • Z<sub>x</sub> = I<sub>xx</sub>/x<sub>max</sub> (Section modulus about X-axis)<br>
        • Z<sub>y</sub> = I<sub>yy</sub>/y<sub>max</sub> (Section modulus about Y-axis)<br>
        • I<sub>xx</sub> = Σx<sub>i</sub>² (Moment of inertia about X-axis)<br>
        • I<sub>yy</sub> = Σy<sub>i</sub>² (Moment of inertia about Y-axis)<br>
        <br>
        <strong>Group Efficiency (Converse-Labarre):</strong><br>
        η = 1 - [θ(n-1)m + (m-1)n] / (90mn)<br>
        where θ = arctan(D/s) in degrees
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        ### 📖 References
        
        1. **Bowles (1996)**  
           Foundation Analysis, 5th Ed.
        
        2. **Das (2016)**  
           Foundation Engineering, 8th Ed.
        
        3. **AASHTO LRFD (2020)**  
           Section 10.7.3.12
        
        4. **Tomlinson (2014)**  
           Pile Design Practice, 6th Ed.
        """)

with tab2:
    st.markdown('<h2 class="section-header">🏗️ Foundation Properties Configuration</h2>', unsafe_allow_html=True)
    
    # Display current configuration
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Current Configuration")
        
        st.markdown('<div class="property-box">', unsafe_allow_html=True)
        st.write("**Pile Properties:**")
        st.write(f"• Diameter: {pile_diameter} m")
        st.write(f"• Capacity: {pile_capacity} tonf")
        st.write(f"• Material: {pile_material}")
        st.write(f"• Type: {pile_type}")
        st.write(f"• Spacing: {pile_spacing} m")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="property-box">', unsafe_allow_html=True)
        st.write("**Pile Cap:**")
        st.write(f"• Thickness: {cap_thickness} m")
        st.write(f"• Material: {cap_material}")
        st.write(f"• Include Weight: {include_cap_weight}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="property-box">', unsafe_allow_html=True)
        st.write("**Design Parameters:**")
        st.write(f"• Safety Factor: {safety_factor}")
        st.write(f"• Target Utilization: {target_utilization:.0%}")
        st.write(f"• Min Spacing: {2.5 * pile_diameter:.1f} m")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Custom foundation creator
        create_custom_foundation_layout()

with tab3:
    st.markdown('<h2 class="section-header">📊 Data Input & Processing</h2>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        df, message = load_data(uploaded_file)
        
        if df is not None:
            st.success(message)
            
            # Display overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                if 'Node' in df.columns:
                    st.metric("Unique Nodes", df['Node'].nunique())
            with col4:
                st.metric("Selected Nodes", len(selected_nodes))
            
            # Data preview
            with st.expander("👁️ Preview Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Run analysis
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    try:
                        # Create analyzer with all properties
                        analyzer = PileGroupAnalyzer(
                            pile_diameter=pile_diameter,
                            pile_capacity=pile_capacity,
                            pile_spacing=pile_spacing,
                            pile_material=pile_material,
                            pile_type=pile_type,
                            cap_thickness=cap_thickness,
                            cap_material=cap_material,
                            safety_factor=safety_factor
                        )
                        
                        # Process analysis
                        all_results = process_pile_analysis(
                            df, selected_nodes, analyzer, 
                            target_utilization, use_custom_foundations
                        )
                        
                        if all_results is not None and len(all_results) > 0:
                            final_results = get_critical_design_per_node(all_results)
                            
                            st.session_state.analysis_results = all_results
                            st.session_state.final_results = final_results
                            
                            st.success(f"✅ Analysis completed for {len(final_results)} nodes!")
                            st.balloons()
                        else:
                            st.error("No data found for selected nodes")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.error(message)
    else:
        st.info("👈 Please upload a CSV file to begin")

# Continue with remaining tabs...
with tab4:
    st.markdown('<h2 class="section-header">🔬 Analysis Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.final_results is not None:
        results = st.session_state.final_results
        
        # Display comprehensive results
        st.dataframe(results, use_container_width=True)
    else:
        st.info("No results yet. Please run analysis.")

with tab5:
    st.markdown('<h2 class="section-header">📈 Visualizations</h2>', unsafe_allow_html=True)
    
    if st.session_state.final_results is not None:
        plots = create_visualizations(st.session_state.final_results)
        for plot in plots.values():
            st.plotly_chart(plot, use_container_width=True)
    else:
        st.info("No results to visualize")

with tab6:
    st.markdown('<h2 class="section-header">🧮 Detailed Calculations</h2>', unsafe_allow_html=True)
    
    if st.session_state.final_results is not None:
        st.write("Detailed calculation breakdown available here")
    else:
        st.info("No results available")

with tab7:
    st.markdown('<h2 class="section-header">💾 Export Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.final_results is not None:
        csv = st.session_state.final_results.to_csv(index=False)
        st.download_button(
            "Download Results CSV",
            data=csv,
            file_name="pile_analysis_results.csv",
            mime="text/csv"
        )
    else:
        st.info("No results to export")
