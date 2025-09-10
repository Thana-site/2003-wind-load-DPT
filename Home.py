import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Pile Foundation Analysis - Theory-Based",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🏗️ Pile Foundation Analysis Tool - Theory-Based Design")

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'final_results' not in st.session_state:
    st.session_state.final_results = None

# Standard pile group configurations based on engineering practice
PILE_CONFIGURATIONS = {
    'F3': {
        'name': '3-Pile Triangle',
        'num_piles': 3,
        'coordinates': [(0, 1.0), (-0.866, -0.5), (0.866, -0.5)],
        'spacing': 1.5
    },
    'F4': {
        'name': '4-Pile Square',
        'num_piles': 4,
        'coordinates': [(-0.75, 0.75), (0.75, 0.75), (-0.75, -0.75), (0.75, -0.75)],
        'spacing': 1.5
    },
    'F5': {
        'name': '5-Pile Pentagon',
        'num_piles': 5,
        'coordinates': [(0, 1.0), (0.951, 0.309), (0.588, -0.809), (-0.588, -0.809), (-0.951, 0.309)],
        'spacing': 1.5
    },
    'F6': {
        'name': '6-Pile Rectangle 2x3',
        'num_piles': 6,
        'coordinates': [(-0.75, 1.5), (0.75, 1.5), (-0.75, 0), (0.75, 0), (-0.75, -1.5), (0.75, -1.5)],
        'spacing': 1.5
    },
    'F7': {
        'name': '7-Pile Hexagon+Center',
        'num_piles': 7,
        'coordinates': [(0, 0), (1.0, 0), (0.5, 0.866), (-0.5, 0.866), (-1.0, 0), (-0.5, -0.866), (0.5, -0.866)],
        'spacing': 1.5
    },
    'F8': {
        'name': '8-Pile Rectangle 2x4',
        'num_piles': 8,
        'coordinates': [(-0.75, 2.25), (0.75, 2.25), (-0.75, 0.75), (0.75, 0.75), 
                       (-0.75, -0.75), (0.75, -0.75), (-0.75, -2.25), (0.75, -2.25)],
        'spacing': 1.5
    },
    'F9': {
        'name': '9-Pile Square 3x3',
        'num_piles': 9,
        'coordinates': [(-1.5, 1.5), (0, 1.5), (1.5, 1.5), (-1.5, 0), (0, 0), (1.5, 0),
                       (-1.5, -1.5), (0, -1.5), (1.5, -1.5)],
        'spacing': 1.5
    },
    'F10': {
        'name': '10-Pile Rectangle 2x5',
        'num_piles': 10,
        'coordinates': [(-0.75, 3.0), (0.75, 3.0), (-0.75, 1.5), (0.75, 1.5), (-0.75, 0), 
                       (0.75, 0), (-0.75, -1.5), (0.75, -1.5), (-0.75, -3.0), (0.75, -3.0)],
        'spacing': 1.5
    },
    'F12': {
        'name': '12-Pile Rectangle 3x4',
        'num_piles': 12,
        'coordinates': [(-1.5, 2.25), (0, 2.25), (1.5, 2.25), (-1.5, 0.75), (0, 0.75), (1.5, 0.75),
                       (-1.5, -0.75), (0, -0.75), (1.5, -0.75), (-1.5, -2.25), (0, -2.25), (1.5, -2.25)],
        'spacing': 1.5
    },
    'F15': {
        'name': '15-Pile Rectangle 3x5',
        'num_piles': 15,
        'coordinates': [(-1.5, 3.0), (0, 3.0), (1.5, 3.0), (-1.5, 1.5), (0, 1.5), (1.5, 1.5),
                       (-1.5, 0), (0, 0), (1.5, 0), (-1.5, -1.5), (0, -1.5), (1.5, -1.5),
                       (-1.5, -3.0), (0, -3.0), (1.5, -3.0)],
        'spacing': 1.5
    },
    'F18': {
        'name': '18-Pile Rectangle 3x6',
        'num_piles': 18,
        'coordinates': [(-1.5, 3.75), (0, 3.75), (1.5, 3.75), (-1.5, 2.25), (0, 2.25), (1.5, 2.25),
                       (-1.5, 0.75), (0, 0.75), (1.5, 0.75), (-1.5, -0.75), (0, -0.75), (1.5, -0.75),
                       (-1.5, -2.25), (0, -2.25), (1.5, -2.25), (-1.5, -3.75), (0, -3.75), (1.5, -3.75)],
        'spacing': 1.5
    },
    'F20': {
        'name': '20-Pile Rectangle 4x5',
        'num_piles': 20,
        'coordinates': [(-2.25, 3.0), (-0.75, 3.0), (0.75, 3.0), (2.25, 3.0),
                       (-2.25, 1.5), (-0.75, 1.5), (0.75, 1.5), (2.25, 1.5),
                       (-2.25, 0), (-0.75, 0), (0.75, 0), (2.25, 0),
                       (-2.25, -1.5), (-0.75, -1.5), (0.75, -1.5), (2.25, -1.5),
                       (-2.25, -3.0), (-0.75, -3.0), (0.75, -3.0), (2.25, -3.0)],
        'spacing': 1.5
    }
}

class PileGroupAnalyzer:
    """Pile group analysis using proper structural engineering theory"""
    
    def __init__(self, pile_diameter=0.6, pile_capacity=120, pile_spacing_factor=2.5):
        self.pile_diameter = pile_diameter
        self.pile_capacity = pile_capacity
        self.min_spacing = pile_spacing_factor * pile_diameter
        
    def calculate_pile_group_properties(self, footing_type):
        """
        Calculate geometric properties of pile group
        
        Theory: For a pile group, the section properties are:
        - Ixx = Σ(xi²) where xi is distance of each pile from Y-axis
        - Iyy = Σ(yi²) where yi is distance of each pile from X-axis
        - Section modulus Zx = Ixx/xmax, Zy = Iyy/ymax
        """
        config = PILE_CONFIGURATIONS[footing_type]
        coords = np.array(config['coordinates']) * config['spacing']
        
        # Calculate centroid (should be at origin for symmetric layouts)
        centroid_x = np.mean(coords[:, 0])
        centroid_y = np.mean(coords[:, 1])
        
        # Adjust coordinates relative to centroid
        x_coords = coords[:, 0] - centroid_x
        y_coords = coords[:, 1] - centroid_y
        
        # Calculate moment of inertia
        Ixx = np.sum(x_coords**2)
        Iyy = np.sum(y_coords**2)
        
        # Maximum distances from centroid
        xmax = np.max(np.abs(x_coords)) if len(x_coords) > 0 else 1
        ymax = np.max(np.abs(y_coords)) if len(y_coords) > 0 else 1
        
        # Section modulus
        Zx = Ixx / xmax if xmax > 0 else float('inf')
        Zy = Iyy / ymax if ymax > 0 else float('inf')
        
        return {
            'Ixx': Ixx,
            'Iyy': Iyy,
            'xmax': xmax,
            'ymax': ymax,
            'Zx': Zx,
            'Zy': Zy,
            'n_piles': config['num_piles'],
            'name': config['name']
        }
    
    def calculate_pile_loads(self, Fz, Mx, My, footing_type):
        """
        Calculate pile loads using structural engineering formula:
        
        P_max = Fz/n + |My|/Zx + |Mx|/Zy
        
        Where:
        - Fz = Vertical load (compression positive)
        - Mx = Moment about X-axis
        - My = Moment about Y-axis
        - n = number of piles
        - Zx, Zy = Section moduli
        """
        props = self.calculate_pile_group_properties(footing_type)
        
        # Component stresses
        axial_stress = abs(Fz) / props['n_piles']
        
        # Moment-induced stresses
        stress_from_My = abs(My) / props['Zx'] if props['Zx'] != float('inf') else 0
        stress_from_Mx = abs(Mx) / props['Zy'] if props['Zy'] != float('inf') else 0
        
        # Maximum pile load (compression)
        max_pile_load = axial_stress + stress_from_My + stress_from_Mx
        
        # Minimum pile load (check for tension)
        min_pile_load = axial_stress - stress_from_My - stress_from_Mx
        
        # Utilization ratio
        utilization = max_pile_load / self.pile_capacity
        
        return {
            'footing_type': footing_type,
            'footing_name': props['name'],
            'n_piles': props['n_piles'],
            'axial_stress': axial_stress,
            'moment_stress_mx': stress_from_Mx,
            'moment_stress_my': stress_from_My,
            'max_pile_load': max_pile_load,
            'min_pile_load': min_pile_load,
            'pile_capacity': self.pile_capacity,
            'utilization_ratio': utilization,
            'is_safe': utilization <= 1.0,
            'has_tension': min_pile_load < 0,
            'Ixx': props['Ixx'],
            'Iyy': props['Iyy'],
            'Zx': props['Zx'],
            'Zy': props['Zy']
        }
    
    def optimize_footing(self, Fz, Mx, My, target_utilization=0.85):
        """Find optimal footing for target utilization"""
        results = []
        
        for footing_type in PILE_CONFIGURATIONS.keys():
            analysis = self.calculate_pile_loads(Fz, Mx, My, footing_type)
            analysis['target_diff'] = abs(analysis['utilization_ratio'] - target_utilization)
            results.append(analysis)
        
        df_results = pd.DataFrame(results)
        
        # Filter safe designs
        safe_designs = df_results[df_results['is_safe']]
        
        if len(safe_designs) > 0:
            # Find closest to target utilization
            optimal_idx = safe_designs['target_diff'].idxmin()
            return safe_designs.loc[optimal_idx].to_dict()
        else:
            # Return design with lowest utilization
            optimal_idx = df_results['utilization_ratio'].idxmin()
            return df_results.loc[optimal_idx].to_dict()

def load_data(uploaded_file):
    """Load CSV file"""
    try:
        return pd.read_csv(uploaded_file), "File loaded successfully"
    except Exception as e:
        return None, f"Error: {str(e)}"

def process_pile_analysis(df, nodes, pile_capacity, target_utilization=0.85):
    """Process pile analysis for all nodes"""
    analyzer = PileGroupAnalyzer(pile_capacity=pile_capacity)
    
    # Filter for selected nodes
    df_filtered = df[df['Node'].isin(nodes)].copy()
    
    results = []
    for idx, row in df_filtered.iterrows():
        # Get loads
        Fz = abs(row.get('FZ (tonf)', row.get('Fz', 0)))
        Mx = abs(row.get('MX (tonf·m)', row.get('Mx', 0)))
        My = abs(row.get('MY (tonf·m)', row.get('My', 0)))
        
        # Optimize footing
        optimal = analyzer.optimize_footing(Fz, Mx, My, target_utilization)
        
        # Add node information
        optimal['Node'] = row['Node']
        optimal['X'] = row.get('X', 0)
        optimal['Y'] = row.get('Y', 0)
        optimal['Z'] = row.get('Z', 0)
        optimal['Fz'] = Fz
        optimal['Mx'] = Mx
        optimal['My'] = My
        optimal['Load_Case'] = row.get('Load Combination', row.get('Load Case', f'LC_{idx}'))
        
        results.append(optimal)
    
    return pd.DataFrame(results)

def categorize_utilization(utilization):
    """Categorize utilization ratio"""
    if utilization < 0.6:
        return "Over-Conservative", "🟢"
    elif utilization < 0.8:
        return "Conservative", "🟡"
    elif utilization <= 0.95:
        return "Optimal", "🔵"
    elif utilization <= 1.0:
        return "Near-Capacity", "🟠"
    else:
        return "Over-Capacity", "🔴"

# Sidebar
st.sidebar.header("📋 Configuration")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

pile_capacity = st.sidebar.number_input(
    "Pile Capacity (tonf)",
    min_value=50,
    max_value=500,
    value=120,
    step=10
)

target_utilization = st.sidebar.slider(
    "Target Utilization",
    0.7, 0.95, 0.85, 0.05
)

# Default nodes
DEFAULT_NODES = [789, 790, 791, 4561, 4572, 4576, 4581, 4586]
selected_nodes = st.sidebar.multiselect(
    "Select Nodes",
    options=DEFAULT_NODES,
    default=DEFAULT_NODES
)

# Main content
if uploaded_file:
    df, message = load_data(uploaded_file)
    
    if df is not None:
        st.success(message)
        
        # Show theory explanation
        with st.expander("📚 Pile Design Theory"):
            st.markdown("""
            ### Pile Group Load Distribution Formula:
            
            **Maximum pile load:** `P_max = Fz/n + |My|/Zx + |Mx|/Zy`
            
            Where:
            - `Fz` = Total vertical load (tonf)
            - `Mx` = Moment about X-axis (tonf·m)
            - `My` = Moment about Y-axis (tonf·m)
            - `n` = Number of piles
            - `Zx` = Section modulus about X-axis = Ixx/xmax
            - `Zy` = Section modulus about Y-axis = Iyy/ymax
            - `Ixx` = Σxi² (moment of inertia)
            - `Iyy` = Σyi² (moment of inertia)
            
            ### Design Criteria:
            - Utilization Ratio = P_max / Pile_Capacity
            - Safe Design: Utilization ≤ 1.0
            - Optimal Range: 0.80 - 0.95
            """)
        
        if st.sidebar.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Analyzing..."):
                results = process_pile_analysis(df, selected_nodes, pile_capacity, target_utilization)
                st.session_state.final_results = results
                st.success("Analysis complete!")
        
        # Display results
        if st.session_state.final_results is not None:
            results = st.session_state.final_results
            
            # Add utilization categories
            results['Category'], results['Icon'] = zip(*results['utilization_ratio'].apply(categorize_utilization))
            
            # Summary metrics
            st.header("📊 Analysis Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Nodes Analyzed", len(results))
            with col2:
                st.metric("Avg Utilization", f"{results['utilization_ratio'].mean():.1%}")
            with col3:
                safe_count = len(results[results['is_safe']])
                st.metric("Safe Designs", f"{safe_count}/{len(results)}")
            with col4:
                optimal = len(results[results['Category'] == 'Optimal'])
                st.metric("Optimal Designs", optimal)
            
            # Theory verification table
            st.header("🔬 Design Verification")
            
            display_cols = ['Node', 'Icon', 'footing_type', 'n_piles', 'Fz', 'Mx', 'My',
                          'axial_stress', 'moment_stress_mx', 'moment_stress_my',
                          'max_pile_load', 'utilization_ratio', 'Category']
            
            display_df = results[display_cols].copy()
            display_df.columns = ['Node', '📊', 'Footing', 'Piles', 'Fz(tonf)', 'Mx(tonf·m)', 'My(tonf·m)',
                                 'P_axial', 'P_Mx', 'P_My', 'P_max', 'Utilization', 'Status']
            
            # Format numeric columns
            for col in ['Fz(tonf)', 'Mx(tonf·m)', 'My(tonf·m)', 'P_axial', 'P_Mx', 'P_My', 'P_max']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}")
            display_df['Utilization'] = display_df['Utilization'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Detailed breakdown for verification
            st.header("🧮 Calculation Breakdown")
            
            selected_node = st.selectbox("Select node for detailed view:", results['Node'].unique())
            node_data = results[results['Node'] == selected_node].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Load Components")
                st.write(f"**Vertical Load (Fz):** {node_data['Fz']:.2f} tonf")
                st.write(f"**Moment X (Mx):** {node_data['Mx']:.2f} tonf·m")
                st.write(f"**Moment Y (My):** {node_data['My']:.2f} tonf·m")
                
                st.subheader("Geometric Properties")
                st.write(f"**Footing:** {node_data['footing_type']} - {node_data['footing_name']}")
                st.write(f"**Number of Piles:** {node_data['n_piles']}")
                st.write(f"**Ixx:** {node_data['Ixx']:.3f} m²")
                st.write(f"**Iyy:** {node_data['Iyy']:.3f} m²")
                st.write(f"**Zx:** {node_data['Zx']:.3f} m³")
                st.write(f"**Zy:** {node_data['Zy']:.3f} m³")
            
            with col2:
                st.subheader("Stress Calculation")
                st.write(f"**P_axial = Fz/n:** {node_data['Fz']:.2f}/{node_data['n_piles']} = {node_data['axial_stress']:.2f} tonf")
                st.write(f"**P_My = My/Zx:** {node_data['My']:.2f}/{node_data['Zx']:.3f} = {node_data['moment_stress_my']:.2f} tonf")
                st.write(f"**P_Mx = Mx/Zy:** {node_data['Mx']:.2f}/{node_data['Zy']:.3f} = {node_data['moment_stress_mx']:.2f} tonf")
                st.write(f"**P_max = P_axial + P_My + P_Mx:** {node_data['max_pile_load']:.2f} tonf")
                
                st.subheader("Design Check")
                st.write(f"**Pile Capacity:** {node_data['pile_capacity']:.2f} tonf")
                st.write(f"**Utilization:** {node_data['utilization_ratio']:.1%}")
                st.write(f"**Status:** {node_data['Category']} {node_data['Icon']}")
                
                if node_data['has_tension']:
                    st.warning(f"⚠️ Minimum pile load: {node_data['min_pile_load']:.2f} tonf (Tension)")
            
            # Visualization
            st.header("📈 Visualizations")
            
            # Utilization distribution
            fig = px.histogram(results, x='utilization_ratio', nbinsx=20,
                             title='Utilization Distribution',
                             labels={'utilization_ratio': 'Utilization Ratio'})
            fig.add_vline(x=0.85, line_dash="dash", line_color="red", annotation_text="Target")
            fig.add_vline(x=1.0, line_dash="solid", line_color="red", annotation_text="Limit")
            st.plotly_chart(fig, use_container_width=True)
            
            # Export
            st.header("💾 Export Results")
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download Results (CSV)",
                data=csv,
                file_name="pile_analysis_results.csv",
                mime="text/csv"
            )
else:
    st.info("👈 Please upload a CSV file to begin analysis")
    
    # Show example calculation
    st.header("📐 Example Calculation")
    
    analyzer = PileGroupAnalyzer()
    example = analyzer.calculate_pile_loads(Fz=400, Mx=80, My=60, footing_type='F6')
    
    st.write("**Example:** Fz=400 tonf, Mx=80 tonf·m, My=60 tonf·m")
    st.write(f"**Footing:** {example['footing_type']} ({example['n_piles']} piles)")
    st.write(f"**P_max:** {example['max_pile_load']:.2f} tonf")
    st.write(f"**Utilization:** {example['utilization_ratio']:.1%}")
    st.write(f"**Status:** {'✅ Safe' if example['is_safe'] else '❌ Unsafe'}")
