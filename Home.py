import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Multi-Foundation Pile Analysis Tool",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .node-assignment {
        background-color: #e8f4fd;
        padding: 0.8rem;
        border-radius: 8px;
        border: 2px solid #4a90e2;
        margin: 0.5rem 0;
    }
    .foundation-group {
        background-color: #fff5ee;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #ff8c00;
        margin: 1rem 0;
    }
    .optimal-design {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .manual-override {
        background-color: #ffefd5;
        border-left: 5px solid #ff6347;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏗️ Multi-Foundation Pile Analysis Tool</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Assign Different Foundation Types to Different Nodes</p>', unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'final_results' not in st.session_state:
    st.session_state.final_results = None
if 'node_foundation_mapping' not in st.session_state:
    st.session_state.node_foundation_mapping = {}
if 'foundation_groups' not in st.session_state:
    st.session_state.foundation_groups = {}
if 'custom_foundations' not in st.session_state:
    st.session_state.custom_foundations = {}
if 'override_mode' not in st.session_state:
    st.session_state.override_mode = False

# Default pile configurations
DEFAULT_PILE_CONFIGURATIONS = {
    'F3': {
        'name': '3-Pile Triangle',
        'num_piles': 3,
        'arrangement': 'Triangular',
        'color': '#FF6B6B',
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
        'color': '#4ECDC4',
        'coordinates': [
            (-1.0, 1.0), (1.0, 1.0),
            (-1.0, -1.0), (1.0, -1.0)
        ]
    },
    'F5': {
        'name': '5-Pile Pentagon',
        'num_piles': 5,
        'arrangement': 'Pentagon',
        'color': '#45B7D1',
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
        'color': '#96CEB4',
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
        'color': '#FFEAA7',
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
        'color': '#DDA0DD',
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
        'color': '#98D8C8',
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
        'color': '#F7DC6F',
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
        'color': '#BB8FCE',
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
        'color': '#85C1E2',
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
        'color': '#F8B739',
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
        'color': '#52BE80',
        'coordinates': [
            (-3.0, 4.0), (-1.0, 4.0), (1.0, 4.0), (3.0, 4.0),
            (-3.0, 2.0), (-1.0, 2.0), (1.0, 2.0), (3.0, 2.0),
            (-3.0, 0), (-1.0, 0), (1.0, 0), (3.0, 0),
            (-3.0, -2.0), (-1.0, -2.0), (1.0, -2.0), (3.0, -2.0),
            (-3.0, -4.0), (-1.0, -4.0), (1.0, -4.0), (3.0, -4.0)
        ]
    }
}

# Default nodes
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
    """Pile group analysis with multi-foundation support"""
    
    def __init__(self, pile_diameter=0.6, pile_capacity=120, pile_spacing=1.5,
                 safety_factor=1.5):
        self.pile_diameter = pile_diameter
        self.pile_capacity = pile_capacity
        self.pile_spacing = pile_spacing
        self.safety_factor = safety_factor
        self.min_spacing = 2.5 * pile_diameter
        
    def calculate_section_properties(self, footing_config, custom_spacing=None):
        """Calculate geometric properties of pile group"""
        if isinstance(footing_config, dict):
            config = footing_config
        else:
            config = DEFAULT_PILE_CONFIGURATIONS.get(footing_config, None)
            if config is None:
                raise ValueError(f"Unknown footing type: {footing_config}")
        
        spacing = custom_spacing if custom_spacing else self.pile_spacing
        coords = np.array(config['coordinates']) * spacing
        
        centroid_x = np.mean(coords[:, 0])
        centroid_y = np.mean(coords[:, 1])
        
        x_coords = coords[:, 0] - centroid_x
        y_coords = coords[:, 1] - centroid_y
        
        Ixx = np.sum(x_coords**2)
        Iyy = np.sum(y_coords**2)
        
        xmax = np.max(np.abs(x_coords)) if len(x_coords) > 0 else 1.0
        ymax = np.max(np.abs(y_coords)) if len(y_coords) > 0 else 1.0
        
        Zx = Ixx / xmax if xmax > 0 else float('inf')
        Zy = Iyy / ymax if ymax > 0 else float('inf')
        
        return {
            'Ixx': Ixx,
            'Iyy': Iyy,
            'xmax': xmax,
            'ymax': ymax,
            'Zx': Zx,
            'Zy': Zy,
            'n_piles': len(coords),
            'name': config.get('name', 'Custom'),
            'arrangement': config.get('arrangement', 'Custom'),
            'coordinates': coords.tolist(),
            'color': config.get('color', '#808080')
        }
    
    def calculate_pile_loads(self, Fz, Mx, My, footing_config):
        """Calculate maximum pile load"""
        props = self.calculate_section_properties(footing_config)
        
        axial_stress = abs(Fz) / props['n_piles']
        
        if props['Zx'] != float('inf'):
            stress_from_My = abs(My) / props['Zx']
        else:
            stress_from_My = 0
            
        if props['Zy'] != float('inf'):
            stress_from_Mx = abs(Mx) / props['Zy']
        else:
            stress_from_Mx = 0
        
        max_pile_load = axial_stress + stress_from_My + stress_from_Mx
        min_pile_load = axial_stress - stress_from_My - stress_from_Mx
        
        utilization = max_pile_load / self.pile_capacity
        
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
            'utilization_ratio': utilization,
            'is_safe': utilization <= 1.0,
            'has_tension': min_pile_load < 0,
            'category': category,
            'category_color': category_color,
            'foundation_color': props['color'],
            'Ixx': props['Ixx'],
            'Iyy': props['Iyy'],
            'Zx': props['Zx'],
            'Zy': props['Zy'],
            'coordinates': props['coordinates']
        }
    
    def find_optimal_footing(self, Fz, Mx, My, target_utilization=0.85, 
                            allowed_foundations=None):
        """Find optimal footing from allowed list"""
        if allowed_foundations is None:
            allowed_foundations = list(DEFAULT_PILE_CONFIGURATIONS.keys())
        
        results = []
        for footing_type in allowed_foundations:
            if footing_type in DEFAULT_PILE_CONFIGURATIONS:
                analysis = self.calculate_pile_loads(Fz, Mx, My, footing_type)
                analysis['footing_key'] = footing_type
                analysis['target_diff'] = abs(analysis['utilization_ratio'] - target_utilization)
                results.append(analysis)
        
        if not results:
            return None
        
        df_results = pd.DataFrame(results)
        safe_designs = df_results[df_results['is_safe']]
        
        if len(safe_designs) > 0:
            optimal_idx = safe_designs['target_diff'].idxmin()
            return safe_designs.loc[optimal_idx].to_dict()
        else:
            optimal_idx = df_results['utilization_ratio'].idxmin()
            return df_results.loc[optimal_idx].to_dict()

def create_foundation_assignment_interface():
    """Create interface for assigning foundations to nodes"""
    st.markdown("### 🎯 Foundation Assignment Methods")
    
    # This function is now integrated directly in tab1
    # Return empty since the interface is created in the main tab
    return None
    
    assignment_method = st.radio(
        "Choose assignment method:",
        ["Automatic Optimization", "Manual Assignment", "Group Assignment", "Load-Based Rules"]
    )
    
    if assignment_method == "Automatic Optimization":
        st.markdown("""
        <div class="optimal-design">
        <strong>🤖 Automatic Optimization</strong><br>
        System will automatically select the optimal foundation for each node based on:
        <ul>
        <li>Load requirements (Fz, Mx, My)</li>
        <li>Target utilization ratio</li>
        <li>Safety factors</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Allowed foundations for optimization
        st.markdown("#### Allowed Foundation Types")
        allowed_foundations = st.multiselect(
            "Select foundations to consider:",
            options=list(DEFAULT_PILE_CONFIGURATIONS.keys()),
            default=list(DEFAULT_PILE_CONFIGURATIONS.keys())
        )
        st.session_state['allowed_foundations'] = allowed_foundations
        
    elif assignment_method == "Manual Assignment":
        st.markdown("""
        <div class="manual-override">
        <strong>✏️ Manual Assignment</strong><br>
        Manually assign specific foundation types to specific nodes.
        </div>
        """, unsafe_allow_html=True)
        
        # Create manual assignment interface
        if 'temp_assignments' not in st.session_state:
            st.session_state.temp_assignments = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_nodes_for_assignment = st.multiselect(
                "Select nodes:",
                options=DEFAULT_NODES,
                key="manual_nodes"
            )
        
        with col2:
            foundation_for_nodes = st.selectbox(
                "Assign foundation type:",
                options=list(DEFAULT_PILE_CONFIGURATIONS.keys()),
                key="manual_foundation"
            )
        
        if st.button("➕ Add Assignment"):
            for node in selected_nodes_for_assignment:
                st.session_state.temp_assignments[node] = foundation_for_nodes
            st.success(f"Assigned {foundation_for_nodes} to {len(selected_nodes_for_assignment)} nodes")
        
        # Display current assignments
        if st.session_state.temp_assignments:
            st.markdown("#### Current Assignments")
            assignments_df = pd.DataFrame([
                {"Node": node, "Foundation": found}
                for node, found in st.session_state.temp_assignments.items()
            ])
            st.dataframe(assignments_df, use_container_width=True)
            
            if st.button("🗑️ Clear All Assignments"):
                st.session_state.temp_assignments = {}
                st.rerun()
        
        # Save to main mapping
        if st.button("💾 Save Manual Assignments", type="primary"):
            st.session_state.node_foundation_mapping = st.session_state.temp_assignments.copy()
            st.session_state.override_mode = True
            st.success("Manual assignments saved!")
    
    elif assignment_method == "Group Assignment":
        st.markdown("""
        <div class="foundation-group">
        <strong>👥 Group Assignment</strong><br>
        Create groups of nodes and assign the same foundation to each group.
        </div>
        """, unsafe_allow_html=True)
        
        # Create groups
        group_name = st.text_input("Group Name", "Group_1")
        
        col1, col2 = st.columns(2)
        
        with col1:
            group_nodes = st.multiselect(
                "Select nodes for this group:",
                options=DEFAULT_NODES,
                key="group_nodes"
            )
        
        with col2:
            group_foundation = st.selectbox(
                "Foundation for this group:",
                options=list(DEFAULT_PILE_CONFIGURATIONS.keys()),
                key="group_foundation"
            )
        
        if st.button("➕ Create Group"):
            if group_name not in st.session_state.foundation_groups:
                st.session_state.foundation_groups[group_name] = {}
            st.session_state.foundation_groups[group_name] = {
                'nodes': group_nodes,
                'foundation': group_foundation
            }
            st.success(f"Created group '{group_name}' with {len(group_nodes)} nodes")
        
        # Display groups
        if st.session_state.foundation_groups:
            st.markdown("#### Foundation Groups")
            for gname, gdata in st.session_state.foundation_groups.items():
                with st.expander(f"{gname} - {gdata['foundation']} ({len(gdata['nodes'])} nodes)"):
                    st.write(f"Foundation: **{gdata['foundation']}**")
                    st.write(f"Nodes: {gdata['nodes']}")
                    if st.button(f"Delete {gname}", key=f"del_{gname}"):
                        del st.session_state.foundation_groups[gname]
                        st.rerun()
        
        # Apply groups
        if st.button("✅ Apply Group Assignments", type="primary"):
            mapping = {}
            for gdata in st.session_state.foundation_groups.values():
                for node in gdata['nodes']:
                    mapping[node] = gdata['foundation']
            st.session_state.node_foundation_mapping = mapping
            st.session_state.override_mode = True
            st.success(f"Applied assignments for {len(mapping)} nodes")
    
    else:  # Load-Based Rules
        st.markdown("""
        <div class="node-assignment">
        <strong>📊 Load-Based Rules</strong><br>
        Automatically assign foundations based on load ranges.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Define Load Rules")
        
        # Create rules
        if 'load_rules' not in st.session_state:
            st.session_state.load_rules = []
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            min_load = st.number_input("Min Fz (tonf)", 0, 1000, 0)
        with col2:
            max_load = st.number_input("Max Fz (tonf)", 0, 1000, 500)
        with col3:
            rule_foundation = st.selectbox(
                "Use foundation:",
                options=list(DEFAULT_PILE_CONFIGURATIONS.keys()),
                key="rule_foundation"
            )
        with col4:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("➕ Add Rule"):
                st.session_state.load_rules.append({
                    'min': min_load,
                    'max': max_load,
                    'foundation': rule_foundation
                })
        
        # Display rules
        if st.session_state.load_rules:
            st.markdown("#### Current Rules")
            rules_df = pd.DataFrame(st.session_state.load_rules)
            st.dataframe(rules_df, use_container_width=True)
            
            if st.button("🗑️ Clear Rules"):
                st.session_state.load_rules = []
                st.rerun()
    
    return assignment_method

def process_multi_foundation_analysis(df, nodes, analyzer, assignment_method, 
                                     target_utilization=0.85):
    """Process analysis with multiple foundation types including custom"""
    
    df_filtered = df[df['Node'].isin(nodes)].copy()
    
    if len(df_filtered) == 0:
        return None
    
    # Get all available foundations (default + custom)
    all_foundations = {**DEFAULT_PILE_CONFIGURATIONS, **st.session_state.custom_pile_configs}
    
    all_results = []
    
    for idx, row in df_filtered.iterrows():
        node = row['Node']
        Fz = abs(row.get('FZ (tonf)', row.get('Fz', 0)))
        Mx = abs(row.get('MX (tonf·m)', row.get('Mx', 0)))
        My = abs(row.get('MY (tonf·m)', row.get('My', 0)))
        
        # Determine foundation based on assignment method
        if assignment_method == "Manual Assignment" and hasattr(st.session_state, 'temp_assignments') and node in st.session_state.temp_assignments:
            # Use manually assigned foundation
            footing_type = st.session_state.temp_assignments[node]
            if footing_type in all_foundations:
                result = analyzer.calculate_pile_loads(Fz, Mx, My, footing_type)
                result['assignment_method'] = 'Manual'
            else:
                # Fallback to optimization if foundation not found
                allowed = st.session_state.get('allowed_foundations', list(all_foundations.keys()))
                result = analyzer.find_optimal_footing(Fz, Mx, My, target_utilization, allowed)
                result['assignment_method'] = 'Auto-Fallback'
        else:
            # Automatic optimization with allowed foundations
            allowed = st.session_state.get('allowed_foundations', list(DEFAULT_PILE_CONFIGURATIONS.keys()))
            # Create dictionary of allowed foundations
            allowed_configs = {k: all_foundations[k] for k in allowed if k in all_foundations}
            
            results = []
            for footing_key, footing_config in allowed_configs.items():
                analysis = analyzer.calculate_pile_loads(Fz, Mx, My, footing_key)
                analysis['footing_key'] = footing_key
                analysis['target_diff'] = abs(analysis['utilization_ratio'] - target_utilization)
                results.append(analysis)
            
            if results:
                # Find optimal
                df_results = pd.DataFrame(results)
                safe_designs = df_results[df_results['is_safe']]
                
                if len(safe_designs) > 0:
                    optimal_idx = safe_designs['target_diff'].idxmin()
                    result = safe_designs.loc[optimal_idx].to_dict()
                else:
                    optimal_idx = df_results['utilization_ratio'].idxmin()
                    result = df_results.loc[optimal_idx].to_dict()
                
                result['assignment_method'] = 'Auto-Optimized'
            else:
                # No valid foundations, use default
                result = analyzer.calculate_pile_loads(Fz, Mx, My, 'F4')
                result['assignment_method'] = 'Default-F4'
        
        # Add node information
        result['Node'] = node
        result['Load_Case'] = row.get('Load Combination', row.get('Load Case', f'LC_{idx}'))
        result['X'] = row.get('X', 0)
        result['Y'] = row.get('Y', 0)
        result['Z'] = row.get('Z', 0)
        result['Fz'] = Fz
        result['Mx'] = Mx
        result['My'] = My
        result['Target_Utilization'] = target_utilization
        
        all_results.append(result)
    
    return pd.DataFrame(all_results)state.node_foundation_mapping:
            # Use group-assigned foundation
            footing_type = st.session_state.node_foundation_mapping[node]
            result = analyzer.calculate_pile_loads(Fz, Mx, My, footing_type)
            result['assignment_method'] = 'Group'
        
        elif assignment_method == "Load-Based Rules" and st.session_state.load_rules:
            # Apply load-based rules
            footing_type = None
            for rule in st.session_state.load_rules:
                if rule['min'] <= Fz <= rule['max']:
                    footing_type = rule['foundation']
                    break
            
            if footing_type:
                result = analyzer.calculate_pile_loads(Fz, Mx, My, footing_type)
                result['assignment_method'] = 'Rule-Based'
            else:
                # No rule matched, use optimization
                result = analyzer.find_optimal_footing(
                    Fz, Mx, My, target_utilization,
                    st.session_state.get('allowed_foundations', None)
                )
                result['assignment_method'] = 'Auto-Optimized'
        
        else:
            # Automatic optimization
            result = analyzer.find_optimal_footing(
                Fz, Mx, My, target_utilization,
                st.session_state.get('allowed_foundations', None)
            )
            result['assignment_method'] = 'Auto-Optimized'
        
        # Add node information
        result['Node'] = node
        result['Load_Case'] = row.get('Load Combination', row.get('Load Case', f'LC_{idx}'))
        result['X'] = row.get('X', 0)
        result['Y'] = row.get('Y', 0)
        result['Z'] = row.get('Z', 0)
        result['Fz'] = Fz
        result['Mx'] = Mx
        result['My'] = My
        result['Target_Utilization'] = target_utilization
        
        all_results.append(result)
    
    return pd.DataFrame(all_results)

def create_foundation_summary_visualization(results_df):
    """Create visualization showing foundation distribution"""
    
    # Count foundations by type
    foundation_counts = results_df['footing_type'].value_counts()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Foundation Type Distribution', 'Utilization by Foundation',
                       'Site Plan with Foundations', 'Foundation Efficiency'),
        specs=[[{'type': 'pie'}, {'type': 'box'}],
               [{'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    # 1. Pie chart of foundation types
    fig.add_trace(
        go.Pie(
            labels=foundation_counts.index,
            values=foundation_counts.values,
            marker=dict(colors=[DEFAULT_PILE_CONFIGURATIONS[f]['color'] 
                              for f in foundation_counts.index if f in DEFAULT_PILE_CONFIGURATIONS]),
            textinfo='label+percent',
            hovertemplate='%{label}: %{value} nodes<br>%{percent}'
        ),
        row=1, col=1
    )
    
    # 2. Box plot of utilization by foundation
    for foundation in results_df['footing_type'].unique():
        data = results_df[results_df['footing_type'] == foundation]
        color = DEFAULT_PILE_CONFIGURATIONS.get(foundation, {}).get('color', '#808080')
        fig.add_trace(
            go.Box(
                y=data['utilization_ratio'],
                name=foundation,
                marker_color=color,
                boxmean='sd'
            ),
            row=1, col=2
        )
    
    # 3. Site plan with foundations (if coordinates exist)
    if 'X' in results_df.columns and 'Y' in results_df.columns:
        for foundation in results_df['footing_type'].unique():
            data = results_df[results_df['footing_type'] == foundation]
            color = DEFAULT_PILE_CONFIGURATIONS.get(foundation, {}).get('color', '#808080')
            fig.add_trace(
                go.Scatter(
                    x=data['X'],
                    y=data['Y'],
                    mode='markers',
                    name=foundation,
                    marker=dict(
                        size=data['n_piles']*2,
                        color=color,
                        line=dict(color='white', width=1),
                        opacity=0.8
                    ),
                    text=[f"Node {n}<br>{f}" for n, f in zip(data['Node'], data['footing_type'])],
                    hovertemplate='%{text}<br>X: %{x:.1f}<br>Y: %{y:.1f}'
                ),
                row=2, col=1
            )
    
    # 4. Bar chart of average utilization by foundation
    avg_utilization = results_df.groupby('footing_type')['utilization_ratio'].mean().sort_values()
    colors = [DEFAULT_PILE_CONFIGURATIONS.get(f, {}).get('color', '#808080') for f in avg_utilization.index]
    
    fig.add_trace(
        go.Bar(
            x=avg_utilization.index,
            y=avg_utilization.values,
            marker_color=colors,
            text=[f"{v:.1%}" for v in avg_utilization.values],
            textposition='outside',
            hovertemplate='%{x}: %{y:.2%}'
        ),
        row=2, col=2
    )
    
    # Add target utilization line
    fig.add_hline(y=0.85, line_dash="dash", line_color="red", 
                  annotation_text="Target", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Multi-Foundation Analysis Summary",
        title_x=0.5
    )
    
    # Update axes
    fig.update_xaxes(title_text="X Coordinate (m)", row=2, col=1)
    fig.update_yaxes(title_text="Y Coordinate (m)", row=2, col=1)
    fig.update_xaxes(title_text="Foundation Type", row=2, col=2)
    fig.update_yaxes(title_text="Average Utilization", row=2, col=2)
    fig.update_yaxes(title_text="Utilization Ratio", row=1, col=2)
    
    return fig

def create_foundation_comparison_table(results_df):
    """Create comparison table of different foundations including custom"""
    
    all_foundations = {**DEFAULT_PILE_CONFIGURATIONS, **st.session_state.custom_pile_configs}
    comparison_data = []
    
    for foundation in results_df['footing_type'].unique():
        data = results_df[results_df['footing_type'] == foundation]
        
        if foundation in all_foundations:
            config = all_foundations[foundation]
            source = "Custom" if foundation in st.session_state.custom_pile_configs else "Default"
        else:
            config = {'name': 'Unknown', 'num_piles': 0}
            source = "Unknown"
        
        comparison_data.append({
            'Foundation': foundation,
            'Source': source,
            'Name': config.get('name', 'Unknown'),
            'Piles': config.get('num_piles', 0),
            'Nodes Using': len(data),
            'Avg Utilization': f"{data['utilization_ratio'].mean():.1%}",
            'Max Utilization': f"{data['utilization_ratio'].max():.1%}",
            'Min Utilization': f"{data['utilization_ratio'].min():.1%}",
            'Safe Designs': f"{len(data[data['is_safe']])}/{len(data)}",
            'Total Piles': len(data) * config.get('num_piles', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Nodes Using', ascending=False)
    
    return comparison_df
        
        comparison_data.append({
            'Foundation': foundation,
            'Name': config.get('name', 'Unknown'),
            'Piles': config.get('num_piles', 0),
            'Nodes Using': len(data),
            'Avg Utilization': f"{data['utilization_ratio'].mean():.1%}",
            'Max Utilization': f"{data['utilization_ratio'].max():.1%}",
            'Min Utilization': f"{data['utilization_ratio'].min():.1%}",
            'Safe Designs': f"{len(data[data['is_safe']])}/{len(data)}",
            'Total Piles': len(data) * config.get('num_piles', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Nodes Using', ascending=False)
    
    return comparison_df

def load_data(uploaded_file):
    """Load CSV file"""
    try:
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(uploaded_file, encoding=encoding)
                return df, f"Successfully loaded with {encoding} encoding"
            except UnicodeDecodeError:
                continue
        return None, "Could not decode file"
    except Exception as e:
        return None, f"Error: {str(e)}"

# ===================== SIDEBAR =====================
st.sidebar.title("📋 Configuration")

uploaded_file = st.sidebar.file_uploader(
    "📁 Upload CSV File",
    type=['csv']
)

st.sidebar.subheader("🔧 Pile Parameters")

pile_diameter = st.sidebar.number_input(
    "Pile Diameter (m)",
    0.3, 2.0, 0.6, 0.1
)

pile_capacity = st.sidebar.number_input(
    "Pile Capacity (tonf)",
    50, 500, 120, 10
)

pile_spacing = st.sidebar.number_input(
    "Pile Spacing (m)",
    1.0, 5.0, 1.5, 0.1
)

safety_factor = st.sidebar.number_input(
    "Safety Factor",
    1.0, 3.0, 1.5, 0.1
)

target_utilization = st.sidebar.slider(
    "Target Utilization",
    0.70, 0.95, 0.85, 0.05
)

# Check minimum spacing
min_spacing = 2.5 * pile_diameter
if pile_spacing < min_spacing:
    st.sidebar.warning(f"⚠️ Spacing should be ≥ {min_spacing:.1f}m")

st.sidebar.subheader("🎯 Node Selection")
use_default_nodes = st.sidebar.checkbox("Use Default Nodes", value=True)

if use_default_nodes:
    selected_nodes = DEFAULT_NODES
    st.sidebar.info(f"Using {len(DEFAULT_NODES)} nodes")
else:
    nodes_input = st.sidebar.text_area(
        "Enter Node Numbers",
        value=",".join(map(str, DEFAULT_NODES[:5]))
    )
    try:
        selected_nodes = [int(x.strip()) for x in nodes_input.split(",") if x.strip()]
        st.sidebar.success(f"Selected {len(selected_nodes)} nodes")
    except:
        st.sidebar.error("Invalid format")
        selected_nodes = DEFAULT_NODES[:5]

# ===================== MAIN TABS =====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏗️ Foundation Assignment",
    "📊 Data Input",
    "🔬 Analysis Results",
    "📈 Visualizations",
    "📋 Comparison",
    "🧮 Details",
    "💾 Export"
])

with tab1:
    st.markdown('<h2 class="section-header">🏗️ Multi-Foundation Assignment</h2>', unsafe_allow_html=True)
    
    # Foundation library display
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📚 Available Foundations")
        
        for ftype, config in DEFAULT_PILE_CONFIGURATIONS.items():
            st.markdown(f"""
            <div class="foundation-card" style="border-left-color: {config['color']}">
            <strong>{ftype}</strong>: {config['name']}<br>
            <small>• {config['num_piles']} piles<br>
            • {config['arrangement']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Assignment Configuration")
        assignment_method = create_foundation_assignment_interface()
        st.session_state['assignment_method'] = assignment_method

with tab2:
    st.markdown('<h2 class="section-header">📊 Data Input</h2>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        df, message = load_data(uploaded_file)
        
        if df is not None:
            st.success(message)
            
            # Data overview
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
            
            # Preview
            with st.expander("👁️ Preview Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Run analysis button
            if st.button("🚀 Run Multi-Foundation Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing with multiple foundation types..."):
                    try:
                        # Create analyzer
                        analyzer = PileGroupAnalyzer(
                            pile_diameter=pile_diameter,
                            pile_capacity=pile_capacity,
                            pile_spacing=pile_spacing,
                            safety_factor=safety_factor
                        )
                        
                        # Process with multi-foundation support
                        results = process_multi_foundation_analysis(
                            df, 
                            selected_nodes,
                            analyzer,
                            st.session_state.get('assignment_method', 'Automatic Optimization'),
                            target_utilization
                        )
                        
                        if results is not None and len(results) > 0:
                            st.session_state.final_results = results
                            st.success(f"✅ Analysis completed! Used {results['footing_type'].nunique()} different foundation types")
                            st.balloons()
                        else:
                            st.error("No data found")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.error(message)
    else:
        st.info("👈 Please upload a CSV file")
        
        # Show example format
        st.subheader("Expected Format")
        example_df = pd.DataFrame({
            'Node': [789, 790, 791],
            'X': [0, 10, 20],
            'Y': [0, 0, 0],
            'FZ (tonf)': [300, 450, 600],
            'MX (tonf·m)': [50, 80, 100],
            'MY (tonf·m)': [40, 60, 80]
        })
        st.dataframe(example_df)

with tab3:
    st.markdown('<h2 class="section-header">🔬 Analysis Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.final_results is not None:
        results = st.session_state.final_results
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Nodes", len(results))
        with col2:
            st.metric("Foundation Types", results['footing_type'].nunique())
        with col3:
            st.metric("Avg Utilization", f"{results['utilization_ratio'].mean():.1%}")
        with col4:
            safe_count = len(results[results['is_safe']])
            st.metric("Safe Designs", f"{safe_count}/{len(results)}")
        with col5:
            total_piles = results['n_piles'].sum()
            st.metric("Total Piles", int(total_piles))
        
        # Foundation distribution
        st.subheader("🏗️ Foundation Type Distribution")
        
        foundation_summary = results.groupby('footing_type').agg({
            'Node': 'count',
            'utilization_ratio': ['mean', 'max', 'min'],
            'n_piles': 'first',
            'assignment_method': 'first'
        }).round(3)
        
        foundation_summary.columns = ['Count', 'Avg Util', 'Max Util', 'Min Util', 'Piles/Node', 'Method']
        st.dataframe(foundation_summary, use_container_width=True)
        
        # Detailed results table
        st.subheader("📋 Detailed Results")
        
        # Add color coding
        def highlight_utilization(val):
            if val > 1.0:
                return 'background-color: #ffcccc'
            elif val > 0.95:
                return 'background-color: #ffe6cc'
            elif val > 0.80:
                return 'background-color: #ccffcc'
            elif val > 0.60:
                return 'background-color: #ffffcc'
            else:
                return 'background-color: #e6e6e6'
        
        display_cols = ['Node', 'footing_type', 'n_piles', 'Fz', 'Mx', 'My',
                       'max_pile_load', 'utilization_ratio', 'category', 
                       'assignment_method', 'is_safe']
        
        styled_df = results[display_cols].style.applymap(
            highlight_utilization, 
            subset=['utilization_ratio']
        )
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
    else:
        st.info("No results yet. Please run analysis.")

with tab4:
    st.markdown('<h2 class="section-header">📈 Visualizations</h2>', unsafe_allow_html=True)
    
    if st.session_state.final_results is not None:
        results = st.session_state.final_results
        
        # Create comprehensive visualization
        fig = create_foundation_summary_visualization(results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional visualizations
        st.subheader("🎨 Foundation Color Map")
        
        if 'X' in results.columns and 'Y' in results.columns:
            fig_map = go.Figure()
            
            for foundation in results['footing_type'].unique():
                data = results[results['footing_type'] == foundation]
                config = DEFAULT_PILE_CONFIGURATIONS.get(foundation, {})
                
                fig_map.add_trace(go.Scatter(
                    x=data['X'],
                    y=data['Y'],
                    mode='markers+text',
                    name=foundation,
                    marker=dict(
                        size=15,
                        color=config.get('color', '#808080'),
                        symbol='square',
                        line=dict(color='white', width=2)
                    ),
                    text=data['Node'],
                    textposition='top center',
                    textfont=dict(size=8),
                    hovertemplate='Node: %{text}<br>Foundation: ' + foundation + '<br>X: %{x:.1f}<br>Y: %{y:.1f}'
                ))
            
            fig_map.update_layout(
                title='Site Plan - Foundation Type Distribution',
                xaxis_title='X Coordinate (m)',
                yaxis_title='Y Coordinate (m)',
                height=600,
                hovermode='closest'
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
        
        # Utilization heatmap
        st.subheader("🔥 Utilization Heatmap")
        
        pivot_data = results.pivot_table(
            values='utilization_ratio',
            index='footing_type',
            columns='category',
            aggfunc='count',
            fill_value=0
        )
        
        fig_heat = px.imshow(
            pivot_data,
            labels=dict(x="Category", y="Foundation Type", color="Count"),
            color_continuous_scale='RdYlGn_r',
            title="Foundation vs Utilization Category Distribution"
        )
        
        st.plotly_chart(fig_heat, use_container_width=True)
        
    else:
        st.info("No results to visualize")

with tab5:
    st.markdown('<h2 class="section-header">📋 Foundation Comparison</h2>', unsafe_allow_html=True)
    
    if st.session_state.final_results is not None:
        results = st.session_state.final_results
        
        # Create comparison table
        comparison_df = create_foundation_comparison_table(results)
        
        st.subheader("📊 Foundation Performance Comparison")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Cost estimation (simplified)
        st.subheader("💰 Cost Estimation (Simplified)")
        
        pile_unit_cost = st.number_input("Cost per pile (currency units)", 1000, 10000, 5000, 100)
        
        cost_data = []
        for _, row in comparison_df.iterrows():
            cost_data.append({
                'Foundation': row['Foundation'],
                'Total Piles': row['Total Piles'],
                'Estimated Cost': row['Total Piles'] * pile_unit_cost,
                'Cost per Node': row['Total Piles'] * pile_unit_cost / row['Nodes Using'] if row['Nodes Using'] > 0 else 0
            })
        
        cost_df = pd.DataFrame(cost_data)
        cost_df['Estimated Cost'] = cost_df['Estimated Cost'].apply(lambda x: f"{x:,.0f}")
        cost_df['Cost per Node'] = cost_df['Cost per Node'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(cost_df, use_container_width=True)
        
        # Total project cost
        total_cost = sum([row['Total Piles'] * pile_unit_cost for _, row in comparison_df.iterrows()])
        st.success(f"💵 Total Estimated Project Cost: {total_cost:,.0f} currency units")
        
    else:
        st.info("No results for comparison")

with tab6:
    st.markdown('<h2 class="section-header">🧮 Detailed Calculations</h2>', unsafe_allow_html=True)
    
    if st.session_state.final_results is not None:
        results = st.session_state.final_results
        
        # Node selector
        selected_node = st.selectbox(
            "Select node for detailed view:",
            results['Node'].unique()
        )
        
        node_data = results[results['Node'] == selected_node].iloc[0]
        
        # Display detailed calculation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📥 Loads")
            st.write(f"**Fz:** {node_data['Fz']:.2f} tonf")
            st.write(f"**Mx:** {node_data['Mx']:.2f} tonf·m")
            st.write(f"**My:** {node_data['My']:.2f} tonf·m")
            
            st.markdown("### 🏗️ Foundation")
            st.write(f"**Type:** {node_data['footing_type']}")
            st.write(f"**Name:** {node_data['footing_name']}")
            st.write(f"**Piles:** {node_data['n_piles']}")
            st.write(f"**Method:** {node_data['assignment_method']}")
        
        with col2:
            st.markdown("### 📐 Properties")
            st.write(f"**Ixx:** {node_data['Ixx']:.3f} m²")
            st.write(f"**Iyy:** {node_data['Iyy']:.3f} m²")
            st.write(f"**Zx:** {node_data['Zx']:.3f} m³")
            st.write(f"**Zy:** {node_data['Zy']:.3f} m³")
        
        with col3:
            st.markdown("### 📊 Results")
            st.write(f"**P_axial:** {node_data['axial_stress']:.2f} tonf")
            st.write(f"**P_Mx:** {node_data['moment_stress_mx']:.2f} tonf")
            st.write(f"**P_My:** {node_data['moment_stress_my']:.2f} tonf")
            st.write(f"**P_max:** {node_data['max_pile_load']:.2f} tonf")
            st.write(f"**Utilization:** {node_data['utilization_ratio']:.1%}")
            
            if node_data['is_safe']:
                st.success("✅ SAFE")
            else:
                st.error("❌ UNSAFE")
    else:
        st.info("No results available")

with tab7:
    st.markdown('<h2 class="section-header">💾 Export Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.final_results is not None:
        results = st.session_state.final_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export results
            csv = results.to_csv(index=False)
            st.download_button(
                "📥 Download Results (CSV)",
                data=csv,
                file_name=f"multi_foundation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Export foundation mapping
            if st.session_state.node_foundation_mapping:
                mapping_df = pd.DataFrame([
                    {'Node': k, 'Foundation': v}
                    for k, v in st.session_state.node_foundation_mapping.items()
                ])
                mapping_csv = mapping_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Foundation Mapping",
                    data=mapping_csv,
                    file_name=f"foundation_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            # Export summary report
            report = f"""# Multi-Foundation Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- Pile Diameter: {pile_diameter} m
- Pile Capacity: {pile_capacity} tonf
- Pile Spacing: {pile_spacing} m
- Safety Factor: {safety_factor}
- Target Utilization: {target_utilization:.0%}

## Results Summary
- Total Nodes: {len(results)}
- Foundation Types Used: {results['footing_type'].nunique()}
- Average Utilization: {results['utilization_ratio'].mean():.1%}
- Safe Designs: {len(results[results['is_safe']])} / {len(results)}
- Total Piles: {results['n_piles'].sum()}

## Foundation Distribution
{results['footing_type'].value_counts().to_string()}

## Assignment Method
{results['assignment_method'].value_counts().to_string()}
"""
            
            st.download_button(
                "📄 Download Report (MD)",
                data=report,
                file_name=f"multi_foundation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        st.success("✅ All export options ready!")
    else:
        st.info("No results to export")
