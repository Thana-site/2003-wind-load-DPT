import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Page configuration
st.set_page_config(
    page_title="Enhanced Pile Analysis Tool",
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
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .optimal-node {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .conservative-node {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .critical-node {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏗️ Enhanced Pile Foundation Analysis Tool</h1>', unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'final_results' not in st.session_state:
    st.session_state.final_results = None

# Sidebar
st.sidebar.title("📋 Configuration")

# Default node list
DEFAULT_NODES = [26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                46,47,48,49,50,
                725,726,727,728,729,733,734,735,737,739,740,741,
                744,745,746,747,748,753,754,755,756,757,758,759,
                760,761,762,763,764,765,766,767,768,769,770,771,
                772,773,774,775,776,777,778,779,780,781,782,783,
                784,785,787,788,
                7489,7491,7493,7495]

# Footing type factors
FOOTING_FACTORS = {
    "Footing Type": ["F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F12", "F15", "F18", "F20"],
    "Num_Piles": [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20],
    "S_Fac_X": [1.000000, 0.277777778, 0.138888889, 0.138888889, 0.138888889, 0.138888889, 0.138888889, 0.007716049, 0.005, 0.003, 0.002, 0.001],
    "S_Fac_Y": [1.000000, 0.277777778, 0.138888889, 0.138888889, 0.138888889, 0.138888889, 0.138888889, 0.011574074, 0.008, 0.005, 0.003, 0.002],
    "I_Fac_X": [1.000000, 0.555555556, 0.555555556, 0.555555556, 0.277777778, 0.277777778, 0.277777778, 0.034293553, 0.025, 0.015, 0.010, 0.008],
    "I_Fac_Y": [1.000000, 0.555555556, 0.555555556, 0.277777778, 0.277777778, 0.277777778, 0.277777778, 0.173611111, 0.120, 0.080, 0.050, 0.040],
}

def load_data(uploaded_file):
    """Load and process uploaded CSV file"""
    try:
        for encoding in ['latin-1', 'cp1252', 'utf-8']:
            try:
                df = pd.read_csv(uploaded_file, encoding=encoding)
                return df, f"Successfully loaded with {encoding} encoding"
            except UnicodeDecodeError:
                continue
        return None, "Could not decode file with any encoding"
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

def extract_footing_number(footing_type):
    """Extract number from footing type (e.g., 'F5' -> 5)"""
    try:
        if pd.isna(footing_type):
            return 0
        return int(str(footing_type).replace('F', ''))
    except:
        return 0

def optimize_footing_for_target_utilization(row, pile_type, pile_capacity, df_pile, target_utilization=0.85):
    """
    Enhanced algorithm to optimize footing selection for target utilization (80-90%)
    """
    
    # Start with minimum required piles (conservative estimate)
    min_piles = max(3, int(np.ceil(abs(row['Fz']) / pile_capacity)))
    
    best_footing = None
    best_utilization = 0
    best_analysis = None
    
    # Try different footing configurations
    for _, footing_row in df_pile.iterrows():
        num_piles = footing_row['Num_Piles']
        footing_type = footing_row['Footing Type']
        
        # Skip if less than minimum required
        if num_piles < min_piles:
            continue
            
        # Calculate stress components based on pile type
        if pile_type == "Spun Pile 600":
            mx_stress = abs(row['Mx']) * footing_row['S_Fac_X']
            my_stress = abs(row['My']) * footing_row['S_Fac_Y']
        else:  # PC I 300
            mx_stress = abs(row['Mx']) * footing_row['I_Fac_X']
            my_stress = abs(row['My']) * footing_row['I_Fac_Y']
        
        # Calculate total stress per pile
        axial_stress = abs(row['Fz']) / num_piles
        total_stress = axial_stress + mx_stress + my_stress
        
        # Calculate utilization ratio
        utilization = total_stress / pile_capacity
        
        # Store analysis results
        analysis = {
            'Footing_Type': footing_type,
            'Num_Piles': num_piles,
            'Axial_Stress': axial_stress,
            'Mx_Stress': mx_stress,
            'My_Stress': my_stress,
            'Total_Stress': total_stress,
            'Utilization_Ratio': utilization,
            'Is_Safe': utilization <= 1.0,
            'Target_Diff': abs(utilization - target_utilization)
        }
        
        # Check if this is a valid solution
        if utilization <= 1.0:  # Safe design
            # Prefer solutions closer to target utilization
            if best_analysis is None or analysis['Target_Diff'] < best_analysis['Target_Diff']:
                best_analysis = analysis
                best_footing = footing_type
                best_utilization = utilization
        
        # Early exit if we found optimal solution
        if 0.80 <= utilization <= 0.95:
            break
    
    # If no safe solution found, use the largest footing
    if best_analysis is None:
        largest_footing = df_pile.iloc[-1]
        num_piles = largest_footing['Num_Piles']
        footing_type = largest_footing['Footing Type']
        
        if pile_type == "Spun Pile 600":
            mx_stress = abs(row['Mx']) * largest_footing['S_Fac_X']
            my_stress = abs(row['My']) * largest_footing['S_Fac_Y']
        else:
            mx_stress = abs(row['Mx']) * largest_footing['I_Fac_X']
            my_stress = abs(row['My']) * largest_footing['I_Fac_Y']
        
        axial_stress = abs(row['Fz']) / num_piles
        total_stress = axial_stress + mx_stress + my_stress
        utilization = total_stress / pile_capacity
        
        best_analysis = {
            'Footing_Type': footing_type,
            'Num_Piles': num_piles,
            'Axial_Stress': axial_stress,
            'Mx_Stress': mx_stress,
            'My_Stress': my_stress,
            'Total_Stress': total_stress,
            'Utilization_Ratio': utilization,
            'Is_Safe': utilization <= 1.0,
            'Target_Diff': abs(utilization - target_utilization)
        }
    
    return best_analysis

def comprehensive_pile_analysis(df, nodes, pile_type, pile_capacity, target_utilization=0.85):
    """Perform optimized pile analysis for multiple load combinations"""
    
    # Filter nodes
    df_filtered = df[df['Node'].isin(nodes)].copy()
    
    # Create footing factors DataFrame
    df_pile = pd.DataFrame(FOOTING_FACTORS)
    
    # Initialize results list
    all_results = []
    
    # Process each row (each load combination for each node)
    for idx, row in df_filtered.iterrows():
        # Optimize footing for target utilization
        analysis = optimize_footing_for_target_utilization(row, pile_type, pile_capacity, df_pile, target_utilization)
        
        # Combine original data with analysis results
        result_row = {
            'Node': row['Node'],
            'Load_Combination': row.get('Load Combination', row.get('Load Case', f'Case_{idx}')),
            'X': row.get('X', 0),
            'Y': row.get('Y', 0), 
            'Z': row.get('Z', 0),
            'Fx': row.get('FX (tonf)', row.get('Fx', 0)),
            'Fy': row.get('FY (tonf)', row.get('Fy', 0)),
            'Fz': row.get('FZ (tonf)', row.get('Fz', 0)),
            'Mx': row.get('MX (tonf·m)', row.get('Mx', 0)),
            'My': row.get('MY (tonf·m)', row.get('My', 0)),
            'Mz': row.get('MZ (tonf·m)', row.get('Mz', 0)),
            'Pile_Type': pile_type,
            'Target_Utilization': target_utilization
        }
        
        # Add optimized analysis results
        result_row.update(analysis)
        all_results.append(result_row)
    
    # Convert to DataFrame
    df_all_cases = pd.DataFrame(all_results)
    
    return df_all_cases

def get_critical_footing_per_node(df_all_cases):
    """For each node, select the maximum footing type needed across all load combinations"""
    
    def get_max_footing_for_node(group):
        """Get the maximum footing requirement for a node across all load cases"""
        group = group.copy()
        
        # Find the case requiring maximum piles
        max_pile_idx = group['Num_Piles'].idxmax()
        critical_case = group.loc[max_pile_idx].copy()
        
        # Add summary information
        critical_case['Total_Load_Cases'] = len(group)
        critical_case['Max_Fz'] = group['Fz'].max()
        critical_case['Min_Fz'] = group['Fz'].min()
        critical_case['Avg_Fz'] = group['Fz'].mean()
        critical_case['Max_Utilization'] = group['Utilization_Ratio'].max()
        critical_case['Min_Utilization'] = group['Utilization_Ratio'].min()
        critical_case['Avg_Utilization'] = group['Utilization_Ratio'].mean()
        
        # Efficiency metrics
        critical_case['Utilization_Category'] = categorize_utilization(critical_case['Utilization_Ratio'])
        
        return critical_case
    
    # Group by node and get critical case for each
    critical_results = df_all_cases.groupby('Node').apply(get_max_footing_for_node).reset_index(drop=True)
    
    return critical_results

def categorize_utilization(utilization):
    """Categorize utilization ratio"""
    if utilization < 0.6:
        return "Over-Conservative"
    elif utilization < 0.8:
        return "Conservative"
    elif utilization <= 0.95:
        return "Optimal"
    elif utilization <= 1.0:
        return "Near-Capacity"
    else:
        return "Over-Capacity"

def create_enhanced_visualizations(df_all_cases, final_results):
    """Create enhanced visualization plots including XY bubble charts and optimized 3D plots"""
    
    plots = {}
    
    # 1. XY Plan View with Bubble Chart (Utilization-based)
    if 'X' in final_results.columns and 'Y' in final_results.columns:
        # Create utilization color scale
        fig_xy = px.scatter(
            final_results,
            x='X', y='Y',
            size='Num_Piles',
            color='Utilization_Ratio',
            color_continuous_scale=['green', 'yellow', 'orange', 'red'],
            size_max=30,
            hover_data=['Node', 'Footing_Type', 'Max_Fz', 'Utilization_Category'],
            title='XY Plan View - Pile Utilization Analysis',
            labels={'X': 'X Coordinate (m)', 'Y': 'Y Coordinate (m)', 
                   'Utilization_Ratio': 'Utilization Ratio'}
        )
        
        # Add target utilization lines
        fig_xy.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3)
        fig_xy.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.3)
        
        # Update color bar
        fig_xy.update_coloraxes(
            colorbar=dict(
                title="Utilization Ratio",
                tickmode="array",
                tickvals=[0.5, 0.7, 0.85, 1.0],
                ticktext=["50%", "70%", "85%", "100%"]
            )
        )
        
        plots['xy_bubble'] = fig_xy
    
    # 2. Enhanced 3D Scatter with Utilization Categories
    if 'X' in final_results.columns and 'Y' in final_results.columns:
        fig_3d = px.scatter_3d(
            final_results,
            x='X', y='Y', z='Z',
            color='Utilization_Category',
            size='Num_Piles',
            hover_data=['Node', 'Footing_Type', 'Utilization_Ratio'],
            title='3D Site Layout - Utilization Categories',
            color_discrete_map={
                'Over-Conservative': '#28a745',
                'Conservative': '#ffc107', 
                'Optimal': '#17a2b8',
                'Near-Capacity': '#fd7e14',
                'Over-Capacity': '#dc3545'
            }
        )
        fig_3d.update_layout(scene=dict(aspectmode='data'))
        plots['3d_utilization'] = fig_3d
    
    # 3. Load vs Utilization Optimization Chart
    fig_opt = px.scatter(
        final_results,
        x='Max_Fz',
        y='Utilization_Ratio',
        color='Utilization_Category',
        size='Num_Piles',
        hover_data=['Node', 'Footing_Type'],
        title='Load vs Utilization Optimization',
        labels={'Max_Fz': 'Maximum Axial Load (tonf)', 'Utilization_Ratio': 'Utilization Ratio'}
    )
    
    # Add target zones
    fig_opt.add_hrect(y0=0.8, y1=0.95, fillcolor="lightgreen", opacity=0.2, 
                      annotation_text="Target Zone (80-95%)")
    fig_opt.add_hline(y=1.0, line_dash="dash", line_color="red", 
                      annotation_text="Capacity Limit")
    plots['load_utilization'] = fig_opt
    
    # 4. Utilization Efficiency Analysis
    utilization_summary = final_results['Utilization_Category'].value_counts()
    fig_efficiency = px.pie(
        values=utilization_summary.values,
        names=utilization_summary.index,
        title='Utilization Efficiency Distribution',
        color_discrete_map={
            'Over-Conservative': '#28a745',
            'Conservative': '#ffc107', 
            'Optimal': '#17a2b8',
            'Near-Capacity': '#fd7e14',
            'Over-Capacity': '#dc3545'
        }
    )
    plots['efficiency_pie'] = fig_efficiency
    
    # 5. Multi-Load Case Analysis
    if 'Total_Load_Cases' in final_results.columns:
        fig_cases = px.scatter(
            final_results,
            x='Total_Load_Cases',
            y='Utilization_Ratio',
            color='Num_Piles',
            size='Max_Fz',
            hover_data=['Node', 'Footing_Type'],
            title='Load Cases vs Utilization Analysis',
            labels={'Total_Load_Cases': 'Number of Load Cases', 'Utilization_Ratio': 'Utilization Ratio'}
        )
        fig_cases.add_hline(y=0.85, line_dash="dash", line_color="blue", 
                           annotation_text="Target Utilization")
        plots['cases_analysis'] = fig_cases
    
    # 6. Comparative Analysis - Before/After Optimization
    if len(df_all_cases) > 0:
        # Create comparison data (simulate old vs new method)
        comparison_data = []
        for _, row in final_results.iterrows():
            # Old method (conservative)
            old_piles = int(np.ceil(row['Max_Fz'] / 120)) + 2
            old_utilization = (row['Max_Fz'] / old_piles) / 120
            
            comparison_data.extend([
                {'Node': row['Node'], 'Method': 'Old (Conservative)', 
                 'Piles': old_piles, 'Utilization': old_utilization},
                {'Node': row['Node'], 'Method': 'New (Optimized)', 
                 'Piles': row['Num_Piles'], 'Utilization': row['Utilization_Ratio']}
            ])
        
        comp_df = pd.DataFrame(comparison_data)
        fig_comp = px.scatter(
            comp_df,
            x='Piles',
            y='Utilization',
            color='Method',
            facet_col='Method',
            hover_data=['Node'],
            title='Method Comparison: Conservative vs Optimized',
            labels={'Piles': 'Number of Piles', 'Utilization': 'Utilization Ratio'}
        )
        plots['method_comparison'] = fig_comp
    
    return plots

# Sidebar inputs
uploaded_file = st.sidebar.file_uploader(
    "📁 Upload CSV File", 
    type=['csv'],
    help="Upload your structural analysis data in CSV format"
)

pile_type = st.sidebar.selectbox(
    "🔧 Select Pile Type",
    ["Spun Pile 600", "PC I 300"],
    help="Choose the type of pile for analysis"
)

pile_capacity = st.sidebar.number_input(
    "⚡ Pile Capacity (tonf)",
    min_value=50,
    max_value=500,
    value=120,
    step=10,
    help="Enter the pile capacity in tons"
)

# Target utilization setting
target_utilization = st.sidebar.slider(
    "🎯 Target Utilization Ratio",
    min_value=0.7,
    max_value=0.95,
    value=0.85,
    step=0.05,
    help="Target utilization ratio for optimization (80-90% recommended)"
)

# Node selection
st.sidebar.subheader("🎯 Node Selection")
use_default_nodes = st.sidebar.checkbox("Use Default Nodes", value=True)

if use_default_nodes:
    selected_nodes = DEFAULT_NODES
    st.sidebar.info(f"Using {len(DEFAULT_NODES)} default nodes")
else:
    nodes_input = st.sidebar.text_area(
        "Enter Node Numbers (comma-separated)",
        value=",".join(map(str, DEFAULT_NODES[:10])),
        help="Enter node numbers separated by commas"
    )
    try:
        selected_nodes = [int(x.strip()) for x in nodes_input.split(",") if x.strip()]
        st.sidebar.success(f"Selected {len(selected_nodes)} nodes")
    except:
        st.sidebar.error("Invalid node format. Use comma-separated integers.")
        selected_nodes = DEFAULT_NODES

# Main content
if uploaded_file is not None:
    # Load data
    df, message = load_data(uploaded_file)
    
    if df is not None:
        st.success(message)
        
        # Display data info
        st.subheader("📊 Data Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            if 'Node' in df.columns:
                unique_nodes = df['Node'].nunique()
                st.metric("Unique Nodes", unique_nodes)
        
        # Show available columns
        st.write("**Available Columns:**", ", ".join(df.columns.tolist()))
        
        # Show sample data
        with st.expander("📋 Preview Data (First 10 rows)"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Auto-detect columns for the specific format provided
        required_cols = ['Node']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            # Enhanced column mapping for the specific format provided
            st.subheader("🔍 Auto-Detected Column Mapping")
            
            df_standardized = df.copy()
            
            # Display the cleaned column names
            st.info(f"**Cleaned Columns:** {', '.join(df.columns.tolist())}")
            
            # Flexible column mapping to handle variations
            column_mapping = {}
            
            # Map force columns
            for col in df.columns:
                col_lower = col.lower()
                if 'fx' in col_lower and 'tonf' in col_lower:
                    column_mapping[col] = 'Fx'
                elif 'fy' in col_lower and 'tonf' in col_lower:
                    column_mapping[col] = 'Fy'  
                elif 'fz' in col_lower and 'tonf' in col_lower:
                    column_mapping[col] = 'Fz'
                elif 'mx' in col_lower and ('tonf' in col_lower or 'moment' in col_lower):
                    column_mapping[col] = 'Mx'
                elif 'my' in col_lower and ('tonf' in col_lower or 'moment' in col_lower):
                    column_mapping[col] = 'My'
                elif 'mz' in col_lower and ('tonf' in col_lower or 'moment' in col_lower):
                    column_mapping[col] = 'Mz'
            
            # Apply column mapping
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df_standardized[new_col] = df[old_col]
            
            # Show the mapping
            if column_mapping:
                st.success("✅ Auto-mapped columns:")
                for old_col, new_col in column_mapping.items():
                    st.write(f"  • `{old_col}` → `{new_col}`")
            
            # Ensure all required columns exist
            required_analysis_cols = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
            for col in required_analysis_cols:
                if col not in df_standardized.columns:
                    df_standardized[col] = 0
                    st.warning(f"Column '{col}' not found, defaulting to 0")
            
            # Ensure coordinate columns exist
            for coord in ['X', 'Y', 'Z']:
                if coord not in df_standardized.columns:
                    df_standardized[coord] = 0
            
            # Check if we have the essential data
            essential_cols = ['Node', 'Fz', 'Mx', 'My']
            missing_essential = [col for col in essential_cols if col not in df_standardized.columns or df_standardized[col].isna().all()]
            
            if missing_essential:
                st.error(f"❌ Missing essential columns for analysis: {missing_essential}")
                st.write("**Required columns:** Node, Fz (axial force), Mx, My (moments)")
            else:
                st.success(f"✅ All essential columns ready for analysis!")
                st.write(f"**Ready to analyze:** {len(df_standardized[df_standardized['Node'].isin(selected_nodes)])} rows for selected nodes")
            
            # Run analysis button
            if st.sidebar.button("🚀 Run Optimized Analysis", type="primary"):
                with st.spinner("Performing optimized pile analysis..."):
                    try:
                        # Perform optimized analysis
                        all_cases_results = comprehensive_pile_analysis(
                            df_standardized, selected_nodes, pile_type, pile_capacity, target_utilization
                        )
                        final_node_results = get_critical_footing_per_node(all_cases_results)
                        
                        # Store in session state
                        st.session_state.analysis_results = all_cases_results
                        st.session_state.final_results = final_node_results
                        
                        st.success("✅ Optimized analysis completed successfully!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.write("**Debug Info:**")
                        st.write("DataFrame shape:", df_standardized.shape)
                        st.write("Selected nodes:", len(selected_nodes))

# Display results if available
if st.session_state.analysis_results is not None and st.session_state.final_results is not None:
    all_cases = st.session_state.analysis_results
    final_results = st.session_state.final_results
    
    # Create tabs for results
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Optimization Summary", "📈 Enhanced Visualizations", "🎯 Utilization Analysis", "📋 Detailed Results", "💾 Export"])
    
    with tab1:
        st.markdown('<h2 class="section-header">📊 Optimization Summary</h2>', unsafe_allow_html=True)
        
        # Key optimization metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Nodes Analyzed", len(final_results))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            avg_utilization = final_results['Utilization_Ratio'].mean()
            st.metric("Avg Utilization", f"{avg_utilization:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            optimal_count = len(final_results[final_results['Utilization_Category'] == 'Optimal'])
            st.metric("Optimal Designs", f"{optimal_count}/{len(final_results)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            total_piles = final_results['Num_Piles'].sum()
            st.metric("Total Piles", int(total_piles))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col5:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            target_ratio = target_utilization
            st.metric("Target Utilization", f"{target_ratio:.0%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Utilization categories breakdown
        st.subheader("🎯 Utilization Efficiency Breakdown")
        
        utilization_counts = final_results['Utilization_Category'].value_counts()
        for category, count in utilization_counts.items():
            percentage = (count / len(final_results)) * 100
            
            if category == "Optimal":
                st.markdown(f'''
                <div class="optimal-node">
                    <strong>{category}</strong>: {count} nodes ({percentage:.1f}%) - 
                    Well-optimized designs with 80-95% utilization
                </div>
                ''', unsafe_allow_html=True)
            elif category in ["Over-Conservative", "Conservative"]:
                st.markdown(f'''
                <div class="conservative-node">
                    <strong>{category}</strong>: {count} nodes ({percentage:.1f}%) - 
                    Could be optimized for better material efficiency
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="critical-node">
                    <strong>{category}</strong>: {count} nodes ({percentage:.1f}%) - 
                    High utilization, review design carefully
                </div>
                ''', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<h2 class="section-header">📈 Enhanced Visualizations</h2>', unsafe_allow_html=True)
        
        # Generate enhanced plots
        plots = create_enhanced_visualizations(all_cases, final_results)
        
        # XY Plan View with Bubble Chart
        if 'xy_bubble' in plots:
            st.subheader("🗺️ XY Plan View - Utilization Bubble Chart")
            st.plotly_chart(plots['xy_bubble'], use_container_width=True)
            st.info("💡 **Bubble size** = Number of piles, **Color** = Utilization ratio (Green=Low, Red=High)")
        
        # 3D Enhanced View
        if '3d_utilization' in plots:
            st.subheader("🏗️ 3D Site Layout - Utilization Categories") 
            st.plotly_chart(plots['3d_utilization'], use_container_width=True)
        
        # Load vs Utilization
        col1, col2 = st.columns(2)
        with col1:
            if 'load_utilization' in plots:
                st.plotly_chart(plots['load_utilization'], use_container_width=True)
        with col2:
            if 'efficiency_pie' in plots:
                st.plotly_chart(plots['efficiency_pie'], use_container_width=True)
        
        # Additional analysis charts
        if 'cases_analysis' in plots:
            st.plotly_chart(plots['cases_analysis'], use_container_width=True)
        
        if 'method_comparison' in plots:
            st.subheader("⚖️ Optimization Impact Analysis")
            st.plotly_chart(plots['method_comparison'], use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="section-header">🎯 Detailed Utilization Analysis</h2>', unsafe_allow_html=True)
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            category_filter = st.selectbox("Filter by Category", 
                                         ['All'] + list(final_results['Utilization_Category'].unique()))
        with col2:
            min_utilization = st.slider("Minimum Utilization", 0.0, 1.0, 0.0, 0.05)
        with col3:
            max_utilization = st.slider("Maximum Utilization", 0.0, 1.2, 1.2, 0.05)
        
        # Apply filters
        filtered_results = final_results.copy()
        if category_filter != 'All':
            filtered_results = filtered_results[filtered_results['Utilization_Category'] == category_filter]
        
        filtered_results = filtered_results[
            (filtered_results['Utilization_Ratio'] >= min_utilization) &
            (filtered_results['Utilization_Ratio'] <= max_utilization)
        ]
        
        # Display filtered results
        st.subheader(f"📊 Filtered Results ({len(filtered_results)} nodes)")
        
        # Key columns for utilization analysis
        display_columns = ['Node', 'X', 'Y', 'Footing_Type', 'Num_Piles', 'Max_Fz', 
                          'Utilization_Ratio', 'Utilization_Category', 'Total_Stress', 'Is_Safe']
        available_columns = [col for col in display_columns if col in filtered_results.columns]
        
        if not filtered_results.empty:
            # Format the display
            display_data = filtered_results[available_columns].copy()
            display_data['Utilization_Ratio'] = display_data['Utilization_Ratio'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(display_data, use_container_width=True, height=400)
            
            # Summary statistics for filtered data
            if len(filtered_results) > 1:
                st.subheader("📈 Filtered Data Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_util = filtered_results['Utilization_Ratio'].mean()
                    st.metric("Average Utilization", f"{avg_util:.1%}")
                with col2:
                    pile_savings = len(filtered_results[filtered_results['Utilization_Category'] == 'Optimal'])
                    st.metric("Well-Optimized", f"{pile_savings}")
                with col3:
                    avg_piles = filtered_results['Num_Piles'].mean()
                    st.metric("Avg Piles", f"{avg_piles:.1f}")
                with col4:
                    total_piles_filtered = filtered_results['Num_Piles'].sum()
                    st.metric("Total Piles", int(total_piles_filtered))
        else:
            st.warning("No nodes match the selected filters.")
    
    with tab4:
        st.markdown('<h2 class="section-header">📋 Comprehensive Analysis Results</h2>', unsafe_allow_html=True)
        
        # Search and filter options
        col1, col2 = st.columns(2)
        with col1:
            search_node = st.number_input("🔍 Search Node", min_value=0, value=0)
        with col2:
            show_all_cases = st.checkbox("Show All Load Cases", value=False)
        
        if search_node > 0:
            if show_all_cases:
                filtered_data = all_cases[all_cases['Node'] == search_node]
                st.subheader(f"All Load Cases for Node {search_node}")
            else:
                filtered_data = final_results[final_results['Node'] == search_node]
                st.subheader(f"Optimized Result for Node {search_node}")
        else:
            if show_all_cases:
                filtered_data = all_cases.copy()
                st.subheader("All Load Cases - Detailed Analysis")
            else:
                filtered_data = final_results.copy()
                st.subheader("Final Optimized Results")
        
        if not filtered_data.empty:
            st.dataframe(filtered_data, use_container_width=True, height=500)
        else:
            st.warning("No data available for the selected criteria.")
    
    with tab5:
        st.markdown('<h2 class="section-header">💾 Export Optimized Results</h2>', unsafe_allow_html=True)
        
        # Create final design table
        design_columns = ['Node', 'X', 'Y', 'Z', 'Footing_Type', 'Num_Piles', 'Max_Fz', 
                         'Utilization_Ratio', 'Utilization_Category', 'Is_Safe']
        final_design = final_results[[col for col in design_columns if col in final_results.columns]].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏗️ Final Design Table")
            final_csv = final_design.to_csv(index=False)
            st.download_button(
                label="📥 Download Optimized Design (CSV)",
                data=final_csv,
                file_name=f"optimized_pile_design_{pile_type.replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
            st.subheader("📊 Complete Analysis")
            complete_csv = final_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Complete Analysis (CSV)",
                data=complete_csv,
                file_name=f"complete_pile_analysis_{pile_type.replace(' ', '_')}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.subheader("📋 All Load Cases")
            all_cases_csv = all_cases.to_csv(index=False)
            st.download_button(
                label="📥 Download All Cases (CSV)",
                data=all_cases_csv,
                file_name=f"all_load_cases_{pile_type.replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
            # Generate optimization report
            optimal_nodes = len(final_results[final_results['Utilization_Category'] == 'Optimal'])
            conservative_nodes = len(final_results[final_results['Utilization_Category'].isin(['Conservative', 'Over-Conservative'])])
            
            report = f"""# Enhanced Pile Foundation Analysis Report

## Optimization Parameters
- **Pile Type**: {pile_type}
- **Pile Capacity**: {pile_capacity} tonf
- **Target Utilization**: {target_utilization:.0%}
- **Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Optimization Results
- **Total Nodes**: {len(final_results)}
- **Optimal Designs (80-95%)**: {optimal_nodes} ({100*optimal_nodes/len(final_results):.1f}%)
- **Conservative Designs (<80%)**: {conservative_nodes} ({100*conservative_nodes/len(final_results):.1f}%)
- **Average Utilization**: {final_results['Utilization_Ratio'].mean():.1%}

## Material Efficiency
- **Total Piles Required**: {final_results['Num_Piles'].sum()}
- **Average Piles per Node**: {final_results['Num_Piles'].mean():.1f}
- **Most Common Footing**: {final_results['Footing_Type'].mode().iloc[0] if len(final_results) > 0 else 'N/A'}

## Load Analysis
- **Maximum Load**: {final_results['Max_Fz'].max():.1f} tonf
- **Average Load**: {final_results['Max_Fz'].mean():.1f} tonf
- **Load Range**: {final_results['Max_Fz'].min():.1f} - {final_results['Max_Fz'].max():.1f} tonf

## Recommendations
1. **Optimal Nodes**: {optimal_nodes} nodes achieve target efficiency (80-95% utilization)
2. **Conservative Nodes**: {conservative_nodes} nodes could be optimized for material savings
3. **Target Achievement**: {100*optimal_nodes/len(final_results):.1f}% of designs meet optimization criteria
"""
            
            st.download_button(
                label="📄 Download Optimization Report (MD)",
                data=report,
                file_name=f"optimization_report_{pile_type.replace(' ', '_')}.md",
                mime="text/markdown"
            )

else:
    # Enhanced instructions with structured data format
    st.markdown("""
    ## 🚀 Enhanced Pile Foundation Analysis Tool
    
    ### 📋 **Structured Data Framework**
    This tool is designed for **your exact data format** with robust BOM and encoding handling.
    
    ### ✨ **Key Features:**
    - **🎯 Optimized for 80-90% Utilization**: No more over-conservative designs
    - **📊 Structured Data Processing**: Expects your exact column format
    - **🗺️ Advanced Visualizations**: XY bubble charts, 3D utilization maps
    - **🔧 Robust File Handling**: Handles BOM, encoding issues automatically
    - **📈 Multi-Load Analysis**: Processes all load combinations per node
    
    ### 📄 **Required CSV Format:**
    Your file must have **exactly these columns** in this order:
    """)
    
    # Show expected format as a table
    expected_columns = [
        "Node", "X", "Y", "Z", "Load Case", "Load Combination", 
        "FX (tonf)", "FY (tonf)", "FZ (tonf)", 
        "MX (tonf·m)", "MY (tonf·m)", "MZ (tonf·m)"
    ]
    
    col_df = pd.DataFrame({
        'Column': expected_columns,
        'Type': ['Integer', 'Float', 'Float', 'Float', 'Text', 'Text', 
                'Float', 'Float', 'Float', 'Float', 'Float', 'Float'],
        'Required': ['✅ Yes', '⚠️ Optional', '⚠️ Optional', '⚠️ Optional', 
                    '⚠️ Optional', '⚠️ Optional', '⚠️ Optional', '⚠️ Optional', 
                    '✅ Yes', '✅ Yes', '✅ Yes', '⚠️ Optional'],
        'Description': [
            'Node ID number',
            'X coordinate (m)', 
            'Y coordinate (m)',
            'Z coordinate (m)',
            'Load case identifier',
            'Load combination name',
            'X-direction force (tonf)',
            'Y-direction force (tonf)', 
            'Z-direction force (tonf) - CRITICAL',
            'X-direction moment (tonf·m) - CRITICAL',
            'Y-direction moment (tonf·m) - CRITICAL',
            'Z-direction moment (tonf·m)'
        ]
    })
    
    st.dataframe(col_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    ### 📊 **Sample Data Format:**
    ```csv
    Node,X,Y,Z,Load Case,Load Combination,FX (tonf),FY (tonf),FZ (tonf),MX (tonf·m),MY (tonf·m),MZ (tonf·m)
    26,0,0,-1.5,cLCB70,SERV :D + (L),6.44,-1.33,393.73,-1.38,13.63,-0.04
    27,0,10,-1.5,cLCB70,SERV :D + (L),7.45,0.02,342.53,-4.76,17.57,-0.27
    28,0,22,-1.5,cLCB70,SERV :D + (L),1.91,3.61,284.31,-13.68,5.15,-0.06
    ```
    
    ### 🔧 **Automatic Processing:**
    1. **BOM Removal**: Handles `ï»¿` and encoding artifacts
    2. **Column Mapping**: Maps your format to internal structure
    3. **Data Validation**: Checks for required data
    4. **Missing Data**: Auto-fills missing optional columns
    5. **Multi-Load Handling**: Processes all combinations per node
    
    ### 🎯 **Expected Results:**
    - **Optimal Utilization**: 80-90% capacity usage
    - **Material Efficiency**: Right-sized footings
    - **Safety Compliance**: All designs ≤ 100% capacity
    - **Visual Analysis**: Interactive charts and 3D plots
    
    ### 🚀 **Process Flow:**
    1. **Upload** your CSV file (BOM handled automatically)
    2. **Validation** shows column mapping results  
    3. **Configuration** set target utilization (85% default)
    4. **Analysis** optimizes each node across all load cases
    5. **Results** interactive visualizations and export
    
    ### 💡 **Pro Tips:**
    - File encoding is handled automatically
    - Missing coordinates default to (0,0,0)
    - Missing forces default to 0
    - Tool shows exactly what columns were found vs expected
    """)
    
    # Show what the tool will display
    st.subheader("📋 What You'll See After Upload")
    
    with st.expander("👁️ Preview of Analysis Flow"):
        st.markdown("""
        **1. File Processing:**
        ```
        ✅ Data loaded successfully with utf-8-sig encoding
        ```
        
        **2. Column Mapping:**
        ```
        Node: ✅ Found
        FX (tonf): ✅ Found  
        FZ (tonf): ✅ Found
        MX (tonf·m): ✅ Found
        MY (tonf·m): ✅ Found
        Load Case: ⚠️ Missing (default)
        ```
        
        **3. Validation:**
        ```  
        ✅ 156 rows of data ready for analysis
        ✅ 52 unique nodes found
        ✅ Load range: -671.1 to 393.7 tonf
        ```
        
        **4. Analysis Ready:**
        ```
        ✅ Ready to analyze 45 nodes with 156 load cases
        ```
        """)
