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
    page_title="Advanced Pile Analysis Tool",
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
    .stTab [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
        font-weight: bold;
    }
    .critical-node {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏗️ Advanced Pile Foundation Analysis Tool</h1>', unsafe_allow_html=True)

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
    "Footing Type": ["F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"],
    "S_Fac_X": [1.000000, 0.277777778, 0.138888889, 0.138888889, 0.138888889, 0.138888889, 0.138888889, 0.007716049],
    "S_Fac_Y": [1.000000, 0.277777778, 0.138888889, 0.138888889, 0.138888889, 0.138888889, 0.138888889, 0.011574074],
    "I_Fac_X": [1.000000, 0.555555556, 0.555555556, 0.555555556, 0.277777778, 0.277777778, 0.277777778, 0.034293553],
    "I_Fac_Y": [1.000000, 0.555555556, 0.555555556, 0.277777778, 0.277777778, 0.277777778, 0.277777778, 0.173611111],
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

def analyze_single_load_case(row, pile_type, pile_capacity, df_pile):
    """Analyze a single load case and determine required footing"""
    
    # Initial calculations
    initial_piles = int(np.round(abs(row['Fz']) / pile_capacity)) + 1
    axial_stress = abs(row['Fz']) / initial_piles
    remaining_ratio = (pile_capacity - axial_stress) / pile_capacity
    
    # Round 2 calculations
    round2_piles = initial_piles + 1
    if remaining_ratio < 0.1:
        round2_piles = initial_piles + 2
    
    footing_type_use = f"F{round2_piles}"
    
    # Get factors from df_pile
    factors = df_pile[df_pile['Footing Type'] == footing_type_use]
    if factors.empty:
        # If footing type not found, use the largest available
        factors = df_pile.iloc[-1:] 
        footing_type_use = factors['Footing Type'].iloc[0]
        round2_piles = extract_footing_number(footing_type_use)
    
    # Calculate ratios based on pile type
    if pile_type == "Spun Pile 600":
        mx_ratio = abs(row['Mx']) * factors['S_Fac_X'].iloc[0]
        my_ratio = abs(row['My']) * factors['S_Fac_Y'].iloc[0]
    else:  # PC I 300
        mx_ratio = abs(row['Mx']) * factors['I_Fac_X'].iloc[0]
        my_ratio = abs(row['My']) * factors['I_Fac_Y'].iloc[0]
    
    p_over_n = abs(row['Fz']) / round2_piles
    ratio_total = p_over_n + mx_ratio + my_ratio
    diff_capacity = ratio_total / pile_capacity
    
    # Round 3 calculations
    round3_piles = round2_piles
    if diff_capacity > 0.999:
        round3_piles = round2_piles + 1
    
    final_footing_type = f"F{round3_piles}"
    
    return {
        'Initial_Piles': initial_piles,
        'Round2_Piles': round2_piles,
        'Round3_Piles': round3_piles,
        'Footing_Type_Round2': footing_type_use,
        'Footing_Type_Round3': final_footing_type,
        'Mx_Ratio': mx_ratio,
        'My_Ratio': my_ratio,
        'P_over_N': p_over_n,
        'Ratio_Total': ratio_total,
        'Diff_Capacity': diff_capacity,
        'Axial_Stress': axial_stress,
        'Remaining_Ratio': remaining_ratio
    }

def comprehensive_pile_analysis(df, nodes, pile_type, pile_capacity):
    """Perform comprehensive pile analysis for multiple load combinations"""
    
    # Filter nodes
    df_filtered = df[df['Node'].isin(nodes)].copy()
    
    # Create footing factors DataFrame
    df_pile = pd.DataFrame(FOOTING_FACTORS)
    
    # Initialize results list
    all_results = []
    
    # Process each row (each load combination for each node)
    for idx, row in df_filtered.iterrows():
        # Analyze this specific load case
        case_analysis = analyze_single_load_case(row, pile_type, pile_capacity, df_pile)
        
        # Combine original data with analysis results
        result_row = {
            'Node': row['Node'],
            'Load_Combination': row.get('Load Combination', row.get('Load Case', f'Case_{idx}')),
            'X': row.get('X', row.get('x', 0)),
            'Y': row.get('Y', row.get('y', 0)), 
            'Z': row.get('Z', row.get('z', 0)),
            'Fx': row.get('Fx', row.get('FX (tonf)', 0)),
            'Fy': row.get('Fy', row.get('FY (tonf)', 0)),
            'Fz': row.get('Fz', row.get('FZ (tonf)', row.get('Fz', 0))),
            'Mx': row.get('Mx', row.get('MX (tonf·m)', row.get('Mx', 0))),
            'My': row.get('My', row.get('MY (tonf·m)', row.get('My', 0))),
            'Mz': row.get('Mz', row.get('MZ (tonf·m)', 0)),
            'Pile_Type': pile_type
        }
        
        # Add analysis results
        result_row.update(case_analysis)
        all_results.append(result_row)
    
    # Convert to DataFrame
    df_all_cases = pd.DataFrame(all_results)
    
    return df_all_cases

def get_critical_footing_per_node(df_all_cases):
    """For each node, select the maximum footing type needed across all load combinations"""
    
    def get_max_footing_for_node(group):
        """Get the maximum footing requirement for a node across all load cases"""
        # Add footing numbers for comparison
        group = group.copy()
        group['Footing_Number_Round3'] = group['Footing_Type_Round3'].apply(extract_footing_number)
        
        # Find the maximum footing requirement
        max_footing_idx = group['Footing_Number_Round3'].idxmax()
        critical_case = group.loc[max_footing_idx].copy()
        
        # Add summary information
        critical_case['Total_Load_Cases'] = len(group)
        critical_case['Max_Fz'] = group['Fz'].max()
        critical_case['Min_Fz'] = group['Fz'].min()
        critical_case['Avg_Fz'] = group['Fz'].mean()
        
        # Get all footing types required for this node
        footing_types = group['Footing_Type_Round3'].unique()
        critical_case['All_Footing_Types_Required'] = ', '.join(sorted(footing_types, key=extract_footing_number))
        
        return critical_case
    
    # Group by node and get critical case for each
    critical_results = df_all_cases.groupby('Node').apply(get_max_footing_for_node).reset_index(drop=True)
    
    # Clean up helper column
    if 'Footing_Number_Round3' in critical_results.columns:
        critical_results = critical_results.drop('Footing_Number_Round3', axis=1)
    
    return critical_results

def create_visualization_plots(df_all_cases, final_results):
    """Create comprehensive visualization plots"""
    
    plots = {}
    
    # 1. 3D Scatter plot of nodes with footing types
    if 'X' in final_results.columns and 'Y' in final_results.columns:
        fig_3d = px.scatter_3d(
            final_results,
            x='X', y='Y', z='Z',
            color='Footing_Type_Round3',
            size='Round3_Piles',
            hover_data=['Node', 'Pile_Type', 'Max_Fz'],
            title='3D Node Layout with Footing Types',
            labels={'X': 'X Coordinate', 'Y': 'Y Coordinate', 'Z': 'Z Coordinate'}
        )
        fig_3d.update_layout(scene=dict(aspectmode='data'))
        plots['3d_layout'] = fig_3d
    
    # 2. Load vs Pile Requirements
    fig_load_pile = px.scatter(
        final_results,
        x='Max_Fz',
        y='Round3_Piles',
        color='Footing_Type_Round3',
        size='Total_Load_Cases',
        hover_data=['Node', 'Pile_Type'],
        title='Maximum Axial Load vs Required Piles',
        labels={'Max_Fz': 'Maximum Axial Load (tonf)', 'Round3_Piles': 'Required Piles'}
    )
    plots['load_vs_pile'] = fig_load_pile
    
    # 3. Footing Type Distribution
    footing_dist = final_results['Footing_Type_Round3'].value_counts().sort_index()
    fig_dist = px.pie(
        values=footing_dist.values,
        names=footing_dist.index,
        title='Distribution of Required Footing Types'
    )
    plots['footing_distribution'] = fig_dist
    
    # 4. Load Combination Analysis per Node
    load_case_summary = df_all_cases.groupby(['Node', 'Footing_Type_Round3']).size().reset_index(name='Count')
    fig_load_cases = px.bar(
        load_case_summary,
        x='Node',
        y='Count',
        color='Footing_Type_Round3',
        title='Load Cases by Node and Required Footing Type',
        labels={'Count': 'Number of Load Cases', 'Node': 'Node ID'}
    )
    plots['load_cases_analysis'] = fig_load_cases
    
    # 5. Capacity Utilization Heatmap
    if len(final_results) > 1:
        # Create capacity utilization matrix
        utilization_data = final_results[['Node', 'Ratio_Total', 'Round3_Piles']].copy()
        utilization_data['Utilization_Ratio'] = utilization_data['Ratio_Total'] / 120  # Assuming 120 as capacity
        
        fig_heatmap = px.scatter(
            utilization_data,
            x='Node',
            y='Utilization_Ratio',
            color='Round3_Piles',
            size='Ratio_Total',
            title='Pile Capacity Utilization by Node',
            labels={'Utilization_Ratio': 'Capacity Utilization Ratio', 'Node': 'Node ID'}
        )
        fig_heatmap.add_hline(y=1.0, line_dash="dash", line_color="red", 
                             annotation_text="Capacity Limit")
        plots['capacity_utilization'] = fig_heatmap
    
    # 6. Force Components Analysis
    if 'Fx' in final_results.columns and 'Fy' in final_results.columns:
        fig_forces = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Axial Forces (Fz)', 'Horizontal Forces (Fx, Fy)', 
                          'Moments (Mx, My)', 'Force Resultants'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Axial forces
        fig_forces.add_trace(
            go.Scatter(x=final_results['Node'], y=final_results['Max_Fz'], 
                      mode='markers+lines', name='Max Fz', 
                      marker=dict(color='blue')),
            row=1, col=1
        )
        
        # Horizontal forces
        fig_forces.add_trace(
            go.Scatter(x=final_results['Node'], y=final_results['Fx'], 
                      mode='markers', name='Fx', 
                      marker=dict(color='red')),
            row=1, col=2
        )
        fig_forces.add_trace(
            go.Scatter(x=final_results['Node'], y=final_results['Fy'], 
                      mode='markers', name='Fy', 
                      marker=dict(color='green')),
            row=1, col=2
        )
        
        # Moments
        fig_forces.add_trace(
            go.Scatter(x=final_results['Node'], y=final_results['Mx'], 
                      mode='markers', name='Mx', 
                      marker=dict(color='orange')),
            row=2, col=1
        )
        fig_forces.add_trace(
            go.Scatter(x=final_results['Node'], y=final_results['My'], 
                      mode='markers', name='My', 
                      marker=dict(color='purple')),
            row=2, col=1
        )
        
        # Resultant forces
        final_results['Force_Resultant'] = np.sqrt(final_results['Fx']**2 + final_results['Fy']**2 + final_results['Max_Fz']**2)
        fig_forces.add_trace(
            go.Scatter(x=final_results['Node'], y=final_results['Force_Resultant'], 
                      mode='markers+lines', name='Force Resultant', 
                      marker=dict(color='black')),
            row=2, col=2
        )
        
        fig_forces.update_layout(title_text="Comprehensive Force Analysis", height=600)
        plots['force_analysis'] = fig_forces
    
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
        
        # Check required columns
        required_cols = ['Node']
        force_cols = [col for col in df.columns if any(x in col.lower() for x in ['fz', 'f_z', 'force_z', 'axial'])]
        moment_cols = [col for col in df.columns if any(x in col.lower() for x in ['mx', 'my', 'm_x', 'm_y', 'moment'])]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        elif not force_cols:
            st.error("No axial force column found. Looking for columns containing: 'Fz', 'FZ', 'force_z', 'axial'")
        elif len(moment_cols) < 2:
            st.error("Need at least 2 moment columns (Mx, My). Found: " + ", ".join(moment_cols))
        else:
            # Auto-detect column mappings
            st.subheader("🔍 Column Mapping")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fz_col = st.selectbox("Axial Force Column (Fz):", force_cols)
            with col2:
                mx_col = st.selectbox("Moment X Column (Mx):", moment_cols)
            with col3:
                my_col = st.selectbox("Moment Y Column (My):", [col for col in moment_cols if col != mx_col])
            
            # Optional columns
            coord_cols = [col for col in df.columns if col.upper() in ['X', 'Y', 'Z']]
            other_force_cols = [col for col in df.columns if any(x in col.lower() for x in ['fx', 'fy', 'f_x', 'f_y'])]
            
            # Standardize column names
            df_standardized = df.copy()
            df_standardized['Fz'] = df[fz_col]
            df_standardized['Mx'] = df[mx_col] 
            df_standardized['My'] = df[my_col]
            
            # Add coordinate columns if available
            for coord in ['X', 'Y', 'Z']:
                coord_options = [col for col in df.columns if col.upper() == coord]
                if coord_options:
                    df_standardized[coord] = df[coord_options[0]]
                else:
                    df_standardized[coord] = 0
            
            # Add other force columns if available  
            for force_col in ['Fx', 'Fy']:
                force_options = [col for col in df.columns if col.upper() == force_col]
                if force_options:
                    df_standardized[force_col] = df[force_options[0]]
                else:
                    df_standardized[force_col] = 0
            
            # Run analysis button
            if st.sidebar.button("🚀 Run Comprehensive Analysis", type="primary"):
                with st.spinner("Performing comprehensive pile analysis..."):
                    try:
                        # Perform comprehensive analysis
                        all_cases_results = comprehensive_pile_analysis(df_standardized, selected_nodes, pile_type, pile_capacity)
                        final_node_results = get_critical_footing_per_node(all_cases_results)
                        
                        # Store in session state
                        st.session_state.analysis_results = all_cases_results
                        st.session_state.final_results = final_node_results
                        
                        st.success("✅ Comprehensive analysis completed successfully!")
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Executive Summary", "📈 Visualizations", "📋 Detailed Analysis", "🎯 Critical Cases", "💾 Export"])
    
    with tab1:
        st.markdown('<h2 class="section-header">📊 Executive Summary</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Nodes Analyzed", len(final_results))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            total_cases = all_cases.shape[0]
            st.metric("Total Load Cases", total_cases)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            max_piles = final_results['Round3_Piles'].max()
            st.metric("Max Piles Required", int(max_piles))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            avg_piles = final_results['Round3_Piles'].mean()
            st.metric("Average Piles", f"{avg_piles:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col5:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            total_piles = final_results['Round3_Piles'].sum()
            st.metric("Total Piles", int(total_piles))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Summary statistics
        st.subheader("📈 Load Distribution Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Axial Load Statistics (tonf):**")
            load_stats = final_results['Max_Fz'].describe()
            st.dataframe(load_stats.to_frame().T, use_container_width=True)
        
        with col2:
            st.write("**Pile Requirements Distribution:**")
            pile_dist = final_results['Round3_Piles'].value_counts().sort_index()
            st.dataframe(pile_dist.to_frame().T, use_container_width=True)
        
        # Critical nodes identification
        st.subheader("🚨 Critical Nodes (Highest Load)")
        critical_nodes = final_results.nlargest(5, 'Max_Fz')[['Node', 'Max_Fz', 'Round3_Piles', 'Footing_Type_Round3', 'Total_Load_Cases']]
        
        for idx, row in critical_nodes.iterrows():
            st.markdown(f'''
            <div class="critical-node">
                <strong>Node {int(row['Node'])}</strong> - Max Load: {row['Max_Fz']:.1f} tonf - 
                Required: {int(row['Round3_Piles'])} piles ({row['Footing_Type_Round3']}) - 
                Load Cases: {int(row['Total_Load_Cases'])}
            </div>
            ''', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<h2 class="section-header">📈 Comprehensive Visualizations</h2>', unsafe_allow_html=True)
        
        # Generate all plots
        plots = create_visualization_plots(all_cases, final_results)
        
        # Display plots
        if '3d_layout' in plots:
            st.subheader("🏗️ 3D Site Layout")
            st.plotly_chart(plots['3d_layout'], use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if 'load_vs_pile' in plots:
                st.plotly_chart(plots['load_vs_pile'], use_container_width=True)
        with col2:
            if 'footing_distribution' in plots:
                st.plotly_chart(plots['footing_distribution'], use_container_width=True)
        
        if 'load_cases_analysis' in plots:
            st.plotly_chart(plots['load_cases_analysis'], use_container_width=True)
        
        if 'capacity_utilization' in plots:
            st.plotly_chart(plots['capacity_utilization'], use_container_width=True)
        
        if 'force_analysis' in plots:
            st.plotly_chart(plots['force_analysis'], use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="section-header">📋 Detailed Analysis Results</h2>', unsafe_allow_html=True)
        
        # Search and filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            search_node = st.number_input("🔍 Search Node", min_value=0, value=0)
        with col2:
            filter_footing = st.selectbox("Filter by Footing Type", 
                                        ['All'] + sorted(final_results['Footing_Type_Round3'].unique()))
        with col3:
            show_all_cases = st.checkbox("Show All Load Cases", value=False)
        
        # Apply filters
        if search_node > 0:
            if show_all_cases:
                filtered_data = all_cases[all_cases['Node'] == search_node]
                st.subheader(f"All Load Cases for Node {search_node}")
            else:
                filtered_data = final_results[final_results['Node'] == search_node]
                st.subheader(f"Critical Case for Node {search_node}")
        else:
            if show_all_cases:
                filtered_data = all_cases.copy()
                st.subheader("All Load Cases")
            else:
                filtered_data = final_results.copy()
                st.subheader("Critical Cases per Node")
        
        if filter_footing != 'All' and not show_all_cases:
            filtered_data = filtered_data[filtered_data['Footing_Type_Round3'] == filter_footing]
        
        # Display interactive table
        if not filtered_data.empty:
            st.dataframe(filtered_data, use_container_width=True, height=400)
            
            # Show summary for filtered data
            if len(filtered_data) > 1:
                st.subheader("📊 Filtered Data Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Records", len(filtered_data))
                with col2:
                    if 'Max_Fz' in filtered_data.columns:
                        st.metric("Avg Max Load", f"{filtered_data['Max_Fz'].mean():.1f} tonf")
                    else:
                        st.metric("Avg Load", f"{filtered_data['Fz'].mean():.1f} tonf")
                with col3:
                    st.metric("Avg Piles", f"{filtered_data['Round3_Piles'].mean():.1f}")
        else:
            st.warning("No data matches the selected filters.")
    
    with tab4:
        st.markdown('<h2 class="section-header">🎯 Final Design Results</h2>', unsafe_allow_html=True)
        
        # Create the final design table
        design_columns = ['Node', 'X', 'Y', 'Z', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Pile_Type', 'Footing_Type_Round3']
        available_design_cols = [col for col in design_columns if col in final_results.columns]
        
        # Use Max_Fz if Fz not available directly
        if 'Fz' not in final_results.columns and 'Max_Fz' in final_results.columns:
            final_results_display = final_results.copy()
            final_results_display['Fz'] = final_results_display['Max_Fz']
        else:
            final_results_display = final_results.copy()
        
        # Ensure all required columns exist
        for col in design_columns:
            if col not in final_results_display.columns:
                if col == 'Mz':
                    final_results_display[col] = 0  # Default value for Mz if not available
                elif col == 'Footing_Type_Round3':
                    # Already exists
                    pass
        
        # Create the final design table
        final_design_table = final_results_display[available_design_cols].copy()
        final_design_table.columns = [col.replace('Footing_Type_Round3', 'Footing Type') for col in final_design_table.columns]
        
        st.subheader("🏗️ Final Design Table")
        st.dataframe(final_design_table, use_container_width=True, height=400)
        
        # Summary by footing type
        st.subheader("📊 Design Summary by Footing Type")
        summary_by_footing = final_results.groupby(['Footing_Type_Round3', 'Pile_Type']).agg({
            'Node': 'count',
            'Round3_Piles': ['sum', 'mean'],
            'Max_Fz': ['max', 'mean']
        }).round(2)
        
        summary_by_footing.columns = ['Nodes_Count', 'Total_Piles', 'Avg_Piles_per_Node', 'Max_Load', 'Avg_Load']
        st.dataframe(summary_by_footing, use_container_width=True)
    
    with tab5:
        st.markdown('<h2 class="section-header">💾 Export Analysis Results</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Final Design Results")
            final_csv = final_design_table.to_csv(index=False)
            st.download_button(
                label="📥 Download Final Design Table (CSV)",
                data=final_csv,
                file_name=f"final_design_{pile_type.replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
            st.subheader("🎯 Critical Cases Only")
            critical_csv = final_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Critical Analysis (CSV)",
                data=critical_csv,
                file_name=f"critical_analysis_{pile_type.replace(' ', '_')}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.subheader("📋 Complete Load Cases")
            all_cases_csv = all_cases.to_csv(index=False)
            st.download_button(
                label="📥 Download All Load Cases (CSV)",
                data=all_cases_csv,
                file_name=f"all_load_cases_{pile_type.replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
            # Generate comprehensive report
            report = f"""# Comprehensive Pile Analysis Report

## Project Parameters
- **Pile Type**: {pile_type}
- **Pile Capacity**: {pile_capacity} tonf
- **Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- **Total Nodes Analyzed**: {len(final_results)}
- **Total Load Cases**: {len(all_cases)}
- **Maximum Piles Required**: {int(final_results['Round3_Piles'].max())}
- **Total Piles Required**: {int(final_results['Round3_Piles'].sum())}
- **Average Piles per Node**: {final_results['Round3_Piles'].mean():.1f}

## Load Statistics
- **Maximum Axial Load**: {final_results['Max_Fz'].max():.1f} tonf
- **Average Axial Load**: {final_results['Max_Fz'].mean():.1f} tonf
- **Minimum Axial Load**: {final_results['Max_Fz'].min():.1f} tonf

## Footing Type Distribution
{final_results['Footing_Type_Round3'].value_counts().to_string()}

## Critical Nodes (Top 10 by Load)
{final_results.nlargest(10, 'Max_Fz')[['Node', 'Max_Fz', 'Round3_Piles', 'Footing_Type_Round3']].to_string(index=False)}
"""
            
            st.download_button(
                label="📄 Download Comprehensive Report (MD)",
                data=report,
                file_name=f"pile_analysis_report_{pile_type.replace(' ', '_')}.md",
                mime="text/markdown"
            )

else:
    # Show enhanced instructions
    st.markdown("""
    ## 🚀 Advanced Pile Analysis Tool
    
    This tool performs comprehensive pile foundation analysis for **multiple load combinations** per node, automatically selecting the optimal footing type that can handle all load scenarios.
    
    ### 🔄 Analysis Process:
    1. **Multi-Case Analysis**: Analyzes each load combination separately
    2. **Footing Optimization**: Determines required footing for each case  
    3. **Critical Selection**: Selects maximum footing needed per node
    4. **Example**: Node 26 (Case 1→F4, Case 2→F5, Case 3→F6) → **Final Selection: F6**
    
    ### 📋 Required CSV Format:
    Your data should contain these columns:
    - **Node**: Node identification number
    - **Load Combination**: Load case identifier (optional)
    - **Coordinates**: X, Y, Z (optional)
    - **Forces**: Fx, Fy, Fz (Fz required)
    - **Moments**: Mx, My, Mz (Mx, My required)
    
    ### 🎯 Analysis Features:
    - ✅ **Multi-load combination handling**
    - ✅ **Automatic critical case selection** 
    - ✅ **3D visualization with coordinates**
    - ✅ **Interactive force analysis**
    - ✅ **Comprehensive reporting**
    - ✅ **Export-ready design tables**
    
    ### 📊 Output Results:
    - **Final Design Table**: Node, X, Y, Z, Fx, Fy, Fz, Mx, My, Mz, Pile Type, Footing Type
    - **Interactive Plotly Diagrams**: 3D layout, load analysis, capacity utilization
    - **Critical Case Analysis**: Maximum requirements per node
    - **Comprehensive Reports**: Export-ready documentation
    """)
    
    # Enhanced example data
    st.subheader("📊 Example Data Structure")
    example_data = pd.DataFrame({
        'Node': [26, 26, 26, 27, 27, 28],
        'Load Combination': ['LC1', 'LC2', 'LC3', 'LC1', 'LC2', 'LC1'],  
        'X': [100.0, 100.0, 100.0, 105.0, 105.0, 110.0],
        'Y': [200.0, 200.0, 200.0, 200.0, 200.0, 200.0],
        'Z': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'Fx': [10.5, 12.8, 15.2, 8.7, 11.3, 9.8],
        'Fy': [25.3, 28.7, 32.1, 22.4, 26.8, 24.1],
        'Fz': [450.5, 520.8, 610.2, 380.2, 420.5, 290.1],
        'Mx': [125.3, 145.7, 165.4, 98.7, 112.3, 75.4],
        'My': [89.2, 102.5, 118.7, 67.3, 78.9, 45.8]
    })
    st.dataframe(example_data, use_container_width=True)
    
    st.info("💡 **Pro Tip**: The tool will automatically detect column variations like 'FZ (tonf)', 'MX (tonf·m)', etc.")
