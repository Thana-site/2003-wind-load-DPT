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
DEFAULT_NODES = [789, 790, 791,
                4561, 4572, 4576, 4581, 4586,
                4627, 4632, 4637,
                4657, 4663,
                4748, 4749, 4752,
                4827, 4831,
                5568, 5569,
                5782, 5784,
                7446, 7447, 7448, 7453, 7461, 7464]

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
        critical_case['Max_Fz'] = group['Fz'].abs().max()
        critical_case['Min_Fz'] = group['Fz'].abs().min()
        critical_case['Avg_Fz'] = group['Fz'].abs().mean()
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

def create_basic_visualizations(final_results):
    """Create basic visualization plots that are more robust"""
    plots = {}
    
    try:
        # 1. Utilization Distribution
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=final_results['Utilization_Ratio'],
            nbinsx=20,
            name='Utilization Distribution',
            marker_color='lightblue',
            opacity=0.7
        ))
        fig_dist.add_vline(x=0.85, line_dash="dash", line_color="red", 
                          annotation_text="Target (85%)")
        fig_dist.update_layout(
            title='Utilization Ratio Distribution',
            xaxis_title='Utilization Ratio',
            yaxis_title='Count',
            showlegend=False
        )
        plots['utilization_dist'] = fig_dist
    except Exception as e:
        st.warning(f"Could not create utilization distribution plot: {str(e)}")
    
    try:
        # 2. Footing Type Distribution
        footing_counts = final_results['Footing_Type'].value_counts()
        fig_footing = go.Figure(data=[
            go.Bar(x=footing_counts.index, y=footing_counts.values,
                  marker_color='lightgreen')
        ])
        fig_footing.update_layout(
            title='Footing Type Distribution',
            xaxis_title='Footing Type',
            yaxis_title='Count'
        )
        plots['footing_dist'] = fig_footing
    except Exception as e:
        st.warning(f"Could not create footing distribution plot: {str(e)}")
    
    try:
        # 3. Load vs Number of Piles
        fig_load_piles = px.scatter(
            final_results,
            x='Max_Fz',
            y='Num_Piles',
            color='Utilization_Ratio',
            color_continuous_scale='RdYlGn_r',
            hover_data=['Node', 'Footing_Type'],
            title='Load vs Number of Piles',
            labels={'Max_Fz': 'Maximum Load (tonf)', 'Num_Piles': 'Number of Piles'}
        )
        plots['load_piles'] = fig_load_piles
    except Exception as e:
        st.warning(f"Could not create load vs piles plot: {str(e)}")
    
    try:
        # 4. Utilization Category Pie Chart
        if 'Utilization_Category' in final_results.columns:
            category_counts = final_results['Utilization_Category'].value_counts()
            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title='Utilization Categories',
                color_discrete_map={
                    'Over-Conservative': '#28a745',
                    'Conservative': '#ffc107',
                    'Optimal': '#17a2b8',
                    'Near-Capacity': '#fd7e14',
                    'Over-Capacity': '#dc3545'
                }
            )
            plots['category_pie'] = fig_pie
    except Exception as e:
        st.warning(f"Could not create category pie chart: {str(e)}")
    
    return plots

def create_advanced_visualizations(final_results):
    """Create advanced visualizations with proper error handling"""
    plots = {}
    
    # Check if X and Y columns exist
    has_coordinates = 'X' in final_results.columns and 'Y' in final_results.columns
    
    if has_coordinates:
        try:
            # XY Bubble Chart
            fig_xy = px.scatter(
                final_results,
                x='X',
                y='Y',
                size='Num_Piles',
                color='Utilization_Ratio',
                color_continuous_scale=['green', 'yellow', 'orange', 'red'],
                size_max=30,
                hover_data=['Node', 'Footing_Type', 'Max_Fz'],
                title='XY Plan View - Pile Utilization Analysis'
            )
            fig_xy.update_coloraxes(
                colorbar=dict(
                    title="Utilization<br>Ratio",
                    tickmode="array",
                    tickvals=[0.5, 0.7, 0.85, 1.0],
                    ticktext=["50%", "70%", "85%", "100%"]
                )
            )
            plots['xy_bubble'] = fig_xy
        except Exception as e:
            st.warning(f"Could not create XY bubble chart: {str(e)}")
        
        # 3D Visualization if Z exists
        if 'Z' in final_results.columns:
            try:
                fig_3d = px.scatter_3d(
                    final_results,
                    x='X',
                    y='Y',
                    z='Z',
                    color='Utilization_Category' if 'Utilization_Category' in final_results.columns else 'Utilization_Ratio',
                    size='Num_Piles',
                    hover_data=['Node', 'Footing_Type'],
                    title='3D Site Layout'
                )
                plots['3d_view'] = fig_3d
            except Exception as e:
                st.warning(f"Could not create 3D plot: {str(e)}")
    
    # Load vs Utilization scatter
    try:
        fig_opt = px.scatter(
            final_results,
            x='Max_Fz',
            y='Utilization_Ratio',
            color='Footing_Type',
            size='Num_Piles',
            hover_data=['Node'],
            title='Load vs Utilization Optimization'
        )
        fig_opt.add_hrect(y0=0.8, y1=0.95, fillcolor="lightgreen", opacity=0.2,
                         annotation_text="Target Zone")
        fig_opt.add_hline(y=1.0, line_dash="dash", line_color="red")
        plots['load_optimization'] = fig_opt
    except Exception as e:
        st.warning(f"Could not create optimization plot: {str(e)}")
    
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
    min_value=30,
    max_value=500,
    value=120,
    step=10,
    help="Enter the pile capacity in tons"
)

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
        
        # Check for required columns
        required_cols = ['Node']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            # Standardize column names
            df_standardized = df.copy()
            
            column_mapping = {
                'FX (tonf)': 'Fx',
                'FY (tonf)': 'Fy', 
                'FZ (tonf)': 'Fz',
                'MX (tonf·m)': 'Mx',
                'MY (tonf·m)': 'My',
                'MZ (tonf·m)': 'Mz'
            }
            
            # Apply column mapping
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df_standardized[new_col] = df[old_col]
            
            # Ensure required columns exist
            for col in ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']:
                if col not in df_standardized.columns:
                    df_standardized[col] = 0
            
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
    tabs = st.tabs(["📊 Summary", "📈 Basic Charts", "🗺️ Advanced Visuals", "📋 Detailed Results", "💾 Export"])
    
    with tabs[0]:
        st.markdown('<h2 class="section-header">📊 Optimization Summary</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nodes Analyzed", len(final_results))
        
        with col2:
            avg_util = final_results['Utilization_Ratio'].mean()
            st.metric("Avg Utilization", f"{avg_util:.1%}")
        
        with col3:
            total_piles = final_results['Num_Piles'].sum()
            st.metric("Total Piles", int(total_piles))
        
        with col4:
            st.metric("Target", f"{target_utilization:.0%}")
        
        # Utilization breakdown
        st.subheader("🎯 Utilization Categories")
        if 'Utilization_Category' in final_results.columns:
            category_counts = final_results['Utilization_Category'].value_counts()
            for category, count in category_counts.items():
                percentage = (count / len(final_results)) * 100
                st.write(f"**{category}**: {count} nodes ({percentage:.1f}%)")
    
    with tabs[1]:
        st.markdown('<h2 class="section-header">📈 Basic Visualizations</h2>', unsafe_allow_html=True)
        
        # Create basic visualizations
        basic_plots = create_basic_visualizations(final_results)
        
        # Display basic plots in a grid
        if 'utilization_dist' in basic_plots:
            st.plotly_chart(basic_plots['utilization_dist'], use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if 'footing_dist' in basic_plots:
                st.plotly_chart(basic_plots['footing_dist'], use_container_width=True)
        
        with col2:
            if 'category_pie' in basic_plots:
                st.plotly_chart(basic_plots['category_pie'], use_container_width=True)
        
        if 'load_piles' in basic_plots:
            st.plotly_chart(basic_plots['load_piles'], use_container_width=True)
    
    with tabs[2]:
        st.markdown('<h2 class="section-header">🗺️ Advanced Visualizations</h2>', unsafe_allow_html=True)
        
        # Check if coordinates exist
        has_coords = 'X' in final_results.columns and 'Y' in final_results.columns
        
        if not has_coords:
            st.warning("⚠️ No X/Y coordinates found in data. Advanced spatial visualizations are not available.")
            st.info("💡 To enable advanced visualizations, ensure your data includes X and Y coordinate columns.")
        else:
            # Create advanced visualizations
            advanced_plots = create_advanced_visualizations(final_results)
            
            if 'xy_bubble' in advanced_plots:
                st.subheader("🗺️ XY Plan View")
                st.plotly_chart(advanced_plots['xy_bubble'], use_container_width=True)
            
            if '3d_view' in advanced_plots:
                st.subheader("🏗️ 3D Site Layout")
                st.plotly_chart(advanced_plots['3d_view'], use_container_width=True)
            
            if 'load_optimization' in advanced_plots:
                st.subheader("📊 Load vs Utilization")
                st.plotly_chart(advanced_plots['load_optimization'], use_container_width=True)
    
    with tabs[3]:
        st.markdown('<h2 class="section-header">📋 Detailed Results</h2>', unsafe_allow_html=True)
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            search_node = st.number_input("🔍 Search Node", min_value=0, value=0)
        with col2:
            show_all = st.checkbox("Show All Load Cases", value=False)
        
        if search_node > 0:
            if show_all:
                filtered = all_cases[all_cases['Node'] == search_node]
            else:
                filtered = final_results[final_results['Node'] == search_node]
        else:
            filtered = final_results if not show_all else all_cases
        
        # Display results
        st.dataframe(filtered, use_container_width=True, height=400)
    
    with tabs[4]:
        st.markdown('<h2 class="section-header">💾 Export Results</h2>', unsafe_allow_html=True)
        
        # Prepare export data
        export_cols = ['Node', 'Footing_Type', 'Num_Piles', 'Max_Fz', 'Utilization_Ratio', 'Is_Safe']
        export_data = final_results[[col for col in export_cols if col in final_results.columns]]
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            csv = export_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Design Summary (CSV)",
                data=csv,
                file_name=f"pile_design_{pile_type.replace(' ', '_')}.csv",
                mime="text/csv"
            )
        
        with col2:
            full_csv = final_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=full_csv,
                file_name=f"pile_analysis_complete_{pile_type.replace(' ', '_')}.csv",
                mime="text/csv"
            )

else:
    # Instructions
    st.markdown("""
    ## 🚀 Enhanced Pile Foundation Analysis Tool
    
    ### 📋 Instructions:
    1. Upload your CSV file with structural analysis data
    2. Configure pile type and capacity in the sidebar
    3. Set your target utilization ratio (85% recommended)
    4. Select nodes for analysis
    5. Click "Run Optimized Analysis" to start
    
    ### 📊 Required Data Format:
    Your CSV should include:
    - **Node**: Node identifier
    - **Load forces**: FX, FY, FZ (tonf)
    - **Moments**: MX, MY, MZ (tonf·m)
    - **Coordinates** (optional): X, Y, Z for spatial visualization
    
    ### 🎯 Optimization Features:
    - Target utilization optimization (80-90%)
    - Multiple footing options (F3 to F20)
    - Load case analysis
    - Efficiency categorization
    """)
