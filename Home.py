import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Pile Analysis Tool",
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
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏗️ Pile Foundation Analysis Tool</h1>', unsafe_allow_html=True)

# Initialize session state
if 'df_results' not in st.session_state:
    st.session_state.df_results = None
if 'critical_footing' not in st.session_state:
    st.session_state.critical_footing = None

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
        # Try different encodings
        for encoding in ['latin-1', 'cp1252', 'utf-8']:
            try:
                df = pd.read_csv(uploaded_file, encoding=encoding)
                return df, f"Successfully loaded with {encoding} encoding"
            except UnicodeDecodeError:
                continue
        return None, "Could not decode file with any encoding"
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

def calculate_pile_analysis(df, nodes, pile_type, pile_capacity):
    """Perform pile analysis calculations"""
    # Filter nodes
    df_filtered = df[df['Node'].isin(nodes)].copy()
    
    # Create footing factors DataFrame
    df_pile = pd.DataFrame(FOOTING_FACTORS)
    
    # Initial calculations
    df_filtered['Number of Initial Pile'] = np.round(df_filtered['FZ (tonf)'] / pile_capacity) + 1
    df_filtered["Initial Pile Capacity"] = pile_capacity * df_filtered["Number of Initial Pile"]
    df_filtered["Axial Stress"] = df_filtered["FZ (tonf)"] / df_filtered["Number of Initial Pile"]
    df_filtered["Pile Capacity Remaining"] = pile_capacity - df_filtered["Axial Stress"]
    df_filtered["Pile Capacity Remaining Ratio"] = (pile_capacity - df_filtered["Axial Stress"]) / pile_capacity
    
    # Round 2 calculations
    df_filtered["Round 2 Num of Pile"] = df_filtered['Number of Initial Pile'] + 1
    df_filtered.loc[df_filtered['Pile Capacity Remaining Ratio'] < 0.1, "Round 2 Num of Pile"] = df_filtered['Number of Initial Pile'] + 2
    df_filtered["Round 2 Ratio"] = df_filtered["FZ (tonf)"] / (df_filtered["Round 2 Num of Pile"] * pile_capacity)
    df_filtered["Footing Type Use"] = "F" + df_filtered["Round 2 Num of Pile"].astype(str)
    df_filtered["Pile Type Use"] = pile_type
    
    # Merge with pile factors
    df_merged = df_filtered.merge(df_pile, left_on="Footing Type Use", right_on="Footing Type", how="left")
    
    # Calculate ratios based on pile type
    def calc_ratios(row):
        if row["Pile Type Use"] == "Spun Pile 600":
            mx_ratio = row["MX (tonf·m)"] * row["S_Fac_X"]
            my_ratio = row["MY (tonf·m)"] * row["S_Fac_Y"]
        elif row["Pile Type Use"] == "PC I 300":
            mx_ratio = row["MX (tonf·m)"] * row["I_Fac_X"]
            my_ratio = row["MY (tonf·m)"] * row["I_Fac_Y"]
        else:
            mx_ratio = None
            my_ratio = None
        return pd.Series([mx_ratio, my_ratio])
    
    df_merged[["Mx Ratio", "My Ratio"]] = df_merged.apply(calc_ratios, axis=1)
    df_merged["P over N"] = df_merged["FZ (tonf)"] / df_merged["Round 2 Num of Pile"]
    df_merged["Ratio Total"] = df_merged["P over N"] + abs(df_merged["My Ratio"]) + abs(df_merged["Mx Ratio"])
    df_merged['Diff Capacity'] = abs((df_merged["Ratio Total"]) / pile_capacity)
    
    # Round 3 calculations
    df_merged['Round 3 num of pile'] = np.where(
        df_merged['Diff Capacity'] > 0.999,
        df_merged["Round 2 Num of Pile"] + 1,
        df_merged["Round 2 Num of Pile"]
    )
    
    def calc_ratios_round3(row):
        if row["Pile Type Use"] == "Spun Pile 600":
            mx_ratio = row["MX (tonf·m)"] / row['Round 3 num of pile'] * row["S_Fac_X"]
            my_ratio = row["MY (tonf·m)"] / row['Round 3 num of pile'] * row["S_Fac_Y"]
        elif row["Pile Type Use"] == "PC I 300":
            mx_ratio = row["MX (tonf·m)"] / row['Round 3 num of pile'] * row["I_Fac_X"]
            my_ratio = row["MY (tonf·m)"] / row['Round 3 num of pile'] * row["I_Fac_Y"]
        else:
            mx_ratio = None
            my_ratio = None
        return pd.Series([mx_ratio, my_ratio])
    
    df_merged[["Mx Ratio Round 3", "My Ratio Round 3"]] = df_merged.apply(calc_ratios_round3, axis=1)
    df_merged["P over N Round 3"] = df_merged["FZ (tonf)"] / df_merged["Round 3 num of pile"]
    df_merged["Ratio Total Round 3"] = df_merged["P over N Round 3"] + abs(df_merged["My Ratio Round 3"]) + abs(df_merged["Mx Ratio Round 3"])
    df_merged['Diff Capacity Round 3'] = abs((df_merged["Ratio Total Round 3"]) / pile_capacity)
    df_merged["Footing Type Use Round 3"] = "F" + df_merged["Round 3 num of pile"].astype(str)
    
    return df_merged

def get_critical_footing(df_merged, nodes):
    """Get critical footing cases for each node"""
    def extract_footing_number(footing_type):
        try:
            if pd.isna(footing_type):
                return 0
            return int(str(footing_type).replace('F', ''))
        except:
            return 0
    
    def get_most_critical_footing(group):
        group = group.copy()
        group['Footing_Number'] = group['Footing Type Use Round 3'].apply(extract_footing_number)
        sorted_group = group.sort_values(['Footing_Number', 'FZ (tonf)'], ascending=[False, False])
        return sorted_group.iloc[0]
    
    critical_footing = df_merged.groupby('Node').apply(get_most_critical_footing).reset_index(drop=True)
    
    if 'Footing_Number' in critical_footing.columns:
        critical_footing = critical_footing.drop('Footing_Number', axis=1)
    
    return critical_footing

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
        
        # Check required columns
        required_cols = ['Node', 'FZ (tonf)', 'MX (tonf·m)', 'MY (tonf·m)']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.info("Available columns: " + ", ".join(df.columns))
        else:
            # Run analysis button
            if st.sidebar.button("🚀 Run Analysis", type="primary"):
                with st.spinner("Performing pile analysis..."):
                    try:
                        # Perform analysis
                        df_results = calculate_pile_analysis(df, selected_nodes, pile_type, pile_capacity)
                        critical_footing = get_critical_footing(df_results, selected_nodes)
                        
                        # Store in session state
                        st.session_state.df_results = df_results
                        st.session_state.critical_footing = critical_footing
                        
                        st.success("Analysis completed successfully!")
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")

# Display results if available
if st.session_state.df_results is not None and st.session_state.critical_footing is not None:
    df_results = st.session_state.df_results
    critical_footing = st.session_state.critical_footing
    
    # Create tabs for results
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📈 Visualizations", "📋 Detailed Results", "💾 Export"])
    
    with tab1:
        st.markdown('<h2 class="section-header">Analysis Summary</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Total Nodes Analyzed", len(critical_footing))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            max_piles = critical_footing['Round 3 num of pile'].max()
            st.metric("Max Piles Required", int(max_piles))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            avg_piles = critical_footing['Round 3 num of pile'].mean()
            st.metric("Average Piles", f"{avg_piles:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            total_piles = critical_footing['Round 3 num of pile'].sum()
            st.metric("Total Piles", int(total_piles))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Footing type distribution
        st.subheader("Footing Type Distribution")
        footing_dist = critical_footing['Footing Type Use Round 3'].value_counts().sort_index()
        
        col1, col2 = st.columns([1, 1])
        with col1:
            fig_pie = px.pie(
                values=footing_dist.values,
                names=footing_dist.index,
                title="Distribution of Footing Types"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(
                x=footing_dist.index,
                y=footing_dist.values,
                title="Footing Type Counts",
                labels={'x': 'Footing Type', 'y': 'Count'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="section-header">Data Visualizations</h2>', unsafe_allow_html=True)
        
        # Load vs Pile Number scatter plot
        fig_scatter = px.scatter(
            critical_footing,
            x='FZ (tonf)',
            y='Round 3 num of pile',
            color='Footing Type Use Round 3',
            title='Axial Load vs Number of Piles',
            labels={'FZ (tonf)': 'Axial Load (tonf)', 'Round 3 num of pile': 'Number of Piles'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Capacity utilization
        if 'Ratio Total Round 3' in critical_footing.columns:
            fig_util = px.histogram(
                critical_footing,
                x='Ratio Total Round 3',
                title='Pile Capacity Utilization Distribution',
                labels={'Ratio Total Round 3': 'Total Ratio', 'count': 'Frequency'}
            )
            fig_util.add_vline(x=pile_capacity, line_dash="dash", line_color="red", 
                              annotation_text=f"Capacity Limit ({pile_capacity})")
            st.plotly_chart(fig_util, use_container_width=True)
        
        # Node-wise analysis for high-load cases
        high_load_nodes = critical_footing.nlargest(10, 'FZ (tonf)')
        fig_node = px.bar(
            high_load_nodes,
            x='Node',
            y='FZ (tonf)',
            color='Round 3 num of pile',
            title='Top 10 Nodes by Axial Load',
            labels={'FZ (tonf)': 'Axial Load (tonf)', 'Round 3 num of pile': 'Piles Required'}
        )
        st.plotly_chart(fig_node, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="section-header">Detailed Analysis Results</h2>', unsafe_allow_html=True)
        
        # Display options
        col1, col2 = st.columns([1, 1])
        with col1:
            show_all = st.checkbox("Show All Calculations", value=False)
        with col2:
            search_node = st.number_input("Search Specific Node", min_value=0, value=0)
        
        if search_node > 0:
            node_data = critical_footing[critical_footing['Node'] == search_node]
            if not node_data.empty:
                st.subheader(f"Results for Node {search_node}")
                st.dataframe(node_data, use_container_width=True)
            else:
                st.warning(f"Node {search_node} not found in results")
        else:
            if show_all:
                st.subheader("Complete Analysis Results")
                st.dataframe(df_results, use_container_width=True)
            else:
                st.subheader("Critical Footing Cases")
                display_cols = ['Node', 'FZ (tonf)', 'MX (tonf·m)', 'MY (tonf·m)', 
                               'Round 3 num of pile', 'Footing Type Use Round 3', 
                               'Ratio Total Round 3', 'Diff Capacity Round 3']
                available_cols = [col for col in display_cols if col in critical_footing.columns]
                st.dataframe(critical_footing[available_cols], use_container_width=True)
    
    with tab4:
        st.markdown('<h2 class="section-header">Export Results</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Critical Footing Results")
            csv_critical = critical_footing.to_csv(index=False)
            st.download_button(
                label="📥 Download Critical Results as CSV",
                data=csv_critical,
                file_name=f"critical_footing_{pile_type.replace(' ', '_')}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.subheader("Complete Analysis Results")
            csv_complete = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Complete Results as CSV",
                data=csv_complete,
                file_name=f"complete_analysis_{pile_type.replace(' ', '_')}.csv",
                mime="text/csv"
            )
        
        # Summary report
        st.subheader("Analysis Summary Report")
        summary_report = f"""
        # Pile Analysis Summary Report
        
        **Analysis Parameters:**
        - Pile Type: {pile_type}
        - Pile Capacity: {pile_capacity} tonf
        - Total Nodes Analyzed: {len(critical_footing)}
        
        **Results Summary:**
        - Maximum Piles Required: {int(critical_footing['Round 3 num of pile'].max())}
        - Average Piles per Node: {critical_footing['Round 3 num of pile'].mean():.1f}
        - Total Piles Required: {int(critical_footing['Round 3 num of pile'].sum())}
        
        **Footing Type Distribution:**
        {footing_dist.to_string()}
        """
        
        st.download_button(
            label="📄 Download Summary Report",
            data=summary_report,
            file_name=f"analysis_summary_{pile_type.replace(' ', '_')}.md",
            mime="text/markdown"
        )

else:
    # Show instructions when no file is uploaded
    st.markdown("""
    ## 🚀 Getting Started
    
    1. **Upload your CSV file** using the file uploader in the sidebar
    2. **Select pile type** (Spun Pile 600 or PC I 300)
    3. **Configure pile capacity** (default: 120 tonf)
    4. **Choose nodes** (use default or specify custom nodes)
    5. **Click "Run Analysis"** to start the calculation
    
    ### 📋 Required CSV Columns
    Your CSV file must contain these columns:
    - `Node`: Node identification number
    - `FZ (tonf)`: Axial force in tons
    - `MX (tonf·m)`: Moment about X-axis
    - `MY (tonf·m)`: Moment about Y-axis
    
    ### 🔧 Analysis Features
    - **Iterative pile calculation** with capacity optimization
    - **Critical case identification** for each node
    - **Interactive visualizations** and charts
    - **Comprehensive reporting** and export options
    - **Support for multiple pile types** with different factors
    """)
    
    # Show example data format
    st.subheader("📊 Example Data Format")
    example_data = pd.DataFrame({
        'Node': [26, 27, 28, 29, 30],
        'FZ (tonf)': [450.5, 380.2, 520.8, 290.1, 410.3],
        'MX (tonf·m)': [125.3, 98.7, 156.2, 75.4, 132.1],
        'MY (tonf·m)': [89.2, 67.3, 112.5, 45.8, 95.6]
    })
    st.dataframe(example_data, use_container_width=True)
