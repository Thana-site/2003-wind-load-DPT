import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import importlib
import numpy as np



st.title("Wind Load Calculation (DPT)")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Building Dimensions", "Building Classification and Wind Load Method", "Wind Load", "Export", "Data Base"])

with tab1:
    st.write("## Building Dimensions")
    # Layout: Two columns
    cols = st.columns(2)
    
    with cols[0]:
      # Input for the number of floors
      num_floors = st.number_input("Enter the number of floors:", min_value=1, max_value=20, step=1, value=3)

      # Inputs for height, Wxz, and Wyz for each floor
      floor_heights = []
      floor_wxz = []
      floor_wyz = []

      for i in range(int(num_floors)):
          st.write(f"### Floor {i+1}")
          height = st.number_input(f"Height (meters) for Floor {i+1}:", min_value=0.0, step=0.1, key=f"height_{i}")
          wxz = st.number_input(f"Width in XZ plane (meters) for Floor {i+1}:", min_value=0.0, step=0.1, key=f"wxz_{i}")
          wyz = st.number_input(f"Width in YZ plane (meters) for Floor {i+1}:", min_value=0.0, step=0.1, key=f"wyz_{i}")

          # Append the input data to respective lists
          floor_heights.append(height)
          floor_wxz.append(wxz)
          floor_wyz.append(wyz)

    with cols[1]:
      # Calculate and display total height
      total_height = sum(floor_heights)

      st.header("Results Summary")

      # Display summary only when "Submit" button is clicked
      if st.button("Submit"):
          st.write("### Summary of Building Dimensions:")
          st.write(f"Number of Floors: {num_floors}")
          st.write(f"Total Height: {total_height} meters")

          # Create 3D visualization using Matplotlib
          fig = plt.figure(figsize=(10, 8))
          ax = fig.add_subplot(111, projection='3d')

          # Plot each floor as a rectangle in 3D
          current_height = 0
          for i, (height, width_xz, width_yz) in enumerate(zip(floor_heights, floor_wxz, floor_wyz)):
              # Define vertices of the rectangle (XZ and YZ planes)
              vertices = [
                  [0, 0, current_height],
                  [width_xz, 0, current_height],
                  [width_xz, width_yz, current_height],
                  [0, width_yz, current_height],
                  [0, 0, current_height + height],
                  [width_xz, 0, current_height + height],
                  [width_xz, width_yz, current_height + height],
                  [0, width_yz, current_height + height]
              ]
              
              # Define faces of the 3D rectangle
              faces = [
                  [vertices[0], vertices[1], vertices[5], vertices[4]],  # Bottom
                  [vertices[2], vertices[3], vertices[7], vertices[6]],  # Top
                  [vertices[0], vertices[4], vertices[7], vertices[3]],  # Left
                  [vertices[1], vertices[5], vertices[6], vertices[2]],  # Right
                  [vertices[0], vertices[3], vertices[2], vertices[1]],  # Front
                  [vertices[4], vertices[5], vertices[6], vertices[7]]   # Back
              ]
              
              # Add the faces as a 3D polygon
              ax.add_collection3d(Poly3DCollection(faces, alpha=0.6, edgecolor='k'))
              
              # Update the current height
              current_height += height

          # Set axis labels
          ax.set_xlabel("Width in XZ (Wxz)")
          ax.set_ylabel("Width in YZ (Wyz)")
          ax.set_zlabel("Height (meters)")

          # Set title
          ax.set_title("3D Visualization of Building Dimensions")

          # Set limits to center the plot
          ax.set_xlim([0, max(floor_wxz)*1.2])  # Add 20% padding
          ax.set_ylim([0, max(floor_wyz)*1.2])  # Add 20% padding
          ax.set_zlim([0, total_height*1.2])  # Add 20% padding

          st.write(f"Total height of the building: {total_height} meters")
          # Display the plot in Streamlit
          st.pyplot(fig)

with tab2:
    # Automatically derive W and Ds from previous inputs
    if floor_wxz and floor_wyz:
        W = floor_wxz + floor_wyz  # Combine all widths
        Ds = min(W) if W else 0  # Minimum width as Ds
    else:
        W = []
        Ds = 0

    # H is total_height calculated from floor dimensions
    H = total_height

    # Display derived parameters
    st.write(f"### Summary Parameters from Building Dimensions:")
    st.write(f"- Combined Widths (W): {W}")
    st.write(f"- Reference Dimension (Ds): {Ds} meters")
    st.write(f"- Total Height (H): {H} meters")

    def calculate_W_over_H(H, W):
        """
        Calculate the minimum H/W ratio from the widths (W).
        """
        if W:  # Avoid division by zero or empty list
            H_W = [H / w for w in W if w > 0]
            return min(H_W)
        return float('inf')  # Return infinity if W is invalid


    def calculate_H_over_Ds(H, Ds):
        """
        Calculate the ratio of H/Ds.
        """
        if Ds > 0:  # Avoid division by zero
            return H / Ds
        return float('inf')  # Return infinity if Ds is invalid


    def classify_primary_structure(H, H_W):
        """
        Classify the primary structure based on H and H/W ratio.
        """
        if H > 80 or H_W > 3:
            return 1, "Equivalent Static Wind Load (Detailed Method)"
        return 0, "Equivalent Static Wind Load (Simple Method)"


    def classify_building(H, Ds):
        """
        Classify the structure as Low Building or High Building.
        """
        H_Ds = calculate_H_over_Ds(H, Ds)
        if H <= 23 and H_Ds < 1:
            return 0, "Low Building"
        return 1, "High Building"


    if st.button("Classify Building"):
        # Perform classifications
        W_H = calculate_W_over_H(H, W)
        primary_method_code, primary_method = classify_primary_structure(H, W_H)
        building_classification_code, building_classification = classify_building(H, Ds)

        # Display classification results
        st.write(f"### Classification Results:")
        st.write(f"- Primary Structure Method: {primary_method} (Code: {primary_method_code})")
        st.write(f"- Building Classification: {building_classification} (Code: {building_classification_code})")
        st.write(f"Total height of the building: {total_height} meters")

with tab5:
    cols = st.columns(2)
    with cols[0]:
      #V50 (wind velocity at 50 year of return period) and Typhoon Factor #data frame
      st.write("V50 and Typhoon Factor (Tf)")
      index_V = ["1", "2", "3", "4A", "4B"]
      V_50 = [25, 27, 29, 25, 25]
      TF = [1.0, 1.0, 1.0, 1.2, 1.08]

      data = {
          "V50 [m/s]": V_50,
          "Typhoon Factor": TF
      }

      df_V = pd.DataFrame(data, index=index_V)
      st.dataframe(df_V)

      #Iw (Importance Factor)
      st.write("Iw (Importance Factor)")
      index_I = ["Low","Normal","High","Very High"]
      I_Strength = [0.8, 1, 1.15, 1.15]
      I_Service = [0.75, 0.75, 0.75, 0.75]

      data_Iw = {
          "I(Strength)": I_Strength,
          "I(Service)": I_Service
      }

      df_Iw = pd.DataFrame(data_Iw, index=index_I)
      st.dataframe(df_Iw)
      
      #Cpi (Pressure Coefficent : Indoor)
      st.write("Cpi (Pressure Coefficent : Indoor)")
      Case = ["Case1", "Case2", "Case3"]
      Cpi_max = [0,0.3,0.7]
      Cpi_min = [-0.15,-0.45,-0.7]

      data = {
          "Cpi_max": Cpi_max,
          "Cpi_min": Cpi_min
      }

      df_Cpi = pd.DataFrame(data, index=Case)
      st.dataframe(df_Cpi)
      
      # Cgi (Gust Effect: Indoor)
      st.write("Cgi (Gust Effect: Indoor)")
      Cgi = 2.0
      st.write("Cgi :",Cgi)

      # Function to calculate Ce values based on height (z)
      st.write("Ce (The multiplication due to geography)")
      def Ce_Value(z):
          A = (z / 10) ** 0.2
          if A < 0.9:
              A = 0.9

          B = 0.7 * (z / 12) ** 0.3
          if B < 0.7:
              B = 0.7
          
          return A, B  # Returning values A and B

      # Example value for total_height
      z = total_height  # You can change this value as per your requirements

      # Calculate Ce values by calling Ce_Value
      A, B = Ce_Value(z)

      # Creating a DataFrame from the results
      Geography = ["A", "B"]
      data_Ce = {
          "Ce": [A, B]
      }

      # Creating the DataFrame
      df_Ce = pd.DataFrame(data_Ce, index=Geography)

      # Display the DataFrame in the Streamlit app
      st.write(df_Ce)

    with cols[1]:
      st.write("## CgCp According to Apendix ข.1")
      #--- Appendix (ข1) CpCg
      #Case 1 the wind direction subjected perpendicular to roof ridge
      st.write("Case 1 the wind direction subjected perpendicular to roof ridge")
      # Define the slope values
      Slope_of_Roof_ridge = list(range(0, 6)) + [20] + list(range(30, 46)) + [90]

      # Define the corresponding values for different ranges
      values = {
          "0-5": [0.75, 1.15, -1.3, -2.0, -0.7, -1.0, -0.55, -0.8],
          "20": [1.0, 1.5, -1.3, -2.0, -0.9, -1.3, -0.8, -1.2],
          "30-45": [1.05, 1.3, 0.4, 0.5, -0.8, -1.0, -0.7, -0.9],
          "90": [1.05, 1.3, 0.4, 0.5, -0.8, -1.0, -0.7, -0.9],
      }

      headers = ["1", "1E", "2", "2E", "3", "3E", "4", "4E"]

      # Create a DataFrame and populate the values
      data = []
      for slope in Slope_of_Roof_ridge:
          if slope in range(0, 6):
              data.append(values["0-5"])
          elif slope == 20:
              data.append(values["20"])
          elif slope in range(30, 46):
              data.append(values["30-45"])
          elif slope == 90:
              data.append(values["90"])

      # Create the DataFrame
      dataframe_AP1_Case1 = pd.DataFrame(data, columns=headers, index=Slope_of_Roof_ridge)

      # Name the index column
      dataframe_AP1_Case1.index.name = "Slope of Roof Ridge"

      # Display the DataFrame
      st.write(dataframe_AP1_Case1)

      #Case 2 the wind direction subjected parallel to roof ridge
      st.write("Case 2 the wind direction subjected parallel to roof ridge")

      # Define the slope values
      Slope_of_Roof_ridge = list(range(0, 91))  # Angles from 0 to 90 degrees

      # Define the corresponding values for Case 2
      values = [-0.85, -0.9, -1.3, -2.0, -0.7, -1.0, -0.85, -0.9, 0.75, 1.15, -0.55, -0.8]

      # Define the headers (Assuming the length matches)
      headers = ["1", "1E", "2", "2E", "3", "3E", "4", "4E", "5", "5E", "6", "6E"]

      # Create the DataFrame and replicate the values across all slopes
      dataframe_AP1_Case2 = pd.DataFrame([values] * len(Slope_of_Roof_ridge), columns=headers, index=Slope_of_Roof_ridge)

      # Name the index column
      dataframe_AP1_Case2.index.name = "Slope of Roof Ridge"

      # Display the DataFrame
      st.write(dataframe_AP1_Case2)

with tab3:
    #Base Equation: p + pi = IwqCe(CgCp + CgiCpi) [kg/sq.m]
    cols = st.columns(2)

    with cols[0]:
      # Get the index input from the user
      selected_index = st.selectbox("Select an index to view the data:", options=index_V)

      # Retrieve and store the data for the selected index
      if selected_index:
          selected_data = df_V.loc[selected_index]
          
          # Store the values in variables
          V_50 = selected_data["V50 [m/s]"]
          Tf = selected_data["Typhoon Factor"]
          
          # Display the stored parameters
          st.write(f"V50 = {V_50} m/s")
          st.write(f"Typhoon Factor = {Tf}")
      # Get user input for Importance Factor index
      selected_index_I = st.selectbox("Select an index for Importance Factor (Iw):", options=index_I)

      # Retrieve and store the data for the selected index
      if selected_index_I:
          selected_data_I = df_Iw.loc[selected_index_I]
          I_Strength_value = selected_data_I["I(Strength)"]
          I_Service_value = selected_data_I["I(Service)"]
          
          st.write(f"I(Strength) = {I_Strength_value}")
          st.write(f"I(Service) = {I_Service_value}")

      # Get user input for Cpi index
      selected_index_Cpi = st.selectbox("Select an index for Cpi (Pressure Coefficient):", options=Case)

      # Retrieve and store the data for the selected index
      if selected_index_Cpi:
          selected_data_Cpi = df_Cpi.loc[selected_index_Cpi]
          Cpi_max_value = selected_data_Cpi["Cpi_max"]
          Cpi_min_value = selected_data_Cpi["Cpi_min"]
          
          st.write(f"Stored Parameters for Cpi at index {selected_index_Cpi}:")
          st.write(f"Cpi_max = {Cpi_max_value}")
          st.write(f"Cpi_min = {Cpi_min_value}")

      # Selectbox for selecting the Ce value (A or B)
      selected_Ce = st.selectbox("Select the Ce value", options=["A", "B"])

      # Get the selected Ce value
      selected_value_Ce = df_Ce.loc[selected_Ce, "Ce"]

      # Display the selected Ce value
      st.write(f"Selected Ce value for {selected_Ce}: {selected_value_Ce}")


      # --- Appendix (ข1) CpCg
      # Case 1: The wind direction subjected perpendicular to roof ridge
      st.write("Case 1: The wind direction subjected perpendicular to roof ridge")

      # Slider for Slope of Roof Ridge input (Case 1)
      selected_slope = st.slider(
          "Select a Slope of Roof Ridge (Case 1):",
          min_value=min(dataframe_AP1_Case1.index),
          max_value=max(dataframe_AP1_Case1.index),
          value=min(dataframe_AP1_Case1.index),
      )

      # Create lists for storing the indexes and CgCp values for Case 1
      case1_indexes = dataframe_AP1_Case1.index.tolist()
      case1_CgCp_values = []

      # Perform interpolation for the selected slope in Case 1 and store CgCp values
      if selected_slope in dataframe_AP1_Case1.index:
          interpolated_values_case1 = dataframe_AP1_Case1.loc[selected_slope]
          case1_CgCp_values.append(interpolated_values_case1)
      else:
          # Perform linear interpolation between rows for Case 1
          lower_index = max([i for i in dataframe_AP1_Case1.index if i <= selected_slope])
          upper_index = min([i for i in dataframe_AP1_Case1.index if i >= selected_slope])

          # Linear interpolation
          lower_values = dataframe_AP1_Case1.loc[lower_index]
          upper_values = dataframe_AP1_Case1.loc[upper_index]
          weight = (selected_slope - lower_index) / (upper_index - lower_index)
          interpolated_values_case1 = lower_values + weight * (upper_values - lower_values)
          case1_CgCp_values.append(interpolated_values_case1)

      # Display the interpolated or exact values for Case 1
      st.write(f"CgCp values for slope {selected_slope} (Case 1):")
      st.write(interpolated_values_case1)

      # Case 2: Wind direction subjected parallel to roof ridge
      st.write("Case 2: The wind direction subjected parallel to roof ridge")

      # Slider for Slope of Roof Ridge input (Case 2)
      selected_slope_case2 = selected_slope

      # Create lists for storing the indexes and CgCp values for Case 2
      case2_indexes = dataframe_AP1_Case2.index.tolist()
      case2_CgCp_values = []

      # Retrieve the values for the selected slope in Case 2
      interpolated_values_case2 = dataframe_AP1_Case2.loc[selected_slope_case2]
      case2_CgCp_values.append(interpolated_values_case2)

      case1_CgCp_values = pd.DataFrame(case1_CgCp_values)
      case2_CgCp_values = pd.DataFrame(case2_CgCp_values)

      # Display the interpolated values for Case 2
      st.write(f"CgCp values for slope {selected_slope_case2} (Case 2):")
      st.write(interpolated_values_case2)


      st.write("Data Frame of CgCp values for Case 1:")
      st.write(case1_CgCp_values)

      st.write("Data Frame of CgCp values for Case 2:")
      st.write(case2_CgCp_values)

    with cols[1]:
    #Base Equation: p + pi = IwqCe(CgCp + CgiCpi) [kg/sq.m]
      st.write(f"V50 = {V_50} m/s")
      st.write(f"Typhoon Factor = {Tf}")
      q = 0.5*(1.25/9.81)*(V_50*Tf)**2
      st.write(f"I(Strength) = {I_Strength_value}")
      st.write(f"Selected Ce value for {selected_Ce}: {selected_value_Ce}")
      IwqCe = I_Strength_value*q*selected_value_Ce
      st.write("Cgi :",Cgi)
      st.write(f"Cpi_max = {Cpi_max_value}")
      st.write(f"Cpi_min = {Cpi_min_value}")
      CgiCpi_max = Cgi*Cpi_max_value
      CgiCpi_min = Cgi*Cpi_min_value

      CgCp_CgiCpi_max_Case1 = case1_CgCp_values + CgiCpi_max
      CgCp_CgiCpi_min_Case1 = case1_CgCp_values + CgiCpi_min
      CgCp_CgiCpi_max_Case2 = case2_CgCp_values + CgiCpi_max
      CgCp_CgiCpi_min_Case2 = case2_CgCp_values + CgiCpi_min

      Pnet_Case1_max = IwqCe*CgCp_CgiCpi_max_Case1
      Pnet_Case1_min = IwqCe*CgCp_CgiCpi_min_Case1
      Pnet_Case2_max = IwqCe*CgCp_CgiCpi_max_Case2
      Pnet_Case2_min = IwqCe*CgCp_CgiCpi_min_Case2

      st.write(Pnet_Case1_max)
      st.write(Pnet_Case1_min)
      st.write(Pnet_Case2_max)
      st.write(Pnet_Case2_min)


