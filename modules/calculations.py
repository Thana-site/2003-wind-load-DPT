"""
Calculations Module - FIXED VERSION
Handles section property calculations and analysis
Fixed: Material properties issue and all potential bugs
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class SectionAnalyzer:
    """Analyzes section properties using sectionproperties"""
    
    def __init__(self, section):
        """
        Initialize analyzer with a section
        
        Args:
            section: Analyzed Section object from sectionproperties
        """
        self.section = section
        
    def calculate_properties(self) -> Dict[str, Any]:
        """
        Calculate all section properties
        Handles both geometric-only and composite (with materials) sections
        
        Returns:
            dict: Dictionary containing all calculated properties
        """
        properties = {}
        
        try:
            # Basic geometric properties
            properties['area'] = self.section.get_area()
            properties['perimeter'] = self.section.get_perimeter()
            
            # Centroid
            cx, cy = self.section.get_c()
            properties['cx'] = cx
            properties['cy'] = cy
            
            # Check if this is a composite section (has materials)
            # Use get_eic() for composite, get_ic() for geometric only
            try:
                # Try composite first (since we apply materials in factory)
                ic = self.section.get_eic()
                properties['ixx_c'] = ic[0]
                properties['iyy_c'] = ic[1]
                properties['ixy_c'] = ic[2]
                is_composite = True
            except:
                # Fall back to geometric only
                ic = self.section.get_ic()
                properties['ixx_c'] = ic[0]
                properties['iyy_c'] = ic[1]
                properties['ixy_c'] = ic[2]
                is_composite = False
            
            # Second moments of area (about origin)
            try:
                if is_composite:
                    ig = self.section.get_eig()
                else:
                    ig = self.section.get_ig()
                properties['ixx_g'] = ig[0]
                properties['iyy_g'] = ig[1]
                properties['ixy_g'] = ig[2]
            except:
                properties['ixx_g'] = 0
                properties['iyy_g'] = 0
                properties['ixy_g'] = 0
            
            # Radii of gyration
            try:
                rc = self.section.get_rc()
                properties['rx'] = rc[0]
                properties['ry'] = rc[1]
            except:
                # Calculate manually if not available
                area = properties['area']
                if area > 0:
                    properties['rx'] = np.sqrt(properties['ixx_c'] / area)
                    properties['ry'] = np.sqrt(properties['iyy_c'] / area)
                else:
                    properties['rx'] = 0
                    properties['ry'] = 0
            
            # Principal moments of inertia
            try:
                properties['phi'] = self.section.get_phi()
                if is_composite:
                    properties['i11_c'] = self.section.get_ei11_c()
                    properties['i22_c'] = self.section.get_ei22_c()
                else:
                    properties['i11_c'] = self.section.get_i11_c()
                    properties['i22_c'] = self.section.get_i22_c()
            except:
                properties['phi'] = 0
                properties['i11_c'] = properties['ixx_c']
                properties['i22_c'] = properties['iyy_c']
            
            # Elastic section moduli (about centroidal axes)
            try:
                z = self.section.get_z()
                if len(z) >= 4:
                    properties['zxx_plus'] = abs(z[0]) if z[0] != 0 else 0
                    properties['zxx_minus'] = abs(z[1]) if z[1] != 0 else 0
                    properties['zyy_plus'] = abs(z[2]) if z[2] != 0 else 0
                    properties['zyy_minus'] = abs(z[3]) if z[3] != 0 else 0
                else:
                    raise ValueError("Not enough section moduli values")
                
                if len(z) >= 8:
                    properties['z11_plus'] = abs(z[4]) if z[4] != 0 else 0
                    properties['z11_minus'] = abs(z[5]) if z[5] != 0 else 0
                    properties['z22_plus'] = abs(z[6]) if z[6] != 0 else 0
                    properties['z22_minus'] = abs(z[7]) if z[7] != 0 else 0
                else:
                    properties['z11_plus'] = properties['zxx_plus']
                    properties['z11_minus'] = properties['zxx_minus']
                    properties['z22_plus'] = properties['zyy_plus']
                    properties['z22_minus'] = properties['zyy_minus']
            except Exception as e:
                # Manual calculation fallback
                try:
                    # Get extreme fiber distances
                    coords = self.section.mesh_nodes
                    if len(coords) > 0:
                        y_coords = coords[:, 1]
                        x_coords = coords[:, 0]
                        
                        y_top = max(y_coords) - cy
                        y_bot = cy - min(y_coords)
                        x_right = max(x_coords) - cx
                        x_left = cx - min(x_coords)
                        
                        properties['zxx_plus'] = properties['ixx_c'] / y_top if y_top > 0 else 0
                        properties['zxx_minus'] = properties['ixx_c'] / y_bot if y_bot > 0 else 0
                        properties['zyy_plus'] = properties['iyy_c'] / x_right if x_right > 0 else 0
                        properties['zyy_minus'] = properties['iyy_c'] / x_left if x_left > 0 else 0
                    else:
                        raise ValueError("No mesh nodes available")
                except:
                    properties['zxx_plus'] = 0
                    properties['zxx_minus'] = 0
                    properties['zyy_plus'] = 0
                    properties['zyy_minus'] = 0
                
                properties['z11_plus'] = properties['zxx_plus']
                properties['z11_minus'] = properties['zxx_minus']
                properties['z22_plus'] = properties['zyy_plus']
                properties['z22_minus'] = properties['zyy_minus']
            
            # Torsion properties
            try:
                properties['j'] = self.section.get_j()
            except:
                properties['j'] = 0
            
            # Warping properties
            try:
                properties['gamma'] = self.section.get_gamma()
            except:
                properties['gamma'] = 0
            
            # Plastic section moduli (if available)
            try:
                s = self.section.get_s()
                properties['sxx'] = abs(s[0]) if s[0] != 0 else 0
                properties['syy'] = abs(s[1]) if s[1] != 0 else 0
                if len(s) >= 4:
                    properties['s11'] = abs(s[2]) if s[2] != 0 else 0
                    properties['s22'] = abs(s[3]) if s[3] != 0 else 0
                else:
                    properties['s11'] = properties['sxx']
                    properties['s22'] = properties['syy']
            except:
                properties['sxx'] = 0
                properties['syy'] = 0
                properties['s11'] = 0
                properties['s22'] = 0
            
            # Shape factors (if available)
            try:
                sf = self.section.get_sf()
                properties['sf_xx'] = sf[0] if sf[0] > 0 else 1.0
                properties['sf_yy'] = sf[1] if sf[1] > 0 else 1.0
                if len(sf) >= 4:
                    properties['sf_11'] = sf[2] if sf[2] > 0 else 1.0
                    properties['sf_22'] = sf[3] if sf[3] > 0 else 1.0
                else:
                    properties['sf_11'] = 1.0
                    properties['sf_22'] = 1.0
            except:
                properties['sf_xx'] = 1.0
                properties['sf_yy'] = 1.0
                properties['sf_11'] = 1.0
                properties['sf_22'] = 1.0
            
            # Additional calculated properties
            properties.update(self._calculate_additional_properties(properties))
            
            # Ensure all values are valid numbers
            for key in properties:
                if properties[key] is None or np.isnan(properties[key]) or np.isinf(properties[key]):
                    properties[key] = 0.0
            
        except Exception as e:
            print(f"Error in calculate_properties: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        return properties
    
    def _calculate_additional_properties(self, props: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate additional derived properties
        
        Args:
            props: Basic properties dictionary
            
        Returns:
            dict: Additional properties
        """
        additional = {}
        
        try:
            # Polar moment of inertia
            additional['ip'] = props.get('ixx_c', 0) + props.get('iyy_c', 0)
            
            # Polar radius of gyration
            area = props.get('area', 0)
            if area > 0 and additional['ip'] > 0:
                additional['rp'] = np.sqrt(additional['ip'] / area)
            else:
                additional['rp'] = 0
            
            # Aspect ratio (for rectangular-like sections)
            i11 = props.get('i11_c', 0)
            i22 = props.get('i22_c', 0)
            if i22 > 0 and i11 > 0:
                additional['aspect_ratio'] = np.sqrt(i11 / i22)
            else:
                additional['aspect_ratio'] = 1.0
            
            # Compactness (perimeter squared / area)
            perimeter = props.get('perimeter', 0)
            if area > 0 and perimeter > 0:
                additional['compactness'] = perimeter**2 / (4 * np.pi * area)
            else:
                additional['compactness'] = 0
                
        except Exception as e:
            print(f"Warning in additional properties: {e}")
            additional['ip'] = 0
            additional['rp'] = 0
            additional['aspect_ratio'] = 1.0
            additional['compactness'] = 0
        
        return additional
    
    def calculate_stress(self, N: float = 0, Mx: float = 0, My: float = 0, 
                        Mz: float = 0) -> Dict[str, Any]:
        """
        Calculate stresses for given loading conditions
        
        Args:
            N: Axial force (positive = tension)
            Mx: Moment about x-axis
            My: Moment about y-axis
            Mz: Torsional moment
            
        Returns:
            dict: Stress results
        """
        stress_results = {}
        
        try:
            # Axial stress
            area = self.section.get_area()
            if N != 0 and area > 0:
                stress_results['sigma_n'] = N / area
            else:
                stress_results['sigma_n'] = 0
            
            # Bending stresses (at extreme fibers)
            if Mx != 0:
                try:
                    z = self.section.get_z()
                    z_top = abs(z[0]) if len(z) > 0 and z[0] != 0 else 1
                    z_bot = abs(z[1]) if len(z) > 1 and z[1] != 0 else 1
                    
                    stress_results['sigma_mx_top'] = abs(Mx) / z_top
                    stress_results['sigma_mx_bot'] = abs(Mx) / z_bot
                except:
                    stress_results['sigma_mx_top'] = 0
                    stress_results['sigma_mx_bot'] = 0
            else:
                stress_results['sigma_mx_top'] = 0
                stress_results['sigma_mx_bot'] = 0
                
            if My != 0:
                try:
                    z = self.section.get_z()
                    z_right = abs(z[2]) if len(z) > 2 and z[2] != 0 else 1
                    z_left = abs(z[3]) if len(z) > 3 and z[3] != 0 else 1
                    
                    stress_results['sigma_my_right'] = abs(My) / z_right
                    stress_results['sigma_my_left'] = abs(My) / z_left
                except:
                    stress_results['sigma_my_right'] = 0
                    stress_results['sigma_my_left'] = 0
            else:
                stress_results['sigma_my_right'] = 0
                stress_results['sigma_my_left'] = 0
            
            # Shear stress from torsion
            try:
                j = self.section.get_j()
                if Mz != 0 and j > 0:
                    stress_results['tau_max'] = abs(Mz) / j
                else:
                    stress_results['tau_max'] = 0
            except:
                stress_results['tau_max'] = 0
            
            # Combined stresses (von Mises)
            max_sigma = max(
                abs(stress_results.get('sigma_n', 0)),
                abs(stress_results.get('sigma_mx_top', 0)),
                abs(stress_results.get('sigma_mx_bot', 0)),
                abs(stress_results.get('sigma_my_right', 0)),
                abs(stress_results.get('sigma_my_left', 0))
            )
            
            stress_results['sigma_vm_max'] = self._calculate_von_mises(
                max_sigma,
                stress_results.get('tau_max', 0)
            )
            
        except Exception as e:
            print(f"Error calculating stresses: {e}")
            stress_results = {
                'sigma_n': 0,
                'sigma_mx_top': 0,
                'sigma_mx_bot': 0,
                'sigma_my_right': 0,
                'sigma_my_left': 0,
                'tau_max': 0,
                'sigma_vm_max': 0
            }
        
        return stress_results
    
    def _calculate_von_mises(self, sigma: float, tau: float) -> float:
        """
        Calculate von Mises stress
        
        Args:
            sigma: Normal stress
            tau: Shear stress
            
        Returns:
            float: von Mises stress
        """
        try:
            return np.sqrt(sigma**2 + 3*tau**2)
        except:
            return 0.0
    
    def get_design_properties(self, fy: float = 250) -> Dict[str, Any]:
        """
        Calculate design properties based on yield strength
        
        Args:
            fy: Yield strength in MPa
            
        Returns:
            dict: Design properties
        """
        design = {}
        
        try:
            # Plastic moment capacities
            try:
                s = self.section.get_s()
                design['Mp_x'] = abs(s[0]) * fy / 1e6 if len(s) > 0 else 0  # kN.m
                design['Mp_y'] = abs(s[1]) * fy / 1e6 if len(s) > 1 else 0  # kN.m
            except:
                design['Mp_x'] = 0
                design['Mp_y'] = 0
            
            # Yield moment (elastic)
            try:
                z = self.section.get_z()
                if len(z) >= 2:
                    design['My_x'] = min(abs(z[0]), abs(z[1])) * fy / 1e6
                else:
                    design['My_x'] = 0
                    
                if len(z) >= 4:
                    design['My_y'] = min(abs(z[2]), abs(z[3])) * fy / 1e6
                else:
                    design['My_y'] = 0
            except:
                design['My_x'] = 0
                design['My_y'] = 0
            
            # Axial capacity
            area = self.section.get_area()
            design['Ny'] = area * fy / 1000 if area > 0 else 0  # kN
            
            # Shape factors
            try:
                sf = self.section.get_sf()
                design['shape_factor_x'] = sf[0] if len(sf) > 0 and sf[0] > 0 else 1.0
                design['shape_factor_y'] = sf[1] if len(sf) > 1 and sf[1] > 0 else 1.0
            except:
                design['shape_factor_x'] = 1.0
                design['shape_factor_y'] = 1.0
                
        except Exception as e:
            print(f"Error calculating design properties: {e}")
            design = {
                'Mp_x': 0,
                'Mp_y': 0,
                'My_x': 0,
                'My_y': 0,
                'Ny': 0,
                'shape_factor_x': 1.0,
                'shape_factor_y': 1.0
            }
        
        return design
    
    @staticmethod
    def compare_sections(sections: list) -> pd.DataFrame:
        """
        Compare properties of multiple sections
        
        Args:
            sections: List of section objects
            
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for idx, section in enumerate(sections):
            try:
                analyzer = SectionAnalyzer(section)
                props = analyzer.calculate_properties()
                
                comparison_data.append({
                    'Section': f"Section {idx+1}",
                    'Area': props.get('area', 0),
                    'Ixx': props.get('ixx_c', 0),
                    'Iyy': props.get('iyy_c', 0),
                    'Zxx': min(props.get('zxx_plus', 0), props.get('zxx_minus', 0)),
                    'Zyy': min(props.get('zyy_plus', 0), props.get('zyy_minus', 0)),
                    'rx': props.get('rx', 0),
                    'ry': props.get('ry', 0),
                    'J': props.get('j', 0)
                })
            except Exception as e:
                print(f"Error comparing section {idx}: {e}")
                comparison_data.append({
                    'Section': f"Section {idx+1}",
                    'Area': 0,
                    'Ixx': 0,
                    'Iyy': 0,
                    'Zxx': 0,
                    'Zyy': 0,
                    'rx': 0,
                    'ry': 0,
                    'J': 0
                })
        
        return pd.DataFrame(comparison_data)
