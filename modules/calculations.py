"""
Calculations Module
Handles section property calculations and analysis
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
            
            # Second moments of area (about centroid)
            ic = self.section.get_ic()
            properties['ixx_c'] = ic[0]
            properties['iyy_c'] = ic[1]
            properties['ixy_c'] = ic[2]
            
            # Second moments of area (about origin)
            ig = self.section.get_ig()
            properties['ixx_g'] = ig[0]
            properties['iyy_g'] = ig[1]
            properties['ixy_g'] = ig[2]
            
            # Radii of gyration
            rc = self.section.get_rc()
            properties['rx'] = rc[0]
            properties['ry'] = rc[1]
            
            # Principal moments of inertia
            try:
                properties['phi'] = self.section.get_phi()
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
                    properties['zxx_plus'] = z[0]   # Top fiber
                    properties['zxx_minus'] = z[1]  # Bottom fiber
                    properties['zyy_plus'] = z[2]   # Right fiber
                    properties['zyy_minus'] = z[3]  # Left fiber
                else:
                    # Fallback calculation
                    properties['zxx_plus'] = properties['ixx_c'] / abs(cy) if cy != 0 else 0
                    properties['zxx_minus'] = properties['zxx_plus']
                    properties['zyy_plus'] = properties['iyy_c'] / abs(cx) if cx != 0 else 0
                    properties['zyy_minus'] = properties['zyy_plus']
                
                if len(z) >= 8:
                    properties['z11_plus'] = z[4]
                    properties['z11_minus'] = z[5]
                    properties['z22_plus'] = z[6]
                    properties['z22_minus'] = z[7]
                else:
                    properties['z11_plus'] = properties['zxx_plus']
                    properties['z11_minus'] = properties['zxx_minus']
                    properties['z22_plus'] = properties['zyy_plus']
                    properties['z22_minus'] = properties['zyy_minus']
            except Exception as e:
                print(f"Warning: Could not calculate section moduli: {e}")
                properties['zxx_plus'] = 0
                properties['zxx_minus'] = 0
                properties['zyy_plus'] = 0
                properties['zyy_minus'] = 0
                properties['z11_plus'] = 0
                properties['z11_minus'] = 0
                properties['z22_plus'] = 0
                properties['z22_minus'] = 0
            
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
                properties['sxx'] = s[0]
                properties['syy'] = s[1]
                if len(s) >= 4:
                    properties['s11'] = s[2]
                    properties['s22'] = s[3]
                else:
                    properties['s11'] = s[0]
                    properties['s22'] = s[1]
            except:
                properties['sxx'] = 0
                properties['syy'] = 0
                properties['s11'] = 0
                properties['s22'] = 0
            
            # Shape factors (if available)
            try:
                sf = self.section.get_sf()
                properties['sf_xx'] = sf[0]
                properties['sf_yy'] = sf[1]
                if len(sf) >= 4:
                    properties['sf_11'] = sf[2]
                    properties['sf_22'] = sf[3]
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
            
        except Exception as e:
            print(f"Error in calculate_properties: {e}")
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
            additional['ip'] = props['ixx_c'] + props['iyy_c']
            
            # Polar radius of gyration
            if props['area'] > 0:
                additional['rp'] = np.sqrt(additional['ip'] / props['area'])
            else:
                additional['rp'] = 0
            
            # Aspect ratio (for rectangular-like sections)
            if props.get('i22_c', 0) > 0:
                additional['aspect_ratio'] = np.sqrt(props['i11_c'] / props['i22_c'])
            else:
                additional['aspect_ratio'] = 1.0
            
            # Compactness (perimeter squared / area)
            if props['area'] > 0:
                additional['compactness'] = props['perimeter']**2 / (4 * np.pi * props['area'])
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
                    stress_results['sigma_mx_top'] = Mx / z[0] if z[0] > 0 else 0
                    stress_results['sigma_mx_bot'] = -Mx / z[1] if z[1] > 0 else 0
                except:
                    stress_results['sigma_mx_top'] = 0
                    stress_results['sigma_mx_bot'] = 0
            else:
                stress_results['sigma_mx_top'] = 0
                stress_results['sigma_mx_bot'] = 0
                
            if My != 0:
                try:
                    z = self.section.get_z()
                    stress_results['sigma_my_right'] = My / z[2] if len(z) > 2 and z[2] > 0 else 0
                    stress_results['sigma_my_left'] = -My / z[3] if len(z) > 3 and z[3] > 0 else 0
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
                    stress_results['tau_max'] = Mz / j
                else:
                    stress_results['tau_max'] = 0
            except:
                stress_results['tau_max'] = 0
            
            # Combined stresses (von Mises)
            stress_results['sigma_vm_max'] = self._calculate_von_mises(
                stress_results['sigma_n'] + max(
                    stress_results.get('sigma_mx_top', 0),
                    stress_results.get('sigma_mx_bot', 0)
                ) + max(
                    stress_results.get('sigma_my_right', 0),
                    stress_results.get('sigma_my_left', 0)
                ),
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
        return np.sqrt(sigma**2 + 3*tau**2)
    
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
                design['Mp_x'] = s[0] * fy / 1e6  # kN.m
                design['Mp_y'] = s[1] * fy / 1e6  # kN.m
            except:
                design['Mp_x'] = 0
                design['Mp_y'] = 0
            
            # Yield moment (elastic)
            try:
                z = self.section.get_z()
                design['My_x'] = min(z[0], z[1]) * fy / 1e6 if len(z) >= 2 else 0
                design['My_y'] = min(z[2], z[3]) * fy / 1e6 if len(z) >= 4 else 0
            except:
                design['My_x'] = 0
                design['My_y'] = 0
            
            # Axial capacity
            design['Ny'] = self.section.get_area() * fy / 1000  # kN
            
            # Shape factors
            try:
                sf = self.section.get_sf()
                design['shape_factor_x'] = sf[0]
                design['shape_factor_y'] = sf[1]
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
