"""
Calculations Module
Handles section property calculations and analysis
"""

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
        
        # Basic geometric properties
        properties['area'] = self.section.get_area()
        properties['perimeter'] = self.section.get_perimeter()
        
        # Centroid
        cx, cy = self.section.get_c()
        properties['cx'] = cx
        properties['cy'] = cy
        
        # Second moments of area (about centroid)
        properties['ixx_c'] = self.section.get_ic()[0]
        properties['iyy_c'] = self.section.get_ic()[1]
        properties['ixy_c'] = self.section.get_ic()[2]
        
        # Second moments of area (about origin)
        properties['ixx_g'] = self.section.get_ig()[0]
        properties['iyy_g'] = self.section.get_ig()[1]
        properties['ixy_g'] = self.section.get_ig()[2]
        
        # Radii of gyration
        properties['rx'] = self.section.get_rc()[0]
        properties['ry'] = self.section.get_rc()[1]
        
        # Principal moments of inertia
        phi, i11, i22 = self.section.get_phi(), self.section.get_i11_c(), self.section.get_i22_c()
        properties['phi'] = phi  # Principal axis angle
        properties['i11_c'] = i11  # Major principal moment
        properties['i22_c'] = i22  # Minor principal moment
        
        # Elastic section moduli (about centroidal axes)
        properties['zxx_plus'] = self.section.get_z()[0]   # Top fiber
        properties['zxx_minus'] = self.section.get_z()[1]  # Bottom fiber
        properties['zyy_plus'] = self.section.get_z()[2]   # Right fiber
        properties['zyy_minus'] = self.section.get_z()[3]  # Left fiber
        
        # Principal section moduli
        properties['z11_plus'] = self.section.get_z()[4]
        properties['z11_minus'] = self.section.get_z()[5]
        properties['z22_plus'] = self.section.get_z()[6]
        properties['z22_minus'] = self.section.get_z()[7]
        
        # Torsion properties
        properties['j'] = self.section.get_j()  # Torsion constant
        
        # Warping properties
        properties['gamma'] = self.section.get_gamma()  # Warping constant
        
        # Plastic section moduli
        properties['sxx'] = self.section.get_s()[0]
        properties['syy'] = self.section.get_s()[1]
        properties['s11'] = self.section.get_s()[2]
        properties['s22'] = self.section.get_s()[3]
        
        # Shape factors
        properties['sf_xx'] = self.section.get_sf()[0]
        properties['sf_yy'] = self.section.get_sf()[1]
        properties['sf_11'] = self.section.get_sf()[2]
        properties['sf_22'] = self.section.get_sf()[3]
        
        # Additional calculated properties
        properties.update(self._calculate_additional_properties(properties))
        
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
        
        # Polar moment of inertia
        additional['ip'] = props['ixx_c'] + props['iyy_c']
        
        # Polar radius of gyration
        additional['rp'] = np.sqrt(additional['ip'] / props['area'])
        
        # Aspect ratio (for rectangular-like sections)
        # Approximate using principal moments
        if props['i22_c'] > 0:
            additional['aspect_ratio'] = np.sqrt(props['i11_c'] / props['i22_c'])
        
        # Compactness (perimeter squared / area)
        additional['compactness'] = props['perimeter']**2 / (4 * np.pi * props['area'])
        
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
        
        # Axial stress
        if N != 0:
            stress_results['sigma_n'] = N / self.section.get_area()
        else:
            stress_results['sigma_n'] = 0
        
        # Bending stresses (at extreme fibers)
        if Mx != 0:
            z = self.section.get_z()
            stress_results['sigma_mx_top'] = Mx / z[0]
            stress_results['sigma_mx_bot'] = -Mx / z[1]
        else:
            stress_results['sigma_mx_top'] = 0
            stress_results['sigma_mx_bot'] = 0
            
        if My != 0:
            z = self.section.get_z()
            stress_results['sigma_my_right'] = My / z[2]
            stress_results['sigma_my_left'] = -My / z[3]
        else:
            stress_results['sigma_my_right'] = 0
            stress_results['sigma_my_left'] = 0
        
        # Shear stress from torsion
        if Mz != 0 and self.section.get_j() > 0:
            # Approximate max shear stress
            # For thin-walled sections: tau = T*t/J
            # This is simplified - actual implementation would need section-specific formulas
            stress_results['tau_max'] = Mz / self.section.get_j()
        else:
            stress_results['tau_max'] = 0
        
        # Combined stresses (von Mises)
        stress_results['sigma_vm_max'] = self._calculate_von_mises(
            stress_results['sigma_n'] + max(
                stress_results['sigma_mx_top'],
                stress_results['sigma_mx_bot']
            ) + max(
                stress_results['sigma_my_right'],
                stress_results['sigma_my_left']
            ),
            stress_results['tau_max']
        )
        
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
        
        # Plastic moment capacities
        design['Mp_x'] = self.section.get_s()[0] * fy / 1e6  # kN.m
        design['Mp_y'] = self.section.get_s()[1] * fy / 1e6  # kN.m
        
        # Yield moment (elastic)
        z = self.section.get_z()
        design['My_x'] = min(z[0], z[1]) * fy / 1e6  # kN.m
        design['My_y'] = min(z[2], z[3]) * fy / 1e6  # kN.m
        
        # Axial capacity
        design['Ny'] = self.section.get_area() * fy / 1000  # kN
        
        # Shape factors
        design['shape_factor_x'] = self.section.get_sf()[0]
        design['shape_factor_y'] = self.section.get_sf()[1]
        
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
        import pandas as pd
        
        comparison_data = []
        
        for idx, section in enumerate(sections):
            analyzer = SectionAnalyzer(section)
            props = analyzer.calculate_properties()
            
            comparison_data.append({
                'Section': f"Section {idx+1}",
                'Area': props['area'],
                'Ixx': props['ixx_c'],
                'Iyy': props['iyy_c'],
                'Zxx': min(props['zxx_plus'], props['zxx_minus']),
                'Zyy': min(props['zyy_plus'], props['zyy_minus']),
                'rx': props['rx'],
                'ry': props['ry'],
                'J': props['j']
            })
        
        return pd.DataFrame(comparison_data)
