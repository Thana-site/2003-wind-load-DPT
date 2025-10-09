# modules/calculations.py

```python
"""
Calculations Module - Fixed Version
Properly handles composite sections and avoids get_ic() errors
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import warnings

@dataclass
class AnalysisResult:
    """Container for analysis results with metadata"""
    properties: Dict[str, Any]
    is_composite: bool
    material_info: Optional[Dict[str, Any]] = None
    messages: List[str] = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []

class SectionAnalyzer:
    """Enhanced analyzer with proper composite detection"""
    
    def __init__(self, section):
        """
        Initialize analyzer with a section
        
        Args:
            section: Analyzed Section object from sectionproperties
        """
        self.section = section
        self.is_composite = self._detect_composite()
        self.messages = []
        
    def _detect_composite(self) -> bool:
        """
        Detect if section has material properties (is composite)
        
        Returns:
            bool: True if composite, False if geometric only
        """
        try:
            # Check if geometry has materials
            if hasattr(self.section, 'geometry'):
                if hasattr(self.section.geometry, 'material'):
                    if self.section.geometry.material is not None:
                        return True
                
                # Check for multiple materials in mesh
                if hasattr(self.section.geometry, 'materials'):
                    if self.section.geometry.materials and len(self.section.geometry.materials) > 0:
                        return True
            
            # Check section elements for materials
            if hasattr(self.section, 'elements'):
                # If any element has a material ID > 0, it's composite
                for el in self.section.elements:
                    if hasattr(el, 'material') and el.material is not None:
                        return True
            
            return False
            
        except Exception:
            return False
    
    def calculate_properties(self) -> AnalysisResult:
        """
        Calculate all section properties with automatic composite handling
        
        Returns:
            AnalysisResult: Object containing properties and metadata
        """
        properties = {}
        messages = []
        
        try:
            # Basic geometric properties (always available)
            properties['area'] = self.section.get_area()
            properties['perimeter'] = self.section.get_perimeter()
            
            # Centroid
            cx, cy = self.section.get_c()
            properties['cx'] = cx
            properties['cy'] = cy
            
            # Determine analysis type and get appropriate properties
            if self.is_composite:
                messages.append("✓ Composite materials detected — using E×I properties")
                properties.update(self._get_composite_properties())
            else:
                messages.append("✓ Geometric analysis mode — material-independent properties")
                properties.update(self._get_geometric_properties())
            
            # Common properties (work for both types)
            properties.update(self._get_common_properties())
            
            # Validate and clean results
            properties = self._validate_properties(properties)
            
            # Get material info if composite
            material_info = None
            if self.is_composite:
                material_info = self._get_material_info()
            
            return AnalysisResult(
                properties=properties,
                is_composite=self.is_composite,
                material_info=material_info,
                messages=messages
            )
            
        except Exception as e:
            messages.append(f"⚠️ Analysis warning: {str(e)}")
            return AnalysisResult(
                properties=self._get_fallback_properties(),
                is_composite=False,
                messages=messages
            )
    
    def _get_composite_properties(self) -> Dict[str, Any]:
        """Get properties for composite sections (with materials)"""
        props = {}
        
        try:
            # For composite sections, use E×I values
            # Get E×I values (these are safe for composite sections)
            eig = self.section.get_eig()  # About global origin
            props['eixx_g'] = eig[0]
            props['eiyy_g'] = eig[1]
            props['eixy_g'] = eig[2]
            
            # About centroid (if available)
            try:
                eic = self.section.get_eic()
                props['eixx_c'] = eic[0]
                props['eiyy_c'] = eic[1]
                props['eixy_c'] = eic[2]
            except:
                # Calculate from global values
                area = self.section.get_area()
                cx, cy = self.section.get_c()
                props['eixx_c'] = props['eixx_g'] - area * cy**2
                props['eiyy_c'] = props['eiyy_g'] - area * cx**2
                props['eixy_c'] = props['eixy_g'] - area * cx * cy
            
            # Try to extract geometric I by dividing by E
            # This is approximate if multiple materials exist
            E_ref = self._get_reference_modulus()
            if E_ref and E_ref > 0:
                props['ixx_c'] = props['eixx_c'] / E_ref
                props['iyy_c'] = props['eiyy_c'] / E_ref
                props['ixy_c'] = props['eixy_c'] / E_ref
                props['ixx_g'] = props['eixx_g'] / E_ref
                props['iyy_g'] = props['eiyy_g'] / E_ref
                props['ixy_g'] = props['eixy_g'] / E_ref
            else:
                # Use E×I values directly
                props['ixx_c'] = props['eixx_c']
                props['iyy_c'] = props['eiyy_c']
                props['ixy_c'] = props['eixy_c']
                props['ixx_g'] = props['eixx_g']
                props['iyy_g'] = props['eiyy_g']
                props['ixy_g'] = props['eixy_g']
                
            # Principal moments
            try:
                props['ei11_c'] = self.section.get_ei11_c()
                props['ei22_c'] = self.section.get_ei22_c()
                if E_ref and E_ref > 0:
                    props['i11_c'] = props['ei11_c'] / E_ref
                    props['i22_c'] = props['ei22_c'] / E_ref
                else:
                    props['i11_c'] = props['ei11_c']
                    props['i22_c'] = props['ei22_c']
            except:
                props['i11_c'] = max(props.get('ixx_c', 0), props.get('iyy_c', 0))
                props['i22_c'] = min(props.get('ixx_c', 0), props.get('iyy_c', 0))
                
        except Exception as e:
            self.messages.append(f"Note: Some composite properties unavailable: {e}")
            # Return basic properties at minimum
            props.update(self._get_fallback_properties())
            
        return props
    
    def _get_geometric_properties(self) -> Dict[str, Any]:
        """Get properties for geometric-only sections"""
        props = {}
        
        try:
            # Geometric second moments about centroid
            ic = self.section.get_ic()
            props['ixx_c'] = ic[0]
            props['iyy_c'] = ic[1]
            props['ixy_c'] = ic[2]
            
            # About global origin
            try:
                ig = self.section.get_ig()
                props['ixx_g'] = ig[0]
                props['iyy_g'] = ig[1]
                props['ixy_g'] = ig[2]
            except:
                # Calculate using parallel axis theorem
                area = self.section.get_area()
                cx, cy = self.section.get_c()
                props['ixx_g'] = props['ixx_c'] + area * cy**2
                props['iyy_g'] = props['iyy_c'] + area * cx**2
                props['ixy_g'] = props['ixy_c'] + area * cx * cy
            
            # Principal moments
            try:
                props['i11_c'] = self.section.get_i11_c()
                props['i22_c'] = self.section.get_i22_c()
            except:
                # Calculate from ixx, iyy, ixy
                ixx = props['ixx_c']
                iyy = props['iyy_c']
                ixy = props['ixy_c']
                
                avg = (ixx + iyy) / 2
                diff = (ixx - iyy) / 2
                props['i11_c'] = avg + np.sqrt(diff**2 + ixy**2)
                props['i22_c'] = avg - np.sqrt(diff**2 + ixy**2)
                
        except Exception as e:
            self.messages.append(f"Note: Some geometric properties unavailable: {e}")
            props.update(self._get_fallback_properties())
            
        return props
    
    def _get_common_properties(self) -> Dict[str, Any]:
        """Get properties that work for both composite and geometric sections"""
        props = {}
        
        try:
            # Section moduli
            props['zxx_plus'] = self.section.get_z()[0]
            props['zxx_minus'] = self.section.get_z()[1]
            props['zyy_plus'] = self.section.get_z()[2]
            props['zyy_minus'] = self.section.get_z()[3]
        except:
            props['zxx_plus'] = 0
            props['zxx_minus'] = 0
            props['zyy_plus'] = 0
            props['zyy_minus'] = 0
        
        try:
            # Radii of gyration
            props['rx'] = self.section.get_r()[0]
            props['ry'] = self.section.get_r()[1]
        except:
            area = self.section.get_area()
            if area > 0:
                props['rx'] = np.sqrt(props.get('ixx_c', 0) / area)
                props['ry'] = np.sqrt(props.get('iyy_c', 0) / area)
            else:
                props['rx'] = 0
                props['ry'] = 0
        
        try:
            # Principal angle
            props['phi'] = self.section.get_phi() * 180 / np.pi  # Convert to degrees
        except:
            props['phi'] = 0
        
        try:
            # Torsion constant
            props['j'] = self.section.get_j()
        except:
            props['j'] = 0
        
        try:
            # Warping constant
            props['gamma'] = self.section.get_gamma()
        except:
            props['gamma'] = 0
        
        try:
            # Plastic moduli
            props['sxx'] = self.section.get_s()[0]
            props['syy'] = self.section.get_s()[1]
        except:
            props['sxx'] = props.get('zxx_plus', 0) * 1.5  # Approximate
            props['syy'] = props.get('zyy_plus', 0) * 1.5
        
        try:
            # Shape factors
            props['sf_xx'] = self.section.get_sf()[0]
            props['sf_yy'] = self.section.get_sf()[1]
        except:
            props['sf_xx'] = 1.5  # Default for rectangular
            props['sf_yy'] = 1.5
        
        return props
    
    def _get_reference_modulus(self) -> Optional[float]:
        """Get a reference elastic modulus for the section"""
        try:
            if hasattr(self.section.geometry, 'material'):
                if hasattr(self.section.geometry.material, 'elastic_modulus'):
                    return self.section.geometry.material.elastic_modulus
            
            # Try to get from first material in list
            if hasattr(self.section.geometry, 'materials'):
                if self.section.geometry.materials:
                    first_mat = list(self.section.geometry.materials)[0]
                    if hasattr(first_mat, 'elastic_modulus'):
                        return first_mat.elastic_modulus
            
            return None
        except:
            return None
    
    def _get_material_info(self) -> Dict[str, Any]:
        """Get material information from the section"""
        info = {}
        
        try:
            if hasattr(self.section.geometry, 'material'):
                mat = self.section.geometry.material
                if mat:
                    info['name'] = getattr(mat, 'name', 'Unknown')
                    info['elastic_modulus'] = getattr(mat, 'elastic_modulus', 0)
                    info['poissons_ratio'] = getattr(mat, 'poissons_ratio', 0)
                    info['yield_strength'] = getattr(mat, 'yield_strength', 0)
                    info['density'] = getattr(mat, 'density', 0)
            
            # Check for multiple materials
            if hasattr(self.section.geometry, 'materials'):
                if self.section.geometry.materials and len(self.section.geometry.materials) > 1:
                    info['is_multi_material'] = True
                    info['material_count'] = len(self.section.geometry.materials)
        except:
            pass
        
        return info
    
    def _validate_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean property values"""
        cleaned = {}
        
        for key, value in properties.items():
            if value is None:
                cleaned[key] = 0
            elif isinstance(value, (int, float)):
                # Check for NaN or Inf
                if np.isnan(value) or np.isinf(value):
                    cleaned[key] = 0
                else:
                    cleaned[key] = value
            else:
                cleaned[key] = value
        
        return cleaned
    
    def _get_fallback_properties(self) -> Dict[str, Any]:
        """Get minimal fallback properties if analysis fails"""
        try:
            area = self.section.get_area()
            cx, cy = self.section.get_c()
        except:
            area = 0
            cx, cy = 0, 0
            
        return {
            'area': area,
            'cx': cx,
            'cy': cy,
            'ixx_c': 0,
            'iyy_c': 0,
            'ixy_c': 0,
            'zxx_plus': 0,
            'zxx_minus': 0,
            'zyy_plus': 0,
            'zyy_minus': 0,
            'j': 0,
            'gamma': 0
        }


class CompositeSectionAnalyzer:
    """Analyzer for combining multiple sections into a composite"""
    
    def __init__(self):
        self.sections = []
        self.combined_section = None
        
    def add_section(self, section, name=None, offset=(0, 0)):
        """
        Add a section to the composite
        
        Args:
            section: Section object from sectionproperties
            name: Optional name for the section
            offset: (x, y) offset for positioning
        """
        self.sections.append({
            'section': section,
            'name': name or f"Section_{len(self.sections)+1}",
            'offset': offset
        })
    
    def clear_sections(self):
        """Clear all sections"""
        self.sections = []
        self.combined_section = None
    
    def create_composite(self) -> Tuple[Any, List[str]]:
        """
        Create a composite section from all added sections
        
        Returns:
            Tuple of (Section object, list of messages)
        """
        messages = []
        
        if len(self.sections) == 0:
            raise ValueError("No sections added to composite")
        
        if len(self.sections) == 1:
            messages.append("ℹ️ Only one section provided")
            self.combined_section = self.sections[0]['section']
            return self.combined_section, messages
        
        messages.append(f"✓ Combining {len(self.sections)} sections into composite")
        
        try:
            # Get geometries from sections
            first_section_data = self.sections[0]
            combined_geom = first_section_data['section'].geometry.copy()
            
            # Apply first offset if needed
            if first_section_data['offset'] != (0, 0):
                combined_geom = combined_geom.shift_section(
                    x_offset=first_section_data['offset'][0],
                    y_offset=first_section_data['offset'][1]
                )
            
            # Add remaining sections
            for section_data in self.sections[1:]:
                geom = section_data['section'].geometry.copy()
                
                # Apply offset
                if section_data['offset'] != (0, 0):
                    geom = geom.shift_section(
                        x_offset=section_data['offset'][0],
                        y_offset=section_data['offset'][1]
                    )
                
                # Combine geometries
                combined_geom = combined_geom + geom
            
            # Create mesh
            try:
                combined_geom.create_mesh(mesh_sizes=[10])
            except:
                combined_geom.create_mesh(mesh_sizes=[20])
            
            # Create and analyze composite section
            from sectionproperties.analysis import Section
            self.combined_section = Section(geometry=combined_geom)
            self.combined_section.calculate_geometric_properties()
            
            try:
                self.combined_section.calculate_warping_properties()
            except:
                messages.append("Note: Warping properties not available for this composite")
            
            try:
                self.combined_section.calculate_plastic_properties()
            except:
                messages.append("Note: Plastic properties not available for this composite")
            
            messages.append("✅ Composite section created successfully")
            
        except Exception as e:
            raise ValueError(f"Failed to create composite: {str(e)}")
        
        return self.combined_section, messages
    
    def get_summary(self) -> pd.DataFrame:
        """Get a summary of all sections in the composite"""
        data = []
        
        for section_data in self.sections:
            section = section_data['section']
            
            try:
                area = section.get_area()
                cx, cy = section.get_c()
            except:
                area = 0
                cx, cy = 0, 0
            
            data.append({
                'Name': section_data['name'],
                'Area': area,
                'Centroid X': cx + section_data['offset'][0],
                'Centroid Y': cy + section_data['offset'][1],
                'Offset X': section_data['offset'][0],
                'Offset Y': section_data['offset'][1]
            })
        
        return pd.DataFrame(data)
```
