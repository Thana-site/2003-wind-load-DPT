"""
Calculations Module - Enhanced Version
Handles section property calculations with automatic composite detection
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
    """Enhanced analyzer with composite detection and multiple section support"""
    
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
            if hasattr(self.section, 'geometry') and hasattr(self.section.geometry, 'material'):
                if self.section.geometry.material is not None:
                    return True
            
            # Try to access composite properties as a test
            try:
                _ = self.section.get_eic()
                return True
            except (AttributeError, TypeError, ValueError):
                pass
                
            # Check for materials in the mesh
            if hasattr(self.section, 'materials') and self.section.materials:
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
            # Basic geometric properties
            properties['area'] = self.section.get_area()
            properties['perimeter'] = self.section.get_perimeter()
            
            # Centroid
            cx, cy = self.section.get_c()
            properties['cx'] = cx
            properties['cy'] = cy
            
            # Determine analysis type and get appropriate properties
            if self.is_composite:
                messages.append("✓ Composite materials detected — using equivalent properties (E*I)")
                properties.update(self._get_composite_properties())
            else:
                messages.append("✓ Geometric analysis mode — material-independent properties")
                properties.update(self._get_geometric_properties())
            
            # Common properties
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
            # Composite second moments of area (E*I)
            eic = self.section.get_eic()
            props['eixx_c'] = eic[0]
            props['eiyy_c'] = eic[1]
            props['eixy_c'] = eic[2]
            
            # For display, also get geometric I (without E)
            try:
                ic = self.section.get_ic()
                props['ixx_c'] = ic[0]
                props['iyy_c'] = ic[1]
                props['ixy_c'] = ic[2]
            except:
                # Estimate geometric I by dividing by E if available
                if hasattr(self.section.geometry, 'material'):
                    E = getattr(self.section.geometry.material, 'elastic_modulus', 1)
                    props['ixx_c'] = props['eixx_c'] / E if E != 0 else props['eixx_c']
                    props['iyy_c'] = props['eiyy_c'] / E if E != 0 else props['eiyy_c']
                    props['ixy_c'] = props['eixy_c'] / E if E != 0 else props['eixy_c']
                else:
                    props['ixx_c'] = props['eixx_c']
                    props['iyy_c'] = props['eiyy_c']
                    props['ixy_c'] = props['eixy_c']
            
            # Composite properties about origin
            try:
                eig = self.section.get_eig()
                props['eixx_g'] = eig[0]
                props['eiyy_g'] = eig[1]
                props['eixy_g'] = eig[2]
            except:
                props['eixx_g'] = 0
                props['eiyy_g'] = 0
                props['eixy_g'] = 0
                
            # Principal moments
            try:
                props['ei11_c'] = self.section.get_ei11_c()
                props['ei22_c'] = self.section.get_ei22_c()
                props['i11_c'] = props['ei11_c'] / getattr(self.section.geometry.material, 'elastic_modulus', 1)
                props['i22_c'] = props['ei22_c'] / getattr(self.section.geometry.material, 'elastic_modulus', 1)
            except:
                props['i11_c'] = props.get('ixx_c', 0)
                props['i22_c'] = props.get('iyy_c', 0)
                
        except Exception as e:
            self.messages.append(f"Composite calculation note: {e}")
            
        return props
    
    def _get_geometric_properties(self) -> Dict[str, Any]:
        """Get properties for geometric-only sections"""
        props = {}
        
        try:
            # Geometric second moments of area
            ic = self.section.get_ic()
            props['ixx_c'] = ic[0]
            props['iyy_c'] = ic[1]
            props['ixy_c'] = ic[2]
            
            # About origin
            try:
                ig = self.section.get_ig()
                props['ixx_g'] = ig[0]
                props['iyy_g'] = ig[1]
                props['ixy_g'] = ig[2]
            except:
                props['ixx_g'] = 0
                props['iyy_g'] = 0
                props['ixy_g'] = 0
            
            # Principal moments
            try:
                props['i11_c'] = self.section.get_i11_c()
                props['i22_c'] = self.section.get_i22_c()
            except:
                props['i11_c'] = props['ixx_c']
                props['i22_c'] = props['iyy_c']
                
        except Exception as e:
            self.messages.append(f"Geometric calculation note: {e}")
            
        return props
    
    def _get_common_properties(self) -> Dict[str, Any]:
        """Get properties common to both composite and geometric sections"""
        props = {}
        
        # Radii of gyration
        try:
            rc = self.section.get_rc()
            props['rx'] = rc[0]
            props['ry'] = rc[1]
        except:
            area = self.section.get_area()
            ixx = props.get('ixx_c', 0)
            iyy = props.get('iyy_c', 0)
            props['rx'] = np.sqrt(ixx / area) if area > 0 else 0
            props['ry'] = np.sqrt(iyy / area) if area > 0 else 0
        
        # Section moduli
        try:
            z = self.section.get_z()
            if len(z) >= 4:
                props['zxx_plus'] = abs(z[0]) if z[0] != 0 else 0
                props['zxx_minus'] = abs(z[1]) if z[1] != 0 else 0
                props['zyy_plus'] = abs(z[2]) if z[2] != 0 else 0
                props['zyy_minus'] = abs(z[3]) if z[3] != 0 else 0
        except:
            # Manual calculation fallback
            props.update(self._calculate_section_moduli_manually())
        
        # Principal angle
        try:
            props['phi'] = self.section.get_phi()
        except:
            props['phi'] = 0
        
        # Torsion constant
        try:
            props['j'] = self.section.get_j()
        except:
            props['j'] = 0
        
        # Warping constant
        try:
            props['gamma'] = self.section.get_gamma()
        except:
            props['gamma'] = 0
        
        # Plastic moduli
        try:
            s = self.section.get_s()
            props['sxx'] = abs(s[0]) if len(s) > 0 else 0
            props['syy'] = abs(s[1]) if len(s) > 1 else 0
        except:
            props['sxx'] = 0
            props['syy'] = 0
        
        return props
    
    def _calculate_section_moduli_manually(self) -> Dict[str, Any]:
        """Manually calculate section moduli from mesh coordinates"""
        props = {}
        
        try:
            coords = self.section.mesh_nodes
            if len(coords) > 0:
                cx, cy = self.section.get_c()
                y_coords = coords[:, 1]
                x_coords = coords[:, 0]
                
                y_top = max(y_coords) - cy
                y_bot = cy - min(y_coords)
                x_right = max(x_coords) - cx
                x_left = cx - min(x_coords)
                
                ixx = self.properties.get('ixx_c', 0)
                iyy = self.properties.get('iyy_c', 0)
                
                props['zxx_plus'] = ixx / y_top if y_top > 0 else 0
                props['zxx_minus'] = ixx / y_bot if y_bot > 0 else 0
                props['zyy_plus'] = iyy / x_right if x_right > 0 else 0
                props['zyy_minus'] = iyy / x_left if x_left > 0 else 0
        except:
            props['zxx_plus'] = 0
            props['zxx_minus'] = 0
            props['zyy_plus'] = 0
            props['zyy_minus'] = 0
            
        return props
    
    def _get_material_info(self) -> Optional[Dict[str, Any]]:
        """Extract material information if available"""
        try:
            if hasattr(self.section.geometry, 'material'):
                mat = self.section.geometry.material
                return {
                    'name': getattr(mat, 'name', 'Unknown'),
                    'elastic_modulus': getattr(mat, 'elastic_modulus', None),
                    'poissons_ratio': getattr(mat, 'poissons_ratio', None),
                    'yield_strength': getattr(mat, 'yield_strength', None),
                    'density': getattr(mat, 'density', None)
                }
        except:
            pass
        return None
    
    def _validate_properties(self, props: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean property values"""
        for key in props:
            if props[key] is None or np.isnan(props[key]) or np.isinf(props[key]):
                props[key] = 0.0
        return props
    
    def _get_fallback_properties(self) -> Dict[str, Any]:
        """Return minimal properties in case of failure"""
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
        self.geometries = []
        self.combined_section = None
        
    def add_section(self, geometry, material=None, name=None, offset=(0, 0)):
        """
        Add a section to the composite
        
        Args:
            geometry: Section geometry from sectionproperties
            material: Optional material properties
            name: Optional name for the section
            offset: (x, y) offset for positioning
        """
        # Apply offset if specified
        if offset != (0, 0):
            geometry = geometry.shift_section(x_offset=offset[0], y_offset=offset[1])
        
        # Store section info
        self.sections.append({
            'geometry': geometry,
            'material': material,
            'name': name or f"Section_{len(self.sections)+1}",
            'offset': offset
        })
        
        self.geometries.append(geometry)
        
    def create_composite(self) -> Tuple[Any, List[str]]:
        """
        Create a composite section from all added sections
        
        Returns:
            Tuple of (Section object, list of messages)
        """
        messages = []
        
        if len(self.geometries) == 0:
            raise ValueError("No sections added to composite")
        
        if len(self.geometries) == 1:
            messages.append("ℹ️ Only one section provided - analyzing as single section")
            combined_geom = self.geometries[0]
        else:
            messages.append(f"✓ Combining {len(self.geometries)} sections into composite")
            
            # Combine geometries using addition
            combined_geom = self.geometries[0]
            for geom in self.geometries[1:]:
                combined_geom = combined_geom + geom
            
            # Check if we have mixed materials
            materials = [s['material'] for s in self.sections if s['material'] is not None]
            if len(set(materials)) > 1:
                messages.append("⚠️ Multiple materials detected - composite analysis will use weighted properties")
        
        # Create mesh
        try:
            combined_geom.create_mesh(mesh_sizes=[10])
        except:
            combined_geom.create_mesh(mesh_sizes=[5])
            
        # Create and analyze section
        from sectionproperties.analysis import Section
        combined_section = Section(geometry=combined_geom)
        combined_section.calculate_geometric_properties()
        
        try:
            combined_section.calculate_warping_properties()
        except:
            messages.append("Note: Warping properties not available for this composite")
            
        try:
            combined_section.calculate_plastic_properties()
        except:
            messages.append("Note: Plastic properties not available for this composite")
        
        self.combined_section = combined_section
        messages.append(f"✓ Composite section created successfully")
        
        return combined_section, messages
    
    def get_summary(self) -> pd.DataFrame:
        """Get a summary of all sections in the composite"""
        data = []
        for section in self.sections:
            # Create temporary section for analysis
            from sectionproperties.analysis import Section
            temp_geom = section['geometry'].copy()
            temp_geom.create_mesh(mesh_sizes=[10])
            temp_section = Section(geometry=temp_geom)
            temp_section.calculate_geometric_properties()
            
            area = temp_section.get_area()
            cx, cy = temp_section.get_c()
            
            data.append({
                'Name': section['name'],
                'Material': section['material'].name if section['material'] else 'None',
                'Area': area,
                'Centroid X': cx + section['offset'][0],
                'Centroid Y': cy + section['offset'][1],
                'Offset X': section['offset'][0],
                'Offset Y': section['offset'][1]
            })
        
        return pd.DataFrame(data)
