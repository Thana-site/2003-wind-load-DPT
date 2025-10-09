"""
Section Factory Module - Fixed Version
Properly exports SectionFactory and handles materials correctly
"""

from sectionproperties.pre.library import (
    rectangular_section, i_section, channel_section,
    circular_section, circular_hollow_section, 
    tee_section, angle_section
)
from sectionproperties.pre import Material, Geometry
from sectionproperties.analysis import Section
from shapely.geometry import Polygon
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

@dataclass
class MaterialLibrary:
    """Predefined materials library"""
    
    @staticmethod
    def get_steel(grade: str = "S355") -> Material:
        """Get steel material properties"""
        grades = {
            "S235": {"fy": 235, "fu": 360},
            "S275": {"fy": 275, "fu": 430},
            "S355": {"fy": 355, "fu": 490},
            "S460": {"fy": 460, "fu": 540}
        }
        
        props = grades.get(grade, grades["S355"])
        
        return Material(
            name=f"Steel {grade}",
            elastic_modulus=200000,  # MPa
            poissons_ratio=0.3,
            yield_strength=props["fy"],
            density=7.85e-9,  # tonnes/mm³
            color="grey"
        )
    
    @staticmethod
    def get_aluminum(alloy: str = "6061") -> Material:
        """Get aluminum material properties"""
        alloys = {
            "6061": {"fy": 240, "fu": 260},
            "6063": {"fy": 170, "fu": 205},
            "7075": {"fy": 460, "fu": 540}
        }
        
        props = alloys.get(alloy, alloys["6061"])
        
        return Material(
            name=f"Aluminum {alloy}",
            elastic_modulus=70000,  # MPa
            poissons_ratio=0.33,
            yield_strength=props["fy"],
            density=2.7e-9,  # tonnes/mm³
            color="silver"
        )
    
    @staticmethod
    def get_concrete(fc: float = 30) -> Material:
        """Get concrete material properties"""
        # Elastic modulus from ACI 318
        Ec = 4700 * np.sqrt(fc)  # MPa
        
        return Material(
            name=f"Concrete fc'={fc}MPa",
            elastic_modulus=Ec,
            poissons_ratio=0.2,
            yield_strength=fc,
            density=2.4e-9,  # tonnes/mm³
            color="lightgrey"
        )
    
    @staticmethod
    def get_timber(species: str = "pine") -> Material:
        """Get timber material properties"""
        species_props = {
            "pine": {"E": 10000, "fb": 20},
            "oak": {"E": 12000, "fb": 30},
            "glulam": {"E": 13000, "fb": 35}
        }
        
        props = species_props.get(species, species_props["pine"])
        
        return Material(
            name=f"Timber ({species})",
            elastic_modulus=props["E"],  # MPa
            poissons_ratio=0.3,
            yield_strength=props["fb"],
            density=0.6e-9,  # tonnes/mm³
            color="brown"
        )


class EnhancedSectionFactory:
    """Enhanced factory with better material and composite support"""
    
    def __init__(self):
        """Initialize factory with material library"""
        self.material_library = MaterialLibrary()
        self.default_material = self.material_library.get_steel("S355")
        
    def create_section(self, section_type: str, params: Dict[str, Any], 
                      material: Optional[Material] = None,
                      analyze: bool = True) -> Section:
        """
        Create a section with proper material handling
        
        Args:
            section_type: Type of section
            params: Geometric parameters
            material: Optional material (uses default steel if None)
            analyze: Whether to perform analysis
            
        Returns:
            Analyzed Section object
        """
        # Use provided material or default
        if material is None:
            material = self.default_material
        
        # Validate parameters first
        is_valid, error_msg = self.validate_parameters(section_type, params)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")
        
        # Create geometry based on type
        geometry = self._create_geometry(section_type, params, material)
        
        # Create mesh with adaptive sizing
        mesh_size = params.get('mesh_size', self._get_adaptive_mesh_size(section_type, params))
        
        try:
            geometry.create_mesh(mesh_sizes=[mesh_size])
        except:
            # Fallback to coarser mesh if fine mesh fails
            geometry.create_mesh(mesh_sizes=[mesh_size * 2])
        
        # Create section
        section = Section(geometry=geometry)
        
        # Perform analysis if requested
        if analyze:
            section.calculate_geometric_properties()
            
            try:
                section.calculate_warping_properties()
            except:
                pass  # Some sections don't support warping
            
            try:
                section.calculate_plastic_properties()
            except:
                pass  # Some sections don't support plastic analysis
        
        return section
    
    def _get_adaptive_mesh_size(self, section_type: str, params: Dict[str, Any]) -> float:
        """Calculate adaptive mesh size based on geometry"""
        # Get characteristic dimension
        if section_type in ['Rectangle', 'Box']:
            char_dim = min(params.get('width', 100), params.get('depth', 100))
        elif section_type == 'Circle':
            char_dim = params.get('diameter', 100)
        elif section_type == 'Circular Hollow':
            char_dim = params.get('thickness', 10)
        else:
            char_dim = 50
        
        # Adaptive mesh size (5-10% of characteristic dimension)
        return max(2, min(20, char_dim * 0.075))
    
    def validate_parameters(self, section_type: str, params: Dict[str, Any]) -> tuple:
        """
        Validate section parameters
        
        Returns:
            (is_valid, error_message)
        """
        required_params = {
            'Rectangle': ['width', 'depth'],
            'I-Beam': ['depth', 'width', 'flange_thickness', 'web_thickness'],
            'Box': ['depth', 'width'],
            'Channel': ['depth', 'width', 'flange_thickness', 'web_thickness'],
            'Circle': ['diameter'],
            'Circular Hollow': ['outer_diameter', 'thickness'],
            'T-Section': ['depth', 'width', 'flange_thickness', 'web_thickness'],
            'Angle': ['leg1_length', 'leg2_length', 'thickness']
        }
        
        if section_type not in required_params:
            return False, f"Unknown section type: {section_type}"
        
        missing = [p for p in required_params[section_type] if p not in params]
        if missing:
            return False, f"Missing parameters: {', '.join(missing)}"
        
        # Validate positive values
        for param, value in params.items():
            if isinstance(value, (int, float)) and value <= 0:
                return False, f"Parameter {param} must be positive"
        
        return True, ""
    
    def _create_geometry(self, section_type: str, params: Dict[str, Any], 
                        material: Material) -> Geometry:
        """Create geometry based on section type"""
        creators = {
            'Rectangle': self._create_rectangle,
            'I-Beam': self._create_ibeam,
            'Box': self._create_box,
            'Channel': self._create_channel,
            'Circle': self._create_circular,
            'Circular Hollow': self._create_circular_hollow,
            'T-Section': self._create_tsection,
            'Angle': self._create_angle
        }
        
        creator = creators.get(section_type)
        if not creator:
            raise ValueError(f"Unknown section type: {section_type}")
        
        return creator(params, material)
    
    def _create_rectangle(self, params: Dict[str, Any], material: Material) -> Geometry:
        """Create rectangular section geometry"""
        return rectangular_section(
            d=float(params['depth']),
            b=float(params['width']),
            material=material
        )
    
    def _create_ibeam(self, params: Dict[str, Any], material: Material) -> Geometry:
        """Create I-beam section geometry"""
        return i_section(
            d=float(params['depth']),
            b=float(params['width']),
            t_f=float(params['flange_thickness']),
            t_w=float(params['web_thickness']),
            r=float(params.get('root_radius', 0)),
            n_r=int(params.get('n_r', 8)),
            material=material
        )
    
    def _create_box(self, params: Dict[str, Any], material: Material) -> Geometry:
        """Create box section geometry"""
        outer = rectangular_section(
            d=float(params['depth']),
            b=float(params['width']),
            material=material
        )
        
        if 'wall_thickness' in params:
            t = float(params['wall_thickness'])
            inner = rectangular_section(
                d=float(params['depth']) - 2*t,
                b=float(params['width']) - 2*t,
                material=material
            ).shift_section(x_offset=t, y_offset=t)
        else:
            t_w = float(params.get('web_thickness', 10))
            t_f = float(params.get('flange_thickness', 10))
            inner = rectangular_section(
                d=float(params['depth']) - 2*t_f,
                b=float(params['width']) - 2*t_w,
                material=material
            ).shift_section(x_offset=t_w, y_offset=t_f)
        
        return outer - inner
    
    def _create_channel(self, params: Dict[str, Any], material: Material) -> Geometry:
        """Create channel section geometry"""
        return channel_section(
            d=float(params['depth']),
            b=float(params['width']),
            t_f=float(params['flange_thickness']),
            t_w=float(params['web_thickness']),
            r=float(params.get('root_radius', 0)),
            n_r=int(params.get('n_r', 8)),
            material=material
        )
    
    def _create_circular(self, params: Dict[str, Any], material: Material) -> Geometry:
        """Create circular section geometry"""
        return circular_section(
            d=float(params['diameter']),
            n=int(params.get('n_circle', 64)),
            material=material
        )
    
    def _create_circular_hollow(self, params: Dict[str, Any], material: Material) -> Geometry:
        """Create circular hollow section geometry"""
        return circular_hollow_section(
            d=float(params['outer_diameter']),
            t=float(params['thickness']),
            n=int(params.get('n_circle', 64)),
            material=material
        )
    
    def _create_tsection(self, params: Dict[str, Any], material: Material) -> Geometry:
        """Create T-section geometry"""
        return tee_section(
            d=float(params['depth']),
            b=float(params['width']),
            t_f=float(params['flange_thickness']),
            t_w=float(params['web_thickness']),
            r=float(params.get('root_radius', 0)),
            n_r=int(params.get('n_r', 8)),
            material=material
        )
    
    def _create_angle(self, params: Dict[str, Any], material: Material) -> Geometry:
        """Create angle section geometry"""
        return angle_section(
            d=float(params['leg1_length']),
            b=float(params['leg2_length']),
            t=float(params['thickness']),
            r_r=float(params.get('root_radius', 0)),
            r_t=float(params.get('toe_radius', 0)),
            n_r=int(params.get('n_r', 8)),
            material=material
        )
    
    def create_composite_section(self, sections: List[Dict[str, Any]]) -> Section:
        """
        Create a composite section from multiple sections
        
        Args:
            sections: List of section definitions with type, params, material, and offset
            
        Returns:
            Analyzed composite Section object
        """
        if not sections:
            raise ValueError("No sections provided for composite")
        
        # Create first section
        first = sections[0]
        combined_geom = self._create_geometry(
            first['type'], 
            first['params'],
            first.get('material', self.default_material)
        )
        
        # Apply offset if specified
        if 'offset' in first and first['offset'] != (0, 0):
            combined_geom = combined_geom.shift_section(
                x_offset=first['offset'][0],
                y_offset=first['offset'][1]
            )
        
        # Add remaining sections
        for section_def in sections[1:]:
            geom = self._create_geometry(
                section_def['type'],
                section_def['params'],
                section_def.get('material', self.default_material)
            )
            
            # Apply offset
            if 'offset' in section_def and section_def['offset'] != (0, 0):
                geom = geom.shift_section(
                    x_offset=section_def['offset'][0],
                    y_offset=section_def['offset'][1]
                )
            
            # Combine geometries
            combined_geom = combined_geom + geom
        
        # Create mesh
        mesh_size = 10
        try:
            combined_geom.create_mesh(mesh_sizes=[mesh_size])
        except:
            combined_geom.create_mesh(mesh_sizes=[mesh_size * 2])
        
        # Create and analyze section
        section = Section(geometry=combined_geom)
        section.calculate_geometric_properties()
        
        try:
            section.calculate_warping_properties()
        except:
            pass
        
        try:
            section.calculate_plastic_properties()
        except:
            pass
        
        return section


# Create an alias for backward compatibility
SectionFactory = EnhancedSectionFactory
