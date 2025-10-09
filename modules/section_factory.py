"""
Section Factory Module - Enhanced Version
Improved material handling and composite section support
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
            ultimate_strength=props["fu"],
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
            ultimate_strength=props["fu"],
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
                      material: Optional[Material] = None) -> Section:
        """
        Create a section with proper material handling
        
        Args:
            section_type: Type of section
            params: Geometric parameters
            material: Optional material (uses default steel if None)
            
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
        mesh_size = params.get('mesh_size', self._get_adaptive_mesh_size(geometry))
        
        try:
            geometry.create_mesh(mesh_sizes=[mesh_size])
        except:
            # Try with coarser mesh if fine mesh fails
            try:
                geometry.create_mesh(mesh_sizes=[mesh_size * 2])
            except:
                # Last resort - very coarse mesh
                geometry.create_mesh(mesh_sizes=[mesh_size * 5])
        
        # Create section
        section = Section(geometry=geometry)
        
        # Perform analysis
        section.calculate_geometric_properties()
        
        # Try advanced analyses with error handling
        try:
            section.calculate_warping_properties()
        except:
            pass  # Warping not available for all sections
        
        try:
            section.calculate_plastic_properties()
        except:
            pass  # Plastic properties not available for all sections
        
        return section
    
    def _create_geometry(self, section_type: str, params: Dict[str, Any], 
                        material: Material) -> Geometry:
        """Create geometry for the specified section type"""
        
        creators = {
            "I-Beam": self._create_ibeam,
            "Box Section": self._create_box,
            "Channel": self._create_channel,
            "Circular": self._create_circular,
            "Circular Hollow": self._create_circular_hollow,
            "T-Section": self._create_tsection,
            "Angle": self._create_angle,
            "Plate": self._create_plate,
            "Custom Polygon": self._create_polygon
        }
        
        creator = creators.get(section_type)
        if not creator:
            raise ValueError(f"Unknown section type: {section_type}")
        
        return creator(params, material)
    
    def _create_ibeam(self, params: Dict[str, Any], material: Material) -> Geometry:
        """Create I-beam geometry"""
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
    
    def _create_plate(self, params: Dict[str, Any], material: Material) -> Geometry:
        """Create a simple rectangular plate"""
        return rectangular_section(
            d=float(params.get('depth', params.get('height', 10))),
            b=float(params['width']),
            material=material
        )
    
    def _create_polygon(self, params: Dict[str, Any], material: Material) -> Geometry:
        """Create custom polygon section geometry"""
        nodes = params.get('nodes', [])
        
        if len(nodes) < 3:
            raise ValueError("Polygon must have at least 3 nodes")
        
        # Create polygon
        polygon = Polygon(nodes)
        
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        
        # Handle holes
        if 'holes' in params and params['holes']:
            for hole in params['holes']:
                if len(hole) >= 3:
                    hole_polygon = Polygon(hole)
                    if hole_polygon.is_valid:
                        polygon = polygon.difference(hole_polygon)
        
        return Geometry(geom=polygon, material=material)
    
    def _get_adaptive_mesh_size(self, geometry: Geometry) -> float:
        """Calculate adaptive mesh size based on geometry size"""
        try:
            # Get bounds of geometry
            bounds = geometry.geom.bounds  # (minx, miny, maxx, maxy)
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            
            # Use ~50 elements along the smallest dimension
            min_dim = min(width, height)
            mesh_size = min_dim / 50
            
            # Clamp between reasonable limits
            return max(0.5, min(mesh_size, 50))
        except:
            return 10  # Default fallback
    
    def create_composite_section(self, sections: List[Dict[str, Any]]) -> Section:
        """
        Create a composite section from multiple sections
        
        Args:
            sections: List of dictionaries containing:
                - 'type': Section type
                - 'params': Parameters
                - 'material': Optional Material object
                - 'offset': (x, y) tuple for positioning
                
        Returns:
            Combined Section object
        """
        if not sections:
            raise ValueError("No sections provided for composite")
        
        geometries = []
        
        for section_data in sections:
            # Create individual geometry
            material = section_data.get('material', self.default_material)
            geom = self._create_geometry(
                section_data['type'],
                section_data['params'],
                material
            )
            
            # Apply offset
            offset = section_data.get('offset', (0, 0))
            if offset != (0, 0):
                geom = geom.shift_section(x_offset=offset[0], y_offset=offset[1])
            
            geometries.append(geom)
        
        # Combine all geometries
        combined_geometry = geometries[0]
        for geom in geometries[1:]:
            combined_geometry = combined_geometry + geom
        
        # Create mesh
        mesh_size = self._get_adaptive_mesh_size(combined_geometry)
        combined_geometry.create_mesh(mesh_sizes=[mesh_size])
        
        # Create and analyze section
        section = Section(geometry=combined_geometry)
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
    
    @staticmethod
    def validate_parameters(section_type: str, params: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate section parameters
        
        Returns:
            (is_valid, error_message)
        """
        # Basic validation rules
        if section_type in ["I-Beam", "Channel", "T-Section"]:
            required = ['depth', 'width', 'flange_thickness', 'web_thickness']
            for key in required:
                if key not in params:
                    return False, f"Missing required parameter: {key}"
                if float(params[key]) <= 0:
                    return False, f"{key} must be positive"
            
            # Check thickness constraints
            if float(params['flange_thickness']) > float(params['depth']) / 2:
                return False, "Flange thickness too large for depth"
            if float(params['web_thickness']) > float(params['width']) / 2:
                return False, "Web thickness too large for width"
        
        elif section_type == "Box Section":
            if 'wall_thickness' in params:
                t = float(params['wall_thickness'])
                if t >= float(params['depth']) / 2 or t >= float(params['width']) / 2:
                    return False, "Wall thickness too large for dimensions"
        
        elif section_type == "Circular Hollow":
            if float(params['thickness']) >= float(params['outer_diameter']) / 2:
                return False, "Thickness must be less than radius"
        
        elif section_type == "Custom Polygon":
            nodes = params.get('nodes', [])
            if len(nodes) < 3:
                return False, "Polygon must have at least 3 nodes"
        
        return True, ""
