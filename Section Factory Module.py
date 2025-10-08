"""
Section Factory Module
Handles creation of various section types using sectionproperties
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

class SectionFactory:
    """Factory class for creating different section types"""
    
    def __init__(self):
        # Default material: structural steel
        self.material = Material(
            name="Steel",
            elastic_modulus=200000,  # MPa
            poissons_ratio=0.3,
            yield_strength=250,  # MPa
            density=7.85e-9,  # tonnes/mm³
            color="grey"
        )
        
    def create_section(self, section_type, params):
        """
        Create a section based on type and parameters
        
        Args:
            section_type (str): Type of section to create
            params (dict): Geometric parameters for the section
            
        Returns:
            Section: Analyzed section object
        """
        # Create geometry based on type
        if section_type == "I-Beam":
            geom = self.create_ibeam(params)
        elif section_type == "Box Section":
            geom = self.create_box(params)
        elif section_type == "Channel":
            geom = self.create_channel(params)
        elif section_type == "Circular":
            geom = self.create_circular(params)
        elif section_type == "Circular Hollow":
            geom = self.create_circular_hollow(params)
        elif section_type == "T-Section":
            geom = self.create_tsection(params)
        elif section_type == "Angle":
            geom = self.create_angle(params)
        elif section_type == "Custom Polygon":
            geom = self.create_polygon(params)
        else:
            raise ValueError(f"Unknown section type: {section_type}")
        
        # Create mesh
        geom.create_mesh(mesh_sizes=[params.get('mesh_size', 10)])
        
        # Create and analyze section
        section = Section(geometry=geom)
        section.calculate_geometric_properties()
        section.calculate_warping_properties()
        section.calculate_plastic_properties()
        
        return section
    
    def create_ibeam(self, params):
        """Create I-beam section"""
        return i_section(
            d=params['depth'],
            b=params['width'],
            t_f=params['flange_thickness'],
            t_w=params['web_thickness'],
            r=params.get('root_radius', 0),
            n_r=params.get('n_r', 8),
            material=self.material
        )
    
    def create_box(self, params):
        """Create box/hollow rectangular section"""
        # Create outer rectangle
        outer = rectangular_section(
            d=params['depth'],
            b=params['width'],
            material=self.material
        )
        
        # Create inner rectangle (hollow)
        if params.get('wall_thickness'):
            t = params['wall_thickness']
            inner = rectangular_section(
                d=params['depth'] - 2*t,
                b=params['width'] - 2*t,
                material=self.material
            ).shift_section(x_offset=t, y_offset=t)
            return outer - inner
        else:
            # Separate web and flange thicknesses
            t_w = params.get('web_thickness', 10)
            t_f = params.get('flange_thickness', 10)
            inner = rectangular_section(
                d=params['depth'] - 2*t_f,
                b=params['width'] - 2*t_w,
                material=self.material
            ).shift_section(x_offset=t_w, y_offset=t_f)
            return outer - inner
    
    def create_channel(self, params):
        """Create channel section"""
        return channel_section(
            d=params['depth'],
            b=params['width'],
            t_f=params['flange_thickness'],
            t_w=params['web_thickness'],
            r=params.get('root_radius', 0),
            n_r=params.get('n_r', 8),
            material=self.material
        )
    
    def create_circular(self, params):
        """Create circular section"""
        return circular_section(
            d=params['diameter'],
            n=params.get('n_circle', 64),
            material=self.material
        )
    
    def create_circular_hollow(self, params):
        """Create circular hollow section (pipe)"""
        return circular_hollow_section(
            d=params['outer_diameter'],
            t=params['thickness'],
            n=params.get('n_circle', 64),
            material=self.material
        )
    
    def create_tsection(self, params):
        """Create T-section"""
        return tee_section(
            d=params['depth'],
            b=params['width'],
            t_f=params['flange_thickness'],
            t_w=params['web_thickness'],
            r=params.get('root_radius', 0),
            n_r=params.get('n_r', 8),
            material=self.material
        )
    
    def create_angle(self, params):
        """Create angle section"""
        return angle_section(
            d=params['leg1_length'],
            b=params['leg2_length'],
            t=params['thickness'],
            r_r=params.get('root_radius', 0),
            r_t=params.get('toe_radius', 0),
            n_r=params.get('n_r', 8),
            material=self.material
        )
    
    def create_polygon(self, params):
        """Create custom polygon section from nodes"""
        nodes = params.get('nodes', [])
        
        if len(nodes) < 3:
            raise ValueError("Polygon must have at least 3 nodes")
        
        # Create polygon from nodes
        polygon = Polygon(nodes)
        
        # Handle holes if specified
        if 'holes' in params and params['holes']:
            for hole in params['holes']:
                if len(hole) >= 3:
                    hole_polygon = Polygon(hole)
                    polygon = polygon.difference(hole_polygon)
        
        # Create geometry
        return Geometry(geom=polygon, material=self.material)
    
    @staticmethod
    def validate_parameters(section_type, params):
        """
        Validate section parameters
        
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            if section_type == "I-Beam":
                if params['flange_thickness'] * 2 >= params['depth']:
                    return False, "Flange thickness too large for depth"
                if params['web_thickness'] >= params['width']:
                    return False, "Web thickness exceeds width"
                    
            elif section_type == "Box Section":
                if 'wall_thickness' in params:
                    if params['wall_thickness'] * 2 >= min(params['width'], params['depth']):
                        return False, "Wall thickness too large"
                        
            elif section_type == "Circular Hollow":
                if params['thickness'] * 2 >= params['outer_diameter']:
                    return False, "Wall thickness exceeds diameter"
                    
            elif section_type == "Custom Polygon":
                if len(params.get('nodes', [])) < 3:
                    return False, "Need at least 3 nodes for polygon"
                    
            # Check all parameters are positive
            for key, value in params.items():
                if isinstance(value, (int, float)) and value <= 0:
                    return False, f"{key} must be positive"
                    
            return True, "Valid parameters"
            
        except Exception as e:
            return False, str(e)
