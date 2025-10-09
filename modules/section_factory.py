"""
Section Factory Module - FIXED VERSION
Handles creation of various section types using sectionproperties
Fixed: Proper material handling to avoid composite/geometric conflicts
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
    
    def __init__(self, use_material=True):
        """
        Initialize factory
        
        Args:
            use_material: Whether to apply material properties (default True)
        """
        self.use_material = use_material
        
        # Default material: structural steel
        if use_material:
            self.material = Material(
                name="Steel",
                elastic_modulus=200000,  # MPa
                poissons_ratio=0.3,
                yield_strength=250,  # MPa
                density=7.85e-9,  # tonnes/mm³
                color="grey"
            )
        else:
            self.material = None
        
    def create_section(self, section_type, params):
        """
        Create a section based on type and parameters
        
        Args:
            section_type (str): Type of section to create
            params (dict): Geometric parameters for the section
            
        Returns:
            Section: Analyzed section object
        """
        try:
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
            
            # Create mesh with error handling
            mesh_size = params.get('mesh_size', 10)
            try:
                geom.create_mesh(mesh_sizes=[mesh_size])
            except Exception as e:
                print(f"Warning: Mesh creation with size {mesh_size} failed: {e}")
                try:
                    geom.create_mesh(mesh_sizes=[5])
                except Exception as e2:
                    print(f"Warning: Mesh creation with size 5 failed: {e2}")
                    geom.create_mesh(mesh_sizes=[1])
            
            # Create and analyze section
            section = Section(geometry=geom)
            
            # Calculate geometric properties first
            section.calculate_geometric_properties()
            
            # Try warping and plastic properties with error handling
            try:
                section.calculate_warping_properties()
            except Exception as e:
                print(f"Note: Warping properties calculation skipped: {e}")
            
            try:
                section.calculate_plastic_properties()
            except Exception as e:
                print(f"Note: Plastic properties calculation skipped: {e}")
            
            return section
            
        except Exception as e:
            import traceback
            print(f"Error creating section: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Error creating section: {str(e)}")
    
    def create_ibeam(self, params):
        """Create I-beam section"""
        try:
            return i_section(
                d=float(params['depth']),
                b=float(params['width']),
                t_f=float(params['flange_thickness']),
                t_w=float(params['web_thickness']),
                r=float(params.get('root_radius', 0)),
                n_r=int(params.get('n_r', 8)),
                material=self.material
            )
        except Exception as e:
            raise ValueError(f"Error creating I-beam: {e}")
    
    def create_box(self, params):
        """Create box/hollow rectangular section"""
        try:
            # Create outer rectangle
            outer = rectangular_section(
                d=float(params['depth']),
                b=float(params['width']),
                material=self.material
            )
            
            # Create inner rectangle (hollow)
            if params.get('wall_thickness'):
                t = float(params['wall_thickness'])
                if params['depth'] <= 2*t or params['width'] <= 2*t:
                    raise ValueError("Wall thickness too large for dimensions")
                
                inner = rectangular_section(
                    d=float(params['depth']) - 2*t,
                    b=float(params['width']) - 2*t,
                    material=self.material
                ).shift_section(x_offset=t, y_offset=t)
                return outer - inner
            else:
                # Separate web and flange thicknesses
                t_w = float(params.get('web_thickness', 10))
                t_f = float(params.get('flange_thickness', 10))
                
                if params['depth'] <= 2*t_f or params['width'] <= 2*t_w:
                    raise ValueError("Wall thickness too large for dimensions")
                
                inner = rectangular_section(
                    d=float(params['depth']) - 2*t_f,
                    b=float(params['width']) - 2*t_w,
                    material=self.material
                ).shift_section(x_offset=t_w, y_offset=t_f)
                return outer - inner
        except Exception as e:
            raise ValueError(f"Error creating box section: {e}")
    
    def create_channel(self, params):
        """Create channel section"""
        try:
            return channel_section(
                d=float(params['depth']),
                b=float(params['width']),
                t_f=float(params['flange_thickness']),
                t_w=float(params['web_thickness']),
                r=float(params.get('root_radius', 0)),
                n_r=int(params.get('n_r', 8)),
                material=self.material
            )
        except Exception as e:
            raise ValueError(f"Error creating channel: {e}")
    
    def create_circular(self, params):
        """Create circular section"""
        try:
            return circular_section(
                d=float(params['diameter']),
                n=int(params.get('n_circle', 64)),
                material=self.material
            )
        except Exception as e:
            raise ValueError(f"Error creating circular section: {e}")
    
    def create_circular_hollow(self, params):
        """Create circular hollow section (pipe)"""
        try:
            outer_d = float(params['outer_diameter'])
            thickness = float(params['thickness'])
            
            if thickness * 2 >= outer_d:
                raise ValueError("Wall thickness must be less than radius")
            
            return circular_hollow_section(
                d=outer_d,
                t=thickness,
                n=int(params.get('n_circle', 64)),
                material=self.material
            )
        except Exception as e:
            raise ValueError(f"Error creating circular hollow section: {e}")
    
    def create_tsection(self, params):
        """Create T-section"""
        try:
            return tee_section(
                d=float(params['depth']),
                b=float(params['width']),
                t_f=float(params['flange_thickness']),
                t_w=float(params['web_thickness']),
                r=float(params.get('root_radius', 0)),
                n_r=int(params.get('n_r', 8)),
                material=self.material
            )
        except Exception as e:
            raise ValueError(f"Error creating T-section: {e}")
    
    def create_angle(self, params):
        """Create angle section"""
        try:
            return angle_section(
                d=float(params['leg1_length']),
                b=float(params['leg2_length']),
                t=float(params['thickness']),
                r_r=float(params.get('root_radius', 0)),
                r_t=float(params.get('toe_radius', 0)),
                n_r=int(params.get('n_r', 8)),
                material=self.material
            )
        except Exception as e:
            raise ValueError(f"Error creating angle: {e}")
    
    def create_polygon(self, params):
        """Create custom polygon section from nodes"""
        try:
            nodes = params.get('nodes', [])
            
            if len(nodes) < 3:
                raise ValueError("Polygon must have at least 3 nodes")
            
            # Validate nodes are numeric
            try:
                validated_nodes = [(float(x), float(y)) for x, y in nodes]
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid node coordinates: {e}")
            
            # Create polygon from nodes
            try:
                polygon = Polygon(validated_nodes)
                
                if not polygon.is_valid:
                    # Try to fix invalid polygon
                    polygon = polygon.buffer(0)
                    if not polygon.is_valid:
                        raise ValueError("Invalid polygon geometry (self-intersecting or malformed)")
                
                if polygon.area == 0:
                    raise ValueError("Polygon has zero area")
                
            except Exception as e:
                raise ValueError(f"Error creating polygon: {e}")
            
            # Handle holes if specified
            if 'holes' in params and params['holes']:
                for hole in params['holes']:
                    if len(hole) >= 3:
                        try:
                            hole_coords = [(float(x), float(y)) for x, y in hole]
                            hole_polygon = Polygon(hole_coords)
                            if hole_polygon.is_valid and hole_polygon.within(polygon):
                                polygon = polygon.difference(hole_polygon)
                        except Exception as e:
                            print(f"Warning: Could not create hole: {e}")
            
            # Create geometry
            return Geometry(geom=polygon, material=self.material)
            
        except Exception as e:
            raise ValueError(f"Error creating polygon: {e}")
    
    @staticmethod
    def validate_parameters(section_type, params):
        """
