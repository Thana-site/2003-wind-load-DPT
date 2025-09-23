"""
geometry.py - Domain, layers, sheet pile, and excavation definition
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class SoilLayer:
    """Represents a soil layer with properties"""
    depth_top: float  # Depth from surface (positive downward)
    depth_bottom: float
    hydraulic_conductivity: float  # m/s
    porosity: float = 0.3
    name: str = "Soil"
    color: str = "#8B4513"  # Default brown color
    
    @property
    def thickness(self) -> float:
        return self.depth_bottom - self.depth_top
    
    def contains_depth(self, depth: float) -> bool:
        """Check if a depth point is within this layer"""
        return self.depth_top <= depth <= self.depth_bottom


@dataclass
class SheetPile:
    """Represents sheet pile wall configuration"""
    x_position: float  # Horizontal position
    top_depth: float = 0.0  # Usually at ground level
    bottom_depth: float = 10.0  # Total penetration depth
    thickness: float = 0.3  # Wall thickness for visualization
    permeability: float = 1e-12  # Very low permeability
    
    @property
    def length(self) -> float:
        return self.bottom_depth - self.top_depth


@dataclass
class Excavation:
    """Represents excavation geometry"""
    left_x: float
    right_x: float
    depth: float
    water_level: Optional[float] = None  # Water level inside excavation (if any)
    
    @property
    def width(self) -> float:
        return self.right_x - self.left_x
    
    def is_inside(self, x: float, y: float) -> bool:
        """Check if point (x,y) is inside excavated area"""
        return self.left_x < x < self.right_x and 0 <= y <= self.depth


@dataclass
class BoundaryCondition:
    """Represents a boundary condition"""
    type: str  # 'dirichlet', 'neumann', 'flux'
    value: float = 0.0  # Head value for Dirichlet, flux value for Neumann
    location: str = ""  # 'left', 'right', 'top', 'bottom', 'sheet_pile'


@dataclass
class Domain:
    """Represents the computational domain"""
    width: float = 40.0  # Total domain width
    depth: float = 15.0  # Total domain depth
    
    # Grid parameters
    nx: int = 201  # Number of nodes in x-direction
    ny: int = 151  # Number of nodes in y-direction
    
    # Soil layers
    soil_layers: List[SoilLayer] = field(default_factory=list)
    
    # Sheet piles (can have multiple)
    sheet_piles: List[SheetPile] = field(default_factory=list)
    
    # Excavation
    excavation: Optional[Excavation] = None
    
    # Boundary conditions
    boundary_conditions: dict = field(default_factory=dict)
    
    # Water levels
    water_level_left: float = 2.0  # Depth below ground
    water_level_right: float = 2.0
    
    def __post_init__(self):
        """Initialize derived properties"""
        self.dx = self.width / (self.nx - 1)
        self.dy = self.depth / (self.ny - 1)
        self.x_coords = np.linspace(0, self.width, self.nx)
        self.y_coords = np.linspace(0, self.depth, self.ny)
        
        # Default boundary conditions if not specified
        if not self.boundary_conditions:
            self.boundary_conditions = {
                'left': BoundaryCondition('dirichlet', -self.water_level_left, 'left'),
                'right': BoundaryCondition('dirichlet', -self.water_level_right, 'right'),
                'top': BoundaryCondition('mixed', 0.0, 'top'),  # Mixed based on location
                'bottom': BoundaryCondition('neumann', 0.0, 'bottom')  # No-flow
            }
    
    def add_soil_layer(self, layer: SoilLayer):
        """Add a soil layer to the domain"""
        self.soil_layers.append(layer)
        self.soil_layers.sort(key=lambda x: x.depth_top)
    
    def add_sheet_pile(self, pile: SheetPile):
        """Add a sheet pile to the domain"""
        self.sheet_piles.append(pile)
    
    def set_excavation(self, excavation: Excavation):
        """Set the excavation geometry"""
        self.excavation = excavation
    
    def get_permeability_at_point(self, x: float, y: float) -> float:
        """Get hydraulic conductivity at a specific point"""
        # Check if point is on a sheet pile
        for pile in self.sheet_piles:
            if abs(x - pile.x_position) < pile.thickness / 2:
                if pile.top_depth <= y <= pile.bottom_depth:
                    return pile.permeability
        
        # Otherwise get soil layer permeability
        for layer in self.soil_layers:
            if layer.contains_depth(y):
                return layer.hydraulic_conductivity
        
        # Default if no layer found
        return 1e-5
    
    def create_permeability_field(self) -> np.ndarray:
        """Create 2D permeability field for the entire domain"""
        k_field = np.zeros((self.ny, self.nx))
        
        for j in range(self.ny):
            for i in range(self.nx):
                x = self.x_coords[i]
                y = self.y_coords[j]
                k_field[j, i] = self.get_permeability_at_point(x, y)
        
        return k_field
    
    def get_boundary_head(self, x: float, y: float, location: str) -> float:
        """Get hydraulic head at boundary based on conditions"""
        bc = self.boundary_conditions.get(location)
        
        if bc and bc.type == 'dirichlet':
            # Special handling for top boundary
            if location == 'top' and self.excavation:
                if self.excavation.is_inside(x, 0):
                    # Inside excavation
                    if self.excavation.water_level is not None:
                        return -self.excavation.water_level
                    else:
                        return -self.excavation.depth  # Dry excavation
                else:
                    # Outside excavation
                    return bc.value
            else:
                return bc.value
        
        return 0.0
    
    def depth_to_index(self, depth: float) -> int:
        """Convert depth to array index"""
        idx = int(round(depth / self.dy))
        return max(0, min(idx, self.ny - 1))
    
    def x_to_index(self, x_pos: float) -> int:
        """Convert x-position to array index"""
        idx = int(round(x_pos / self.dx))
        return max(0, min(idx, self.nx - 1))
    
    def create_mesh_refinement_map(self) -> np.ndarray:
        """Create a map indicating where mesh refinement is needed"""
        refinement = np.ones((self.ny, self.nx))
        
        # Refine near sheet piles
        for pile in self.sheet_piles:
            ix = self.x_to_index(pile.x_position)
            iy_top = self.depth_to_index(pile.top_depth)
            iy_bottom = self.depth_to_index(pile.bottom_depth)
            
            # Higher refinement near pile
            for j in range(max(0, iy_top - 5), min(self.ny, iy_bottom + 5)):
                for i in range(max(0, ix - 5), min(self.nx, ix + 5)):
                    distance = np.sqrt((i - ix)**2 + (j - (iy_top + iy_bottom)/2)**2)
                    refinement[j, i] = max(0.2, 1.0 - 0.8 * np.exp(-distance/3))
        
        # Refine near excavation boundaries
        if self.excavation:
            ix_left = self.x_to_index(self.excavation.left_x)
            ix_right = self.x_to_index(self.excavation.right_x)
            iy_depth = self.depth_to_index(self.excavation.depth)
            
            # Excavation edges
            for j in range(max(0, iy_depth - 3), min(self.ny, iy_depth + 3)):
                for i in range(max(0, ix_left - 3), min(self.nx, ix_right + 3)):
                    refinement[j, i] = min(refinement[j, i], 0.5)
        
        return refinement
    
    def validate(self) -> List[str]:
        """Validate domain configuration and return any warnings"""
        warnings = []
        
        # Check soil layers coverage
        if self.soil_layers:
            if self.soil_layers[0].depth_top > 0:
                warnings.append("Warning: No soil layer starts at surface (depth=0)")
            if self.soil_layers[-1].depth_bottom < self.depth:
                warnings.append(f"Warning: No soil layer extends to domain bottom ({self.depth}m)")
        else:
            warnings.append("Warning: No soil layers defined")
        
        # Check sheet pile configuration
        for i, pile in enumerate(self.sheet_piles):
            if pile.bottom_depth > self.depth:
                warnings.append(f"Warning: Sheet pile {i+1} extends beyond domain bottom")
            if pile.x_position < 0 or pile.x_position > self.width:
                warnings.append(f"Warning: Sheet pile {i+1} outside domain width")
        
        # Check excavation
        if self.excavation:
            if self.excavation.depth > self.depth:
                warnings.append("Warning: Excavation deeper than domain")
            if not self.sheet_piles:
                warnings.append("Warning: Excavation defined without sheet piles")
        
        return warnings


def create_cofferdam_domain(
    sheet_pile_length: float = 10.0,
    excavation_depth: float = 6.0,
    excavation_width: float = 10.0,
    domain_width: float = 40.0,
    domain_depth: float = 15.0,
    water_level_outside: float = 2.0,
    water_level_inside: float = 4.0,
    soil_layers_config: List[dict] = None
) -> Domain:
    """
    Create a standard cofferdam configuration
    
    Args:
        sheet_pile_length: Total length of sheet pile
        excavation_depth: Depth of excavation below ground
        excavation_width: Width between sheet piles
        domain_width: Total domain width
        domain_depth: Total domain depth
        water_level_outside: Water level outside excavation (depth below ground)
        water_level_inside: Water level inside excavation (depth below ground)
        soil_layers_config: List of dicts with layer properties
    
    Returns:
        Configured Domain object
    """
    domain = Domain(width=domain_width, depth=domain_depth)
    
    # Set water levels
    domain.water_level_left = water_level_outside
    domain.water_level_right = water_level_outside
    
    # Add soil layers
    if soil_layers_config:
        for layer_config in soil_layers_config:
            layer = SoilLayer(**layer_config)
            domain.add_soil_layer(layer)
    else:
        # Default two-layer system
        domain.add_soil_layer(SoilLayer(0, 5, 1e-5, name="Sand"))
        domain.add_soil_layer(SoilLayer(5, 15, 1e-6, name="Clay"))
    
    # Calculate sheet pile positions
    center_x = domain_width / 2
    left_pile_x = center_x - excavation_width / 2
    right_pile_x = center_x + excavation_width / 2
    
    # Add sheet piles
    domain.add_sheet_pile(SheetPile(left_pile_x, 0, sheet_pile_length))
    domain.add_sheet_pile(SheetPile(right_pile_x, 0, sheet_pile_length))
    
    # Set excavation
    excavation = Excavation(left_pile_x, right_pile_x, excavation_depth, water_level_inside)
    domain.set_excavation(excavation)
    
    return domain
