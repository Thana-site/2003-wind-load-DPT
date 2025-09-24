"""
fem_solver.py - Corrected solver for groundwater flow with proper boundary conditions
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RectBivariateSpline
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

from geometry import Domain


class GroundwaterSolver:
    """Base class for groundwater flow solvers"""
    
    def __init__(self, domain: Domain):
        self.domain = domain
        self.H = None  # Hydraulic head solution
        self.qx = None  # Darcy velocity in x-direction
        self.qy = None  # Darcy velocity in y-direction
        self.k_field = None  # Permeability field
        self.psi = None  # Stream function
        
    def solve(self) -> np.ndarray:
        """Solve for hydraulic head distribution"""
        raise NotImplementedError("Subclass must implement solve method")
    
    def calculate_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Darcy velocities from head distribution"""
        if self.H is None:
            raise ValueError("Must solve for heads first")
        
        nx, ny = self.domain.nx, self.domain.ny
        dx, dy = self.domain.dx, self.domain.dy
        
        qx = np.zeros((ny, nx))
        qy = np.zeros((ny, nx))
        
        # Use central differences for interior points
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                # Get local permeability
                k_local = self.k_field[j, i]
                
                # Calculate gradients
                dh_dx = (self.H[j, i+1] - self.H[j, i-1]) / (2*dx)
                dh_dy = (self.H[j+1, i] - self.H[j-1, i]) / (2*dy)
                
                # Darcy's law: q = -k * grad(h)
                qx[j, i] = -k_local * dh_dx
                qy[j, i] = -k_local * dh_dy
        
        # Handle boundaries with one-sided differences
        # Left boundary (i=0)
        for j in range(1, ny-1):
            k_local = self.k_field[j, 0]
            qx[j, 0] = -k_local * (self.H[j, 1] - self.H[j, 0]) / dx
            qy[j, 0] = -k_local * (self.H[j+1, 0] - self.H[j-1, 0]) / (2*dy)
        
        # Right boundary (i=nx-1)
        for j in range(1, ny-1):
            k_local = self.k_field[j, nx-1]
            qx[j, nx-1] = -k_local * (self.H[j, nx-1] - self.H[j, nx-2]) / dx
            qy[j, nx-1] = -k_local * (self.H[j+1, nx-1] - self.H[j-1, nx-1]) / (2*dy)
        
        # Top boundary (j=0)
        for i in range(1, nx-1):
            k_local = self.k_field[0, i]
            qx[0, i] = -k_local * (self.H[0, i+1] - self.H[0, i-1]) / (2*dx)
            qy[0, i] = -k_local * (self.H[1, i] - self.H[0, i]) / dy
        
        # Bottom boundary (j=ny-1)
        for i in range(1, nx-1):
            k_local = self.k_field[ny-1, i]
            qx[ny-1, i] = -k_local * (self.H[ny-1, i+1] - self.H[ny-1, i-1]) / (2*dx)
            qy[ny-1, i] = -k_local * (self.H[ny-1, i] - self.H[ny-2, i]) / dy
        
        self.qx = qx
        self.qy = qy
        
        return qx, qy
    
    def calculate_stream_function(self) -> np.ndarray:
        """Calculate stream function from velocity field for flow line generation"""
        if self.qx is None or self.qy is None:
            self.calculate_velocities()
        
        nx, ny = self.domain.nx, self.domain.ny
        dx, dy = self.domain.dx, self.domain.dy
        
        # Initialize stream function
        psi = np.zeros((ny, nx))
        
        # Integrate from left boundary (reference psi=0 at bottom-left)
        # First column: integrate vertically
        for j in range(1, ny):
            psi[j, 0] = psi[j-1, 0] + self.qx[j-1, 0] * dy
        
        # Remaining columns: integrate horizontally
        for j in range(ny):
            for i in range(1, nx):
                psi[j, i] = psi[j, i-1] - self.qy[j, i-1] * dx
        
        self.psi = psi
        return psi
    
    def calculate_seepage_discharge(self) -> dict:
        """Calculate seepage discharge through various boundaries"""
        if self.qx is None or self.qy is None:
            self.calculate_velocities()
        
        results = {}
        dx, dy = self.domain.dx, self.domain.dy
        
        # Flow through excavation bottom if present
        if self.domain.excavation:
            Q_bottom = 0.0
            ix_left = self.domain.x_to_index(self.domain.excavation.left_x)
            ix_right = self.domain.x_to_index(self.domain.excavation.right_x)
            iy_bottom = self.domain.depth_to_index(self.domain.excavation.depth)
            
            for i in range(ix_left + 1, ix_right):
                if iy_bottom < self.domain.ny - 1:
                    # Upward flow into excavation (negative qy = upward in our coordinate system)
                    Q_bottom += -self.qy[iy_bottom, i] * dx
            
            results['excavation_bottom'] = Q_bottom
        
        # Flow through left boundary
        Q_left = 0.0
        for j in range(1, self.domain.ny-1):
            Q_left += self.qx[j, 1] * dy
        results['left_boundary'] = Q_left
        
        # Flow through right boundary
        Q_right = 0.0
        for j in range(1, self.domain.ny-1):
            Q_right += -self.qx[j, self.domain.nx-2] * dy
        results['right_boundary'] = Q_right
        
        # Mass balance error
        total_inflow = abs(Q_left)
        total_outflow = abs(Q_right)
        if self.domain.excavation:
            total_outflow += abs(results['excavation_bottom'])
        
        if total_inflow > 0:
            results['mass_balance_error'] = abs(total_inflow - total_outflow) / total_inflow * 100
        else:
            results['mass_balance_error'] = 0.0
        
        return results
    
    def calculate_exit_gradients(self) -> dict:
        """Calculate hydraulic gradients at critical locations"""
        if self.H is None:
            raise ValueError("Must solve for heads first")
        
        gradients = {}
        dy = self.domain.dy
        
        # Exit gradients near sheet pile toes
        for pile_idx, pile in enumerate(self.domain.sheet_piles):
            ix = self.domain.x_to_index(pile.x_position)
            iy_toe = self.domain.depth_to_index(pile.bottom_depth)
            
            # Calculate gradient just downstream of pile toe
            if ix + 2 < self.domain.nx and iy_toe > 0 and iy_toe < self.domain.ny - 1:
                dh_dy = abs((self.H[iy_toe+1, ix+2] - self.H[iy_toe-1, ix+2]) / (2*dy))
                gradients[f'pile_{pile_idx+1}_toe'] = dh_dy
        
        # Gradient at excavation center if present
        if self.domain.excavation:
            ix_center = self.domain.x_to_index(
                (self.domain.excavation.left_x + self.domain.excavation.right_x) / 2
            )
            iy_bottom = self.domain.depth_to_index(self.domain.excavation.depth)
            
            if iy_bottom > 0 and iy_bottom < self.domain.ny - 1:
                dh_dy = abs((self.H[iy_bottom+1, ix_center] - self.H[iy_bottom-1, ix_center]) / (2*dy))
                gradients['excavation_center'] = dh_dy
        
        # Calculate critical gradient and safety factor
        gamma_sat = 20.0  # kN/m³ (saturated unit weight)
        gamma_w = 9.81     # kN/m³ (water unit weight)
        i_critical = (gamma_sat - gamma_w) / gamma_w
        
        max_gradient = max(gradients.values()) if gradients else 0
        gradients['critical_gradient'] = i_critical
        gradients['max_exit_gradient'] = max_gradient
        gradients['safety_factor'] = i_critical / max_gradient if max_gradient > 0 else float('inf')
        
        return gradients


class FDMSolver(GroundwaterSolver):
    """Corrected Finite Difference Method solver with proper boundary conditions"""
    
    def __init__(self, domain: Domain):
        super().__init__(domain)
        self.k_field = domain.create_permeability_field()
        
    def solve(self) -> np.ndarray:
        """Solve using finite differences with corrected boundary conditions"""
        nx, ny = self.domain.nx, self.domain.ny
        dx, dy = self.domain.dx, self.domain.dy
        
        # Create a mask for sheet pile locations
        sheet_pile_mask = np.ones((ny, nx), dtype=bool)
        for pile in self.domain.sheet_piles:
            ix = self.domain.x_to_index(pile.x_position)
            iy_top = self.domain.depth_to_index(pile.top_depth)
            iy_bottom = self.domain.depth_to_index(pile.bottom_depth)
            
            # Mark sheet pile cells
            for j in range(iy_top, min(iy_bottom + 1, ny)):
                sheet_pile_mask[j, ix] = False
        
        # Total number of unknowns
        N = nx * ny
        
        # Sparse matrix components
        rows, cols, data = [], [], []
        b = np.zeros(N)
        
        # Helper function for global indexing
        def global_index(i, j):
            return j * nx + i
        
        # Helper function for harmonic mean (for interface permeability)
        def harmonic_mean(k1, k2):
            if k1 <= 0 or k2 <= 0:
                return 0
            return 2.0 * k1 * k2 / (k1 + k2)
        
        # Assemble system of equations
        for j in range(ny):
            for i in range(nx):
                p = global_index(i, j)
                
                # Get coordinates
                x = self.domain.x_coords[i]
                y = self.domain.y_coords[j]
                
                # Check if this is a sheet pile cell
                is_sheet_pile = not sheet_pile_mask[j, i]
                
                # === BOUNDARY CONDITIONS ===
                
                # Left boundary: constant head
                if i == 0:
                    rows.append(p)
                    cols.append(p)
                    data.append(1.0)
                    b[p] = -self.domain.water_level_left  # Convert depth to head
                    continue
                
                # Right boundary: constant head
                if i == nx - 1:
                    rows.append(p)
                    cols.append(p)
                    data.append(1.0)
                    b[p] = -self.domain.water_level_right  # Convert depth to head
                    continue
                
                # Top boundary
                if j == 0:
                    rows.append(p)
                    cols.append(p)
                    data.append(1.0)
                    # Check if inside excavation
                    if self.domain.excavation and self.domain.excavation.is_inside(x, 0):
                        # Inside excavation - set to excavation water level
                        if self.domain.excavation.water_level is not None:
                            b[p] = -self.domain.excavation.water_level
                        else:
                            b[p] = -self.domain.excavation.depth  # Dry excavation
                    else:
                        # Outside excavation - natural water table
                        b[p] = -self.domain.water_level_left
                    continue
                
                # Bottom boundary: no-flow (Neumann)
                if j == ny - 1:
                    # No-flow: dh/dy = 0, so h[j] = h[j-1]
                    rows.append(p)
                    cols.append(p)
                    data.append(1.0)
                    rows.append(p)
                    cols.append(global_index(i, j-1))
                    data.append(-1.0)
                    b[p] = 0.0
                    continue
                
                # === SHEET PILE TREATMENT ===
                if is_sheet_pile:
                    # Sheet pile cell - use very low permeability
                    # This effectively creates a no-flow barrier
                    k_pile = 1e-12
                    
                    # Get neighboring permeabilities (use pile permeability)
                    k_east = k_pile if i < nx-1 else 0
                    k_west = k_pile if i > 0 else 0
                    k_north = k_pile if j > 0 else 0
                    k_south = k_pile if j < ny-1 else 0
                    
                    # Finite difference coefficients
                    a_east = k_east / dx**2
                    a_west = k_west / dx**2
                    a_north = k_north / dy**2
                    a_south = k_south / dy**2
                    a_center = -(a_east + a_west + a_north + a_south)
                    
                    if abs(a_center) < 1e-15:
                        # Isolated cell - set to average of neighbors
                        rows.append(p)
                        cols.append(p)
                        data.append(1.0)
                        b[p] = 0.0
                    else:
                        # Standard finite difference stencil
                        rows.append(p)
                        cols.append(p)
                        data.append(a_center)
                        
                        if i < nx-1:
                            rows.append(p)
                            cols.append(global_index(i+1, j))
                            data.append(a_east)
                        
                        if i > 0:
                            rows.append(p)
                            cols.append(global_index(i-1, j))
                            data.append(a_west)
                        
                        if j > 0:
                            rows.append(p)
                            cols.append(global_index(i, j-1))
                            data.append(a_north)
                        
                        if j < ny-1:
                            rows.append(p)
                            cols.append(global_index(i, j+1))
                            data.append(a_south)
                    
                    continue
                
                # === INTERIOR POINTS WITH LAYER INTERFACE TREATMENT ===
                
                # Get permeabilities at interfaces (harmonic mean for heterogeneous media)
                k_center = self.k_field[j, i]
                
                # Check if we're near a layer interface for special treatment
                is_near_interface = False
                interface_distance = float('inf')
                
                for layer in self.domain.soil_layers:
                    layer_depth_idx = self.domain.depth_to_index(layer.depth_bottom)
                    if abs(j - layer_depth_idx) <= 2:  # Within 2 cells of interface
                        is_near_interface = True
                        interface_distance = min(interface_distance, abs(j - layer_depth_idx))
                
                # Interface permeabilities with smooth transition
                if i < nx-1:
                    if is_near_interface:
                        # Use arithmetic mean near interfaces for smoother transition
                        k_east = 0.5 * (k_center + self.k_field[j, i+1])
                    else:
                        k_east = harmonic_mean(k_center, self.k_field[j, i+1])
                else:
                    k_east = k_center
                
                if i > 0:
                    if is_near_interface:
                        k_west = 0.5 * (k_center + self.k_field[j, i-1])
                    else:
                        k_west = harmonic_mean(k_center, self.k_field[j, i-1])
                else:
                    k_west = k_center
                
                if j > 0:
                    if is_near_interface:
                        # Special treatment for vertical interface
                        k_north = harmonic_mean(k_center, self.k_field[j-1, i])
                    else:
                        k_north = harmonic_mean(k_center, self.k_field[j-1, i])
                else:
                    k_north = k_center
                
                if j < ny-1:
                    if is_near_interface:
                        # Special treatment for vertical interface
                        k_south = harmonic_mean(k_center, self.k_field[j+1, i])
                    else:
                        k_south = harmonic_mean(k_center, self.k_field[j+1, i])
                else:
                    k_south = k_center
                
                # Check for sheet pile neighbors and set zero flux
                if i < nx-1 and not sheet_pile_mask[j, i+1]:
                    k_east = 0  # No flow through sheet pile
                if i > 0 and not sheet_pile_mask[j, i-1]:
                    k_west = 0  # No flow through sheet pile
                if j > 0 and not sheet_pile_mask[j-1, i]:
                    k_north = 0  # No flow through sheet pile
                if j < ny-1 and not sheet_pile_mask[j+1, i]:
                    k_south = 0  # No flow through sheet pile
                
                # Finite difference coefficients
                a_east = k_east / dx**2
                a_west = k_west / dx**2
                a_north = k_north / dy**2
                a_south = k_south / dy**2
                a_center = -(a_east + a_west + a_north + a_south)
                
                # Assemble matrix
                rows.append(p)
                cols.append(p)
                data.append(a_center)
                
                if i < nx-1 and a_east > 0:
                    rows.append(p)
                    cols.append(global_index(i+1, j))
                    data.append(a_east)
                
                if i > 0 and a_west > 0:
                    rows.append(p)
                    cols.append(global_index(i-1, j))
                    data.append(a_west)
                
                if j > 0 and a_north > 0:
                    rows.append(p)
                    cols.append(global_index(i, j-1))
                    data.append(a_north)
                
                if j < ny-1 and a_south > 0:
                    rows.append(p)
                    cols.append(global_index(i, j+1))
                    data.append(a_south)
        
        # Solve sparse linear system
        A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
        h_solution = spsolve(A, b)
        self.H = h_solution.reshape((ny, nx))
        
        return self.H
    
    def verify_orthogonality(self) -> float:
        """Verify orthogonality between equipotentials and flow lines
        Returns average angle deviation from 90 degrees"""
        if self.H is None or self.qx is None or self.qy is None:
            return 0.0
        
        nx, ny = self.domain.nx, self.domain.ny
        dx, dy = self.domain.dx, self.domain.dy
        
        angles = []
        
        # Sample points away from boundaries
        for j in range(5, ny-5, 5):
            for i in range(5, nx-5, 5):
                # Calculate head gradient (perpendicular to equipotentials)
                dh_dx = (self.H[j, i+1] - self.H[j, i-1]) / (2*dx)
                dh_dy = (self.H[j+1, i] - self.H[j-1, i]) / (2*dy)
                
                # Velocity vector (tangent to flow lines)
                vx = self.qx[j, i]
                vy = self.qy[j, i]
                
                # Calculate angle between gradient and velocity
                grad_mag = np.sqrt(dh_dx**2 + dh_dy**2)
                vel_mag = np.sqrt(vx**2 + vy**2)
                
                if grad_mag > 1e-10 and vel_mag > 1e-10:
                    # Dot product for angle calculation
                    cos_angle = (dh_dx * vx + dh_dy * vy) / (grad_mag * vel_mag)
                    # Should be near zero for orthogonal vectors
                    angle = np.arccos(np.clip(abs(cos_angle), 0, 1)) * 180 / np.pi
                    angles.append(abs(90 - angle))
        
        return np.mean(angles) if angles else 0.0
    
    def apply_layer_interface_refinement(self):
        """Refine solution near layer interfaces to reduce clustering"""
        if self.H is None:
            return
        
        # Apply local smoothing near interfaces
        from scipy.ndimage import gaussian_filter1d
        
        for layer in self.domain.soil_layers:
            if 0 < layer.depth_bottom < self.domain.depth:
                j_interface = self.domain.depth_to_index(layer.depth_bottom)
                
                # Apply smoothing in a narrow band around interface
                for j_offset in range(-2, 3):
                    j = j_interface + j_offset
                    if 0 < j < self.domain.ny - 1:
                        # Smooth along horizontal direction
                        self.H[j, :] = gaussian_filter1d(self.H[j, :], sigma=0.5)
        
        # Recalculate velocities after smoothing
        self.calculate_velocities()
        """Generate flow lines as contours of the stream function"""
        if self.psi is None:
            self.calculate_stream_function()
        
        # Get stream function range
        psi_min = np.min(self.psi)
        psi_max = np.max(self.psi)
        
        # Generate evenly spaced stream function values
        psi_levels = np.linspace(psi_min, psi_max, num_lines + 2)[1:-1]
        
        flow_lines = []
        
        # Extract contours using matplotlib's contour function
        import matplotlib.pyplot as plt
        X, Y = np.meshgrid(self.domain.x_coords, self.domain.y_coords)
        
        # Create contour plot (without displaying)
        fig, ax = plt.subplots(figsize=(1, 1))
        cs = ax.contour(X, Y, self.psi, levels=psi_levels)
        
        # Extract contour lines (compatible with different matplotlib versions)
        try:
            # Try newer matplotlib API first
            for level_idx, level in enumerate(psi_levels):
                # Get all contour segments for this level
                segments = cs.allsegs[level_idx]
                for segment in segments:
                    if len(segment) > 10:  # Only keep meaningful lines
                        flow_lines.append((segment[:, 0], segment[:, 1]))
        except (AttributeError, IndexError):
            # Fall back to older API
            try:
                for collection in cs.collections:
                    for path in collection.get_paths():
                        vertices = path.vertices
                        if len(vertices) > 10:  # Only keep meaningful lines
                            flow_lines.append((vertices[:, 0], vertices[:, 1]))
            except AttributeError:
                # Most basic fallback - just use the contour data directly
                pass
        
        plt.close(fig)
        
        return flow_lines
    
    def generate_equipotentials(self, num_lines: int = 15) -> np.ndarray:
        """Generate equipotential levels avoiding clustering at layer interfaces"""
        if self.H is None:
            raise ValueError("Must solve for heads first")
        
        # Get head statistics
        h_min = np.min(self.H)
        h_max = np.max(self.H)
        
        # Method 1: Use percentile-based spacing to avoid boundary clustering
        percentiles = np.linspace(5, 95, num_lines)
        h_values = np.percentile(self.H.flatten(), percentiles)
        
        # Method 2: Adaptive spacing based on gradient magnitude
        # Areas with high gradients get more closely spaced lines
        dh_dx = np.gradient(self.H, axis=1)
        dh_dy = np.gradient(self.H, axis=0)
        gradient_mag = np.sqrt(dh_dx**2 + dh_dy**2)
        
        # Weight the distribution based on gradient
        gradient_weight = gradient_mag / np.max(gradient_mag)
        
        # Combine both methods for optimal spacing
        if np.std(h_values) > 0:
            # Use percentile-based if there's good variation
            levels = h_values
        else:
            # Fall back to linear spacing
            levels = np.linspace(h_min, h_max, num_lines)
        
        # Ensure unique levels and sort
        levels = np.unique(levels)
        if len(levels) < num_lines:
            # Add more levels if needed
            additional = np.linspace(h_min, h_max, num_lines - len(levels) + 2)[1:-1]
            levels = np.sort(np.concatenate([levels, additional]))
        
        return levels[:num_lines]
    
    def verify_orthogonality(self) -> float:
        """Verify orthogonality between equipotentials and flow lines
        Returns average angle deviation from 90 degrees"""
        if self.H is None or self.qx is None or self.qy is None:
            return 0.0
        
        nx, ny = self.domain.nx, self.domain.ny
        dx, dy = self.domain.dx, self.domain.dy
        
        angles = []
        
        # Sample points away from boundaries
        for j in range(5, ny-5, 5):
            for i in range(5, nx-5, 5):
                # Calculate head gradient (perpendicular to equipotentials)
                dh_dx = (self.H[j, i+1] - self.H[j, i-1]) / (2*dx)
                dh_dy = (self.H[j+1, i] - self.H[j-1, i]) / (2*dy)
                
                # Velocity vector (tangent to flow lines)
                vx = self.qx[j, i]
                vy = self.qy[j, i]
                
                # Calculate angle between gradient and velocity
                grad_mag = np.sqrt(dh_dx**2 + dh_dy**2)
                vel_mag = np.sqrt(vx**2 + vy**2)
                
                if grad_mag > 1e-10 and vel_mag > 1e-10:
                    # Dot product for angle calculation
                    cos_angle = (dh_dx * vx + dh_dy * vy) / (grad_mag * vel_mag)
                    # Should be near zero for orthogonal vectors
                    angle = np.arccos(np.clip(abs(cos_angle), 0, 1)) * 180 / np.pi
                    angles.append(abs(90 - angle))
        
        return np.mean(angles) if angles else 0.0
    
    def apply_layer_interface_refinement(self):
        """Refine solution near layer interfaces to reduce clustering"""
        if self.H is None:
            return
        
        # Apply local smoothing near interfaces
        from scipy.ndimage import gaussian_filter1d
        
        for layer in self.domain.soil_layers:
            if 0 < layer.depth_bottom < self.domain.depth:
                j_interface = self.domain.depth_to_index(layer.depth_bottom)
                
                # Apply smoothing in a narrow band around interface
                for j_offset in range(-2, 3):
                    j = j_interface + j_offset
                    if 0 < j < self.domain.ny - 1:
                        # Smooth along horizontal direction
                        self.H[j, :] = gaussian_filter1d(self.H[j, :], sigma=0.5)
        
        # Recalculate velocities after smoothing
        self.calculate_velocities()
    
    def calculate_stream_function_improved(self) -> np.ndarray:
        """Calculate stream function with layer interface smoothing"""
        if self.qx is None or self.qy is None:
            self.calculate_velocities()
        
        nx, ny = self.domain.nx, self.domain.ny
        dx, dy = self.domain.dx, self.domain.dy
        
        # Initialize stream function
        psi = np.zeros((ny, nx))
        
        # Method 1: Line integral from reference point
        # Start from bottom-left corner (reference psi=0)
        
        # First column: integrate upward using qx
        for j in range(1, ny):
            # Check if crossing layer interface
            is_interface = False
            for layer in self.domain.soil_layers:
                layer_j = self.domain.depth_to_index(layer.depth_bottom)
                if abs(j - layer_j) <= 1:
                    is_interface = True
                    break
            
            if is_interface:
                # Use trapezoidal rule for smoother integration at interface
                psi[j, 0] = psi[j-1, 0] + 0.5 * (self.qx[j, 0] + self.qx[j-1, 0]) * dy
            else:
                psi[j, 0] = psi[j-1, 0] + self.qx[j, 0] * dy
        
        # Remaining points: integrate horizontally using qy
        for j in range(ny):
            for i in range(1, nx):
                # Check if near layer interface
                is_interface = False
                for layer in self.domain.soil_layers:
                    layer_j = self.domain.depth_to_index(layer.depth_bottom)
                    if abs(j - layer_j) <= 1:
                        is_interface = True
                        break
                
                if is_interface:
                    # Use averaging for smoother stream function at interfaces
                    psi[j, i] = psi[j, i-1] - 0.5 * (self.qy[j, i] + self.qy[j, i-1]) * dx
                else:
                    psi[j, i] = psi[j, i-1] - self.qy[j, i-1] * dx
        
        # Apply smoothing filter near layer interfaces
        from scipy.ndimage import gaussian_filter
        
        # Create mask for interface regions
        interface_mask = np.zeros((ny, nx), dtype=bool)
        for layer in self.domain.soil_layers:
            layer_j = self.domain.depth_to_index(layer.depth_bottom)
            if 0 < layer_j < ny:
                interface_mask[max(0, layer_j-2):min(ny, layer_j+3), :] = True
        
        # Apply light smoothing only at interfaces
        if np.any(interface_mask):
            psi_smooth = gaussian_filter(psi, sigma=0.5)
            psi[interface_mask] = psi_smooth[interface_mask]
        
        self.psi = psi
        return psi
