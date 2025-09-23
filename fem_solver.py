"""
fem_solver.py - Solver for groundwater flow using FDM or FEM
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RectBivariateSpline, griddata
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
        
        # Central differences for interior points
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                # Horizontal gradient
                dh_dx = (self.H[j, i+1] - self.H[j, i-1]) / (2*dx)
                qx[j, i] = -self.k_field[j, i] * dh_dx
                
                # Vertical gradient
                dh_dy = (self.H[j+1, i] - self.H[j-1, i]) / (2*dy)
                qy[j, i] = -self.k_field[j, i] * dh_dy
        
        # Boundary points using forward/backward differences
        self._calculate_boundary_velocities(qx, qy)
        
        self.qx = qx
        self.qy = qy
        
        return qx, qy
    
    def _calculate_boundary_velocities(self, qx: np.ndarray, qy: np.ndarray):
        """Calculate velocities at boundary points"""
        nx, ny = self.domain.nx, self.domain.ny
        dx, dy = self.domain.dx, self.domain.dy
        
        # Left and right boundaries
        for j in range(ny):
            # Left boundary
            if j > 0 and j < ny-1:
                dh_dx = (self.H[j, 1] - self.H[j, 0]) / dx
                qx[j, 0] = -self.k_field[j, 0] * dh_dx
                dh_dy = (self.H[j+1, 0] - self.H[j-1, 0]) / (2*dy)
                qy[j, 0] = -self.k_field[j, 0] * dh_dy
            
            # Right boundary
            if j > 0 and j < ny-1:
                dh_dx = (self.H[j, nx-1] - self.H[j, nx-2]) / dx
                qx[j, nx-1] = -self.k_field[j, nx-1] * dh_dx
                dh_dy = (self.H[j+1, nx-1] - self.H[j-1, nx-1]) / (2*dy)
                qy[j, nx-1] = -self.k_field[j, nx-1] * dh_dy
        
        # Top and bottom boundaries
        for i in range(nx):
            # Top boundary
            if i > 0 and i < nx-1:
                dh_dx = (self.H[0, i+1] - self.H[0, i-1]) / (2*dx)
                qx[0, i] = -self.k_field[0, i] * dh_dx
            dh_dy = (self.H[1, i] - self.H[0, i]) / dy
            qy[0, i] = -self.k_field[0, i] * dh_dy
            
            # Bottom boundary
            if i > 0 and i < nx-1:
                dh_dx = (self.H[ny-1, i+1] - self.H[ny-1, i-1]) / (2*dx)
                qx[ny-1, i] = -self.k_field[ny-1, i] * dh_dx
            dh_dy = (self.H[ny-1, i] - self.H[ny-2, i]) / dy
            qy[ny-1, i] = -self.k_field[ny-1, i] * dh_dy
    
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
                if iy_bottom < self.domain.ny:
                    # Upward flow into excavation (negative qy = upward)
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
        for pile in self.domain.sheet_piles:
            ix = self.domain.x_to_index(pile.x_position)
            iy_toe = self.domain.depth_to_index(pile.bottom_depth)
            
            # Calculate gradient just downstream of pile toe
            if ix + 2 < self.domain.nx and iy_toe > 0 and iy_toe < self.domain.ny - 1:
                dh_dy = abs((self.H[iy_toe+1, ix+2] - self.H[iy_toe-1, ix+2]) / (2*dy))
                gradients[f'pile_toe_x{pile.x_position:.1f}'] = dh_dy
        
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
    """Finite Difference Method solver for groundwater flow"""
    
    def __init__(self, domain: Domain):
        super().__init__(domain)
        self.k_field = domain.create_permeability_field()
    
    def solve(self) -> np.ndarray:
        """Solve using finite differences"""
        nx, ny = self.domain.nx, self.domain.ny
        dx, dy = self.domain.dx, self.domain.dy
        
        # Total number of unknowns
        N = nx * ny
        
        # Sparse matrix components
        rows, cols, data = [], [], []
        b = np.zeros(N)
        
        # Helper function for global indexing
        def global_index(i, j):
            return j * nx + i
        
        # Helper function for harmonic mean
        def harmonic_mean(k1, k2):
            return 2.0 * k1 * k2 / (k1 + k2 + 1e-20)
        
        # Assemble system of equations
        for j in range(ny):
            for i in range(nx):
                p = global_index(i, j)
                
                # Get boundary conditions
                x = self.domain.x_coords[i]
                y = self.domain.y_coords[j]
                
                # Check if inside excavation
                is_inside_excav = (self.domain.excavation and 
                                 self.domain.excavation.is_inside(x, y))
                
                # === Boundary Conditions ===
                
                # Left boundary
                if i == 0:
                    rows.append(p)
                    cols.append(p)
                    data.append(1.0)
                    b[p] = self.domain.get_boundary_head(x, y, 'left')
                    continue
                
                # Right boundary
                if i == nx - 1:
                    rows.append(p)
                    cols.append(p)
                    data.append(1.0)
                    b[p] = self.domain.get_boundary_head(x, y, 'right')
                    continue
                
                # Top boundary
                if j == 0:
                    rows.append(p)
                    cols.append(p)
                    data.append(1.0)
                    b[p] = self.domain.get_boundary_head(x, y, 'top')
                    continue
                
                # Bottom boundary (no-flow)
                if j == ny - 1:
                    bc = self.domain.boundary_conditions.get('bottom')
                    if bc and bc.type == 'neumann':
                        # No-flow: dh/dy = 0
                        rows.append(p)
                        cols.append(p)
                        data.append(1.0)
                        rows.append(p)
                        cols.append(global_index(i, j-1))
                        data.append(-1.0)
                        b[p] = 0.0
                    else:
                        # Dirichlet
                        rows.append(p)
                        cols.append(p)
                        data.append(1.0)
                        b[p] = self.domain.get_boundary_head(x, y, 'bottom')
                    continue
                
                # === Interior Points ===
                
                # Interface permeabilities (harmonic mean)
                k_east = harmonic_mean(self.k_field[j, i], 
                                      self.k_field[j, min(i+1, nx-1)])
                k_west = harmonic_mean(self.k_field[j, i], 
                                      self.k_field[j, max(i-1, 0)])
                k_north = harmonic_mean(self.k_field[j, i], 
                                       self.k_field[max(j-1, 0), i])
                k_south = harmonic_mean(self.k_field[j, i], 
                                       self.k_field[min(j+1, ny-1), i])
                
                # Finite difference coefficients
                a_east = k_east / dx**2
                a_west = k_west / dx**2
                a_north = k_north / dy**2
                a_south = k_south / dy**2
                a_center = a_east + a_west + a_north + a_south
                
                # Assemble matrix
                rows.append(p)
                cols.append(p)
                data.append(a_center)
                
                # Neighbors
                rows.append(p)
                cols.append(global_index(min(i+1, nx-1), j))
                data.append(-a_east)
                
                rows.append(p)
                cols.append(global_index(max(i-1, 0), j))
                data.append(-a_west)
                
                rows.append(p)
                cols.append(global_index(i, max(j-1, 0)))
                data.append(-a_north)
                
                rows.append(p)
                cols.append(global_index(i, min(j+1, ny-1)))
                data.append(-a_south)
        
        # Solve sparse linear system
        A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
        h_solution = spsolve(A, b)
        self.H = h_solution.reshape((ny, nx))
        
        return self.H
    
    def generate_streamlines(self, num_lines: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate streamlines for flow visualization"""
        if self.qx is None or self.qy is None:
            self.calculate_velocities()
        
        streamlines = []
        
        # Create interpolation functions
        x_interp = self.domain.x_coords
        y_interp = self.domain.y_coords
        
        qx_interp = RectBivariateSpline(y_interp, x_interp, self.qx, kx=1, ky=1)
        qy_interp = RectBivariateSpline(y_interp, x_interp, self.qy, kx=1, ky=1)
        
        # Find effective flow region
        effective_depth_start = 1.0
        effective_depth_end = self.domain.depth - 1.0
        
        # Generate starting points
        start_depths = np.linspace(effective_depth_start, effective_depth_end, num_lines)
        
        for start_depth in start_depths:
            # Start from left boundary
            x_current = 0.5
            y_current = start_depth
            
            stream_x = [x_current]
            stream_y = [y_current]
            
            # Integration parameters
            dt = 0.1
            max_steps = int(self.domain.width / dt * 2)
            
            for step in range(max_steps):
                # Check boundaries
                if (x_current >= self.domain.width - 0.5 or x_current <= 0.5 or
                    y_current >= self.domain.depth - 0.5 or y_current <= 0.5):
                    break
                
                # Get velocities
                try:
                    vx = float(qx_interp(y_current, x_current))
                    vy = float(qy_interp(y_current, x_current))
                except:
                    break
                
                # Check for stagnation
                v_mag = np.sqrt(vx**2 + vy**2)
                if v_mag < 1e-10:
                    break
                
                # Normalize and integrate
                vx_norm = vx / v_mag
                vy_norm = vy / v_mag
                
                x_new = x_current + vx_norm * dt
                y_new = y_current + vy_norm * dt
                
                stream_x.append(x_new)
                stream_y.append(y_new)
                
                x_current = x_new
                y_current = y_new
            
            if len(stream_x) > 5:  # Only keep meaningful streamlines
                streamlines.append((np.array(stream_x), np.array(stream_y)))
        
        return streamlines
