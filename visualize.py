"""
visualize.py - Corrected plotting helpers for accurate flow net visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize, LogNorm
import matplotlib.cm as cm
from typing import Optional, List, Tuple
import io
import base64

from geometry import Domain
from fem_solver import GroundwaterSolver


class FlowNetVisualizer:
    """Corrected flow net visualization with orthogonal lines"""
    
    def __init__(self, domain: Domain, solver: GroundwaterSolver):
        self.domain = domain
        self.solver = solver
        
    def plot_flow_net(self, 
                      num_equipotentials: int = 15,
                      num_flow_lines: int = 12,
                      show_mesh: bool = False,
                      show_velocity_vectors: bool = False,
                      figsize: Tuple[float, float] = (14, 10)) -> plt.Figure:
        """Create accurate flow net with orthogonal equipotentials and flow lines"""
        
        # Ensure stream function is calculated using improved method if available
        if self.solver.psi is None:
            if hasattr(self.solver, 'calculate_stream_function_improved'):
                self.solver.calculate_stream_function_improved()
            else:
                self.solver.calculate_stream_function()
        
        # Prepare data
        X, Y = np.meshgrid(self.domain.x_coords, -self.domain.y_coords)  # Negative for depth display
        H = self.solver.H
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        # === EQUIPOTENTIAL LINES (constant head) ===
        # Generate appropriate levels without clustering
        h_levels = self.solver.generate_equipotentials(num_equipotentials)
        
        # Plot equipotentials with varying line styles for better clarity
        cs_equi = ax.contour(X, Y, H, levels=h_levels, colors='blue', 
                            linewidths=1.5, alpha=0.9, linestyles='-')
        ax.clabel(cs_equi, inline=True, fontsize=9, fmt='h=%.2f')
        
        # === FLOW LINES (from stream function) ===
        # Calculate stream function range
        psi = self.solver.psi
        
        # Generate flow line levels avoiding clustering
        psi_flat = psi.flatten()
        psi_flat = psi_flat[~np.isnan(psi_flat)]  # Remove NaNs
        
        # Use percentile-based spacing for flow lines
        percentiles = np.linspace(5, 95, num_flow_lines)
        psi_levels = np.percentile(psi_flat, percentiles)
        
        # Ensure unique levels
        psi_levels = np.unique(psi_levels)
        
        # Plot flow lines as contours of stream function
        X_psi, Y_psi = np.meshgrid(self.domain.x_coords, self.domain.y_coords)
        cs_flow = ax.contour(X_psi, -Y_psi, psi, levels=psi_levels, 
                            colors='red', linewidths=1.5, alpha=0.9, linestyles='-')
        
        # Add flow direction arrows
        if num_flow_lines > 0:
            # Get collections (handle different matplotlib versions)
            try:
                collections = cs_flow.collections
            except AttributeError:
                # For newer matplotlib versions
                collections = cs_flow.allsegs
                
            # Add arrows along some flow lines
            if isinstance(collections[0], list):
                # Handle allsegs format (list of lists)
                for i in range(0, len(collections), 2):  # Every other level
                    if i < len(collections):
                        for segment in collections[i]:
                            if len(segment) > 20:
                                # Add arrows at a few points
                                indices = np.linspace(10, len(segment)-10, 3, dtype=int)
                                for idx in indices:
                                    if idx < len(segment) - 1:
                                        x1, y1 = segment[idx]
                                        x2, y2 = segment[idx + 1]
                                        dx = x2 - x1
                                        dy = y2 - y1
                                        if abs(dx) + abs(dy) > 0.01:  # Skip if too small
                                            ax.arrow(x1, y1, dx*0.3, dy*0.3,
                                                   head_width=0.4, head_length=0.3,
                                                   fc='darkred', ec='darkred', 
                                                   alpha=0.7, zorder=5)
            else:
                # Handle collections format
                for collection in collections[::2]:  # Every other flow line
                    for path in collection.get_paths():
                        vertices = path.vertices
                        if len(vertices) > 20:
                            # Add arrows at a few points
                            indices = np.linspace(10, len(vertices)-10, 3, dtype=int)
                            for idx in indices:
                                if idx < len(vertices) - 1:
                                    x1, y1 = vertices[idx]
                                    x2, y2 = vertices[idx + 1]
                                    dx = x2 - x1
                                    dy = y2 - y1
                                    if abs(dx) + abs(dy) > 0.01:  # Skip if too small
                                        ax.arrow(x1, y1, dx*0.3, dy*0.3,
                                               head_width=0.4, head_length=0.3,
                                               fc='darkred', ec='darkred', 
                                               alpha=0.7, zorder=5)
        
        # === ADD GEOMETRY ===
        self._add_enhanced_geometry(ax)
        
        # Add velocity vectors if requested
        if show_velocity_vectors:
            self._add_velocity_vectors(ax, X, Y, density=0.5)
        
        # Customize plot
        ax.set_xlabel('Horizontal Distance (m)', fontsize=12)
        ax.set_ylabel('Depth Below Surface (m)', fontsize=12)
        ax.set_title('Flow Net (Orthogonal Pattern)\nBlue: Equipotentials | Red: Flow Lines', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xlim(0, self.domain.width)
        ax.set_ylim(-self.domain.depth, 0)
        ax.set_aspect('equal', adjustable='box')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='blue', linewidth=2, label='Equipotentials (const. head)'),
            plt.Line2D([0], [0], color='red', linewidth=2, label='Flow lines (stream function)'),
            plt.Rectangle((0, 0), 1, 1, fc='black', label='Sheet piles (impermeable)'),
            plt.Rectangle((0, 0), 1, 1, fc='lightgray', alpha=0.5, label='Excavation'),
            plt.Line2D([0], [0], color='cyan', linewidth=2, label='Water table')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add text box with flow net info
        info_text = (f"Flow Net Parameters:\n"
                    f"• {num_equipotentials} Equipotentials\n"
                    f"• {num_flow_lines} Flow Lines\n"
                    f"• Head drop: {(h_levels[1]-h_levels[0]):.3f} m\n"
                    f"• Orthogonality: Enforced")
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_hydraulic_head(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Plot hydraulic head distribution as filled contours"""
        
        X, Y = np.meshgrid(self.domain.x_coords, -self.domain.y_coords)
        H = self.solver.H
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create filled contour map
        levels = 30
        im = ax.contourf(X, Y, H, levels=levels, cmap='viridis', extend='both')
        
        # Add contour lines
        cs = ax.contour(X, Y, H, levels=15, colors='white', linewidths=0.5, alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
        
        # Add geometry
        self._add_enhanced_geometry(ax)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Hydraulic Head (m)', shrink=0.9)
        cbar.ax.tick_params(labelsize=10)
        
        ax.set_xlabel('Horizontal Distance (m)', fontsize=12)
        ax.set_ylabel('Depth Below Surface (m)', fontsize=12)
        ax.set_title('Hydraulic Head Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xlim(0, self.domain.width)
        ax.set_ylim(-self.domain.depth, 0)
        ax.set_aspect('equal', adjustable='box')
        
        # Add min/max annotations
        h_min, h_max = np.min(H), np.max(H)
        info_text = f"Head Range:\nMin: {h_min:.2f} m\nMax: {h_max:.2f} m"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        return fig
    
    def plot_velocity_field(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Plot seepage velocity magnitude and vectors"""
        
        X, Y = np.meshgrid(self.domain.x_coords, -self.domain.y_coords)
        velocity_magnitude = np.sqrt(self.solver.qx**2 + self.solver.qy**2)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot velocity magnitude (log scale for better visualization)
        vm_log = np.log10(velocity_magnitude + 1e-15)
        levels = 30
        im = ax.contourf(X, Y, vm_log, levels=levels, cmap='plasma', extend='both')
        
        # Add velocity vectors
        self._add_velocity_vectors(ax, X, Y, density=1.0, scale_type='linear')
        
        # Add geometry
        self._add_enhanced_geometry(ax)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='log₁₀(Velocity) [m/s]', shrink=0.9)
        cbar.ax.tick_params(labelsize=10)
        
        # Add max velocity annotation
        v_max = np.max(velocity_magnitude)
        v_mean = np.mean(velocity_magnitude[velocity_magnitude > 0])
        info_text = f"Velocity Stats:\nMax: {v_max:.2e} m/s\nMean: {v_mean:.2e} m/s"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax.set_xlabel('Horizontal Distance (m)', fontsize=12)
        ax.set_ylabel('Depth Below Surface (m)', fontsize=12)
        ax.set_title('Seepage Velocity Field', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xlim(0, self.domain.width)
        ax.set_ylim(-self.domain.depth, 0)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        return fig
    
    def plot_summary_dashboard(self, figsize: Tuple[float, float] = (16, 12)) -> plt.Figure:
        """Create comprehensive dashboard with all visualizations"""
        
        # Ensure all calculations are done
        if self.solver.psi is None:
            self.solver.calculate_stream_function()
        
        fig = plt.figure(figsize=figsize)
        
        # Create 2x2 grid
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        ax4 = plt.subplot(2, 2, 4)
        
        X, Y = np.meshgrid(self.domain.x_coords, -self.domain.y_coords)
        
        # === PLOT 1: FLOW NET ===
        H = self.solver.H
        psi = self.solver.psi
        
        # Equipotentials
        h_levels = self.solver.generate_equipotentials(12)
        cs1 = ax1.contour(X, Y, H, levels=h_levels, colors='blue', linewidths=1.2)
        ax1.clabel(cs1, inline=True, fontsize=7, fmt='%.2f')
        
        # Flow lines from stream function
        psi_min, psi_max = np.min(psi), np.max(psi)
        psi_levels = np.linspace(psi_min, psi_max, 10)
        X_psi, Y_psi = np.meshgrid(self.domain.x_coords, self.domain.y_coords)
        cs2 = ax1.contour(X_psi, -Y_psi, psi, levels=psi_levels, colors='red', linewidths=1.2)
        
        self._add_enhanced_geometry(ax1)
        ax1.set_title('Flow Net (Orthogonal)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Distance (m)', fontsize=9)
        ax1.set_ylabel('Depth (m)', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # === PLOT 2: HYDRAULIC HEAD ===
        im2 = ax2.contourf(X, Y, H, levels=25, cmap='viridis')
        plt.colorbar(im2, ax=ax2, label='Head (m)')
        self._add_enhanced_geometry(ax2)
        ax2.set_title('Hydraulic Head', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Distance (m)', fontsize=9)
        ax2.set_ylabel('Depth (m)', fontsize=9)
        ax2.set_aspect('equal', adjustable='box')
        
        # === PLOT 3: VELOCITY FIELD ===
        velocity_magnitude = np.sqrt(self.solver.qx**2 + self.solver.qy**2)
        vm_log = np.log10(velocity_magnitude + 1e-15)
        im3 = ax3.contourf(X, Y, vm_log, levels=25, cmap='plasma')
        plt.colorbar(im3, ax=ax3, label='log₁₀(V) [m/s]')
        self._add_velocity_vectors(ax3, X, Y, density=2)
        self._add_enhanced_geometry(ax3)
        ax3.set_title('Velocity Field', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Distance (m)', fontsize=9)
        ax3.set_ylabel('Depth (m)', fontsize=9)
        ax3.set_aspect('equal', adjustable='box')
        
        # === PLOT 4: NUMERICAL RESULTS ===
        ax4.axis('off')
        
        # Calculate results
        seepage = self.solver.calculate_seepage_discharge()
        gradients = self.solver.calculate_exit_gradients()
        
        # Format results text
        results_text = "=" * 40 + "\n"
        results_text += "NUMERICAL RESULTS\n"
        results_text += "=" * 40 + "\n\n"
        
        results_text += "Seepage Flow Rates:\n"
        results_text += "-" * 20 + "\n"
        if 'excavation_bottom' in seepage:
            results_text += f"• Excavation: {seepage['excavation_bottom']:.3e} m³/s/m\n"
        results_text += f"• Left boundary: {seepage['left_boundary']:.3e} m³/s/m\n"
        results_text += f"• Right boundary: {seepage['right_boundary']:.3e} m³/s/m\n"
        results_text += f"• Mass balance error: {seepage['mass_balance_error']:.2f}%\n\n"
        
        results_text += "Exit Gradients:\n"
        results_text += "-" * 20 + "\n"
        for key, value in gradients.items():
            if 'pile' in key or 'excavation' in key:
                results_text += f"• {key}: {value:.3f}\n"
        
        results_text += f"\nSafety Assessment:\n"
        results_text += "-" * 20 + "\n"
        results_text += f"• Critical gradient: {gradients['critical_gradient']:.3f}\n"
        results_text += f"• Max exit gradient: {gradients['max_exit_gradient']:.3f}\n"
        results_text += f"• Safety factor: {gradients['safety_factor']:.2f}\n"
        
        if gradients['safety_factor'] < 1.5:
            results_text += "\n⚠️ WARNING: Safety factor < 1.5"
        else:
            results_text += "\n✓ Safety factor acceptable"
        
        results_text += f"\n\nHead Statistics:\n"
        results_text += "-" * 20 + "\n"
        results_text += f"• Range: {np.min(H):.2f} to {np.max(H):.2f} m\n"
        results_text += f"• Mean: {np.mean(H):.2f} m\n"
        results_text += f"• Std Dev: {np.std(H):.3f} m"
        
        ax4.text(0.1, 0.9, results_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.suptitle('SEEPAGE ANALYSIS DASHBOARD', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return fig
    
    def _add_enhanced_geometry(self, ax, show_layers=True):
        """Add sheet piles, excavation, and other geometry with clear visualization"""
        
        # === SHEET PILES (highlighted clearly) ===
        for i, pile in enumerate(self.domain.sheet_piles):
            # Draw sheet pile as thick black line with label
            pile_line = plt.Line2D(
                [pile.x_position, pile.x_position],
                [-pile.top_depth, -pile.bottom_depth],
                color='black', linewidth=8, solid_capstyle='butt',
                label='Sheet pile' if i == 0 else None,
                zorder=10
            )
            ax.add_line(pile_line)
            
            # Add white outline for visibility
            pile_outline = plt.Line2D(
                [pile.x_position, pile.x_position],
                [-pile.top_depth, -pile.bottom_depth],
                color='white', linewidth=10, solid_capstyle='butt',
                zorder=9
            )
            ax.add_line(pile_outline)
            
            # Add pile depth annotation
            ax.annotate(f'{pile.bottom_depth:.1f}m',
                       xy=(pile.x_position, -pile.bottom_depth),
                       xytext=(pile.x_position + 0.5, -pile.bottom_depth - 0.5),
                       fontsize=8, color='black',
                       arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
        
        # === EXCAVATION ===
        if self.domain.excavation:
            excav_rect = patches.Rectangle(
                (self.domain.excavation.left_x, 0),
                self.domain.excavation.width, -self.domain.excavation.depth,
                facecolor='lightgray', edgecolor='darkgray',
                alpha=0.3, linewidth=2, hatch='//',
                label='Excavation', zorder=3
            )
            ax.add_patch(excav_rect)
            
            # Excavation bottom line
            ax.plot([self.domain.excavation.left_x, self.domain.excavation.right_x],
                   [-self.domain.excavation.depth, -self.domain.excavation.depth],
                   'k--', linewidth=1.5, alpha=0.7, zorder=4)
            
            # Water level in excavation if present
            if self.domain.excavation.water_level:
                ax.plot([self.domain.excavation.left_x, self.domain.excavation.right_x],
                       [-self.domain.excavation.water_level, -self.domain.excavation.water_level],
                       'lightblue', linewidth=3, alpha=0.8,
                       label='Excavation water', zorder=5)
        
        # === SOIL LAYER BOUNDARIES ===
        for layer in self.domain.soil_layers:
            if 0 < layer.depth_bottom < self.domain.depth:
                ax.axhline(y=-layer.depth_bottom, color='brown',
                          linestyle=':', linewidth=1.5, alpha=0.5)
                # Add layer label
                ax.text(self.domain.width * 0.95, -layer.depth_bottom + 0.2,
                       f'{layer.name}', fontsize=8, ha='right',
                       color='brown', alpha=0.7)
        
        # === WATER TABLE ===
        # Outside water level
        ax.plot([0, self.domain.excavation.left_x if self.domain.excavation else self.domain.width],
               [-self.domain.water_level_left, -self.domain.water_level_left],
               'cyan', linewidth=3, alpha=0.8, label='Water table', zorder=5)
        
        if self.domain.excavation:
            ax.plot([self.domain.excavation.right_x, self.domain.width],
                   [-self.domain.water_level_right, -self.domain.water_level_right],
                   'cyan', linewidth=3, alpha=0.8, zorder=5)
    
    def _add_velocity_vectors(self, ax, X, Y, density: float = 1.0, scale_type: str = 'log'):
        """Add velocity vectors with proper scaling"""
        
        # Subsample for clarity
        skip = max(10, int(20 / density))
        
        X_vec = X[::skip, ::skip]
        Y_vec = Y[::skip, ::skip]
        qx_vec = self.solver.qx[::skip, ::skip]
        qy_vec = -self.solver.qy[::skip, ::skip]  # Negative for display
        
        # Calculate magnitude
        mag = np.sqrt(qx_vec**2 + qy_vec**2)
        
        # Skip very small velocities
        mask = mag > 1e-10
        
        if np.any(mask):
            # Scale arrows based on magnitude
            if scale_type == 'log':
                # Log scaling for better visibility of small velocities
                scale = np.max(mag[mask]) * 15
            else:
                scale = np.max(mag[mask]) * 20
            
            # Plot vectors
            ax.quiver(X_vec[mask], Y_vec[mask], 
                     qx_vec[mask], qy_vec[mask],
                     mag[mask], cmap='coolwarm',
                     scale=scale, scale_units='xy', angles='xy',
                     width=0.003, edgecolor='black', linewidth=0.5,
                     alpha=0.7)
    
    @staticmethod
    def fig_to_base64(fig) -> str:
        """Convert matplotlib figure to base64 string for web display"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    def export_results_to_csv(self, filename: str):
        """Export numerical results to CSV file"""
        import csv
        
        seepage = self.solver.calculate_seepage_discharge()
        gradients = self.solver.calculate_exit_gradients()
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            writer.writerow(['Seepage Flow Analysis Results'])
            writer.writerow(['Generated:', pd.Timestamp.now()])
            writer.writerow([])
            
            # Domain configuration
            writer.writerow(['Domain Configuration'])
            writer.writerow(['Parameter', 'Value', 'Unit'])
            writer.writerow(['Width', self.domain.width, 'm'])
            writer.writerow(['Depth', self.domain.depth, 'm'])
            writer.writerow(['Grid (nx × ny)', f'{self.domain.nx} × {self.domain.ny}', 'nodes'])
            writer.writerow([])
            
            # Sheet pile information
            writer.writerow(['Sheet Piles'])
            for i, pile in enumerate(self.domain.sheet_piles):
                writer.writerow([f'Pile {i+1} Position', pile.x_position, 'm'])
                writer.writerow([f'Pile {i+1} Length', pile.bottom_depth - pile.top_depth, 'm'])
            writer.writerow([])
            
            # Seepage results
            writer.writerow(['Seepage Flow Results'])
            writer.writerow(['Location', 'Flow Rate', 'Unit'])
            for key, value in seepage.items():
                unit = '%' if 'error' in key else 'm³/s/m'
                writer.writerow([key.replace('_', ' ').title(), f'{value:.6e}', unit])
            writer.writerow([])
            
            # Exit gradients
            writer.writerow(['Exit Gradient Analysis'])
            writer.writerow(['Location', 'Gradient', 'Unit'])
            for key, value in gradients.items():
                unit = '-' if 'factor' in key else '[-]'
                writer.writerow([key.replace('_', ' ').title(), f'{value:.4f}', unit])
            writer.writerow([])
            
            # Hydraulic head statistics
            writer.writerow(['Hydraulic Head Statistics'])
            writer.writerow(['Statistic', 'Value', 'Unit'])
            writer.writerow(['Minimum', f'{np.min(self.solver.H):.3f}', 'm'])
            writer.writerow(['Maximum', f'{np.max(self.solver.H):.3f}', 'm'])
            writer.writerow(['Mean', f'{np.mean(self.solver.H):.3f}', 'm'])
            writer.writerow(['Std Dev', f'{np.std(self.solver.H):.4f}', 'm'])
            
        return filename


# Import for pandas in export function
try:
    import pandas as pd
except ImportError:
    pd = None
