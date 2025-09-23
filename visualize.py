"""
visualize.py - Plotting helpers for flow net visualization
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
    """Class for creating flow net visualizations"""
    
    def __init__(self, domain: Domain, solver: GroundwaterSolver):
        self.domain = domain
        self.solver = solver
        
    def plot_flow_net(self, 
                      num_equipotentials: int = 15,
                      num_streamlines: int = 10,
                      show_mesh: bool = False,
                      show_velocity_vectors: bool = True,
                      figsize: Tuple[float, float] = (14, 10)) -> plt.Figure:
        """Create comprehensive flow net visualization with physically correct patterns"""
        
        # Prepare data
        X, Y = np.meshgrid(self.domain.x_coords, -self.domain.y_coords)  # Negative for depth
        H = self.solver.H
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        # Plot hydraulic head contours (equipotentials)
        # Generate proper equipotential levels
        if hasattr(self.solver, 'generate_equipotentials'):
            levels = self.solver.generate_equipotentials(num_equipotentials)
        else:
            h_min, h_max = np.min(H), np.max(H)
            levels = np.linspace(h_min, h_max, num_equipotentials)
        
        # Draw equipotentials with better styling
        cs = ax.contour(X, Y, H, levels=levels, colors='blue', 
                       linewidths=1.5, alpha=0.8)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
        
        # Plot streamlines with improved generation
        streamlines = self.solver.generate_streamlines(num_streamlines)
        
        # Use varied colors for better visualization
        stream_colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(streamlines)))
        
        for idx, (stream_x, stream_y) in enumerate(streamlines):
            ax.plot(stream_x, -stream_y, color=stream_colors[idx], 
                   linewidth=1.5, alpha=0.7)
        
        # Add flow direction arrows on streamlines
        for stream_x, stream_y in streamlines[::2]:  # Every other streamline for clarity
            if len(stream_x) > 20:
                # Add arrows at several points along the streamline
                arrow_indices = np.linspace(10, len(stream_x)-10, 3, dtype=int)
                for i in arrow_indices:
                    if i < len(stream_x) - 1:
                        dx = stream_x[i+1] - stream_x[i]
                        dy = -(stream_y[i+1] - stream_y[i])
                        ax.arrow(stream_x[i], -stream_y[i], dx*0.5, dy*0.5,
                                head_width=0.3, head_length=0.2, 
                                fc='red', ec='red', alpha=0.6)
        
        # Plot velocity vectors if requested
        if show_velocity_vectors:
            self._add_velocity_vectors(ax, X, Y)
        
        # Add geometry elements
        self._add_geometry_to_plot(ax)
        
        # Customize plot
        ax.set_xlabel('Horizontal Distance (m)', fontsize=12)
        ax.set_ylabel('Depth Below Surface (m)', fontsize=12)
        ax.set_title('Flow Net Analysis\nBlue: Equipotentials | Red: Flow Lines with Direction', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, self.domain.width)
        ax.set_ylim(-self.domain.depth, 0)
        
        # Add legend with better elements
        legend_elements = [
            plt.Line2D([0], [0], color='blue', label='Equipotentials (constant head)'),
            plt.Line2D([0], [0], color='red', label='Flow lines (flow direction)'),
            plt.Rectangle((0, 0), 1, 1, fc='black', label='Sheet piles'),
            plt.Rectangle((0, 0), 1, 1, fc='lightgray', alpha=0.5, label='Excavation')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_hydraulic_head(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Plot hydraulic head distribution as a color map"""
        
        X, Y = np.meshgrid(self.domain.x_coords, -self.domain.y_coords)
        H = self.solver.H
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create color map
        im = ax.contourf(X, Y, H, levels=30, cmap='viridis', extend='both')
        
        # Add contour lines
        cs = ax.contour(X, Y, H, levels=15, colors='white', linewidths=0.5, alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
        
        # Add geometry
        self._add_geometry_to_plot(ax)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Hydraulic Head (m)', shrink=0.8)
        
        ax.set_xlabel('Horizontal Distance (m)', fontsize=12)
        ax.set_ylabel('Depth Below Surface (m)', fontsize=12)
        ax.set_title('Hydraulic Head Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, self.domain.width)
        ax.set_ylim(-self.domain.depth, 0)
        
        plt.tight_layout()
        return fig
    
    def plot_velocity_field(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Plot seepage velocity field"""
        
        X, Y = np.meshgrid(self.domain.x_coords, -self.domain.y_coords)
        velocity_magnitude = np.sqrt(self.solver.qx**2 + self.solver.qy**2)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot velocity magnitude
        # Use log scale for better visualization
        vm_log = np.log10(velocity_magnitude + 1e-15)
        im = ax.contourf(X, Y, vm_log, levels=30, cmap='plasma')
        
        # Add velocity vectors
        self._add_velocity_vectors(ax, X, Y, density=1)
        
        # Add geometry
        self._add_geometry_to_plot(ax)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='log₁₀(Velocity) [m/s]', shrink=0.8)
        
        ax.set_xlabel('Horizontal Distance (m)', fontsize=12)
        ax.set_ylabel('Depth Below Surface (m)', fontsize=12)
        ax.set_title('Seepage Velocity Field', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, self.domain.width)
        ax.set_ylim(-self.domain.depth, 0)
        
        plt.tight_layout()
        return fig
    
    def plot_flow_channels(self, 
                          num_channels: int = 8,
                          highlight_critical: bool = True,
                          figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Plot flow channels with highlighting of critical zones"""
        
        X, Y = np.meshgrid(self.domain.x_coords, -self.domain.y_coords)
        H = self.solver.H
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Background: velocity magnitude
        velocity_magnitude = np.sqrt(self.solver.qx**2 + self.solver.qy**2)
        im = ax.contourf(X, Y, velocity_magnitude, levels=20, cmap='YlOrRd', alpha=0.6)
        
        # Generate and plot flow channels
        streamlines = self.solver.generate_streamlines(num_channels)
        
        # Color channels based on their position
        channel_colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(streamlines)))
        
        for idx, (stream_x, stream_y) in enumerate(streamlines):
            # Create filled channel regions
            if idx < len(streamlines) - 1:
                next_x, next_y = streamlines[idx + 1]
                
                # Create polygon for channel
                if len(stream_x) == len(next_x):
                    vertices = list(zip(stream_x, -stream_y)) + \
                              list(zip(next_x[::-1], -next_y[::-1]))
                    channel_patch = patches.Polygon(vertices, 
                                                  facecolor=channel_colors[idx],
                                                  edgecolor='black',
                                                  alpha=0.4,
                                                  linewidth=0.5)
                    ax.add_patch(channel_patch)
            
            # Plot channel boundaries
            ax.plot(stream_x, -stream_y, 'b-', linewidth=1.5, alpha=0.8)
        
        # Highlight critical zones if requested
        if highlight_critical:
            self._highlight_critical_zones(ax)
        
        # Add equipotentials
        cs = ax.contour(X, Y, H, levels=10, colors='gray', linewidths=0.5, alpha=0.5)
        
        # Add geometry
        self._add_geometry_to_plot(ax)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Velocity Magnitude (m/s)', shrink=0.8)
        
        ax.set_xlabel('Horizontal Distance (m)', fontsize=12)
        ax.set_ylabel('Depth Below Surface (m)', fontsize=12)
        ax.set_title('Flow Channels Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, self.domain.width)
        ax.set_ylim(-self.domain.depth, 0)
        
        plt.tight_layout()
        return fig
    
    def plot_summary_dashboard(self, figsize: Tuple[float, float] = (16, 12)) -> plt.Figure:
        """Create a comprehensive dashboard with multiple visualizations"""
        
        fig = plt.figure(figsize=figsize)
        
        # Create 2x2 grid of subplots
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        ax4 = plt.subplot(2, 2, 4)
        
        X, Y = np.meshgrid(self.domain.x_coords, -self.domain.y_coords)
        
        # Plot 1: Flow Net
        H = self.solver.H
        h_min, h_max = np.min(H), np.max(H)
        levels = np.linspace(h_min, h_max, 12)
        
        cs1 = ax1.contour(X, Y, H, levels=levels, colors='blue', linewidths=1.2)
        ax1.clabel(cs1, inline=True, fontsize=7, fmt='%.2f')
        
        streamlines = self.solver.generate_streamlines(8)
        for stream_x, stream_y in streamlines:
            ax1.plot(stream_x, -stream_y, 'r-', linewidth=1.2, alpha=0.7)
        
        self._add_geometry_to_plot(ax1)
        ax1.set_title('Flow Net', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Distance (m)', fontsize=10)
        ax1.set_ylabel('Depth (m)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Hydraulic Head
        im2 = ax2.contourf(X, Y, H, levels=20, cmap='viridis')
        plt.colorbar(im2, ax=ax2, label='Head (m)')
        self._add_geometry_to_plot(ax2)
        ax2.set_title('Hydraulic Head', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Distance (m)', fontsize=10)
        ax2.set_ylabel('Depth (m)', fontsize=10)
        
        # Plot 3: Velocity Magnitude
        velocity_magnitude = np.sqrt(self.solver.qx**2 + self.solver.qy**2)
        vm_log = np.log10(velocity_magnitude + 1e-15)
        im3 = ax3.contourf(X, Y, vm_log, levels=20, cmap='plasma')
        plt.colorbar(im3, ax=ax3, label='log₁₀(V) [m/s]')
        self._add_velocity_vectors(ax3, X, Y, density=2)
        self._add_geometry_to_plot(ax3)
        ax3.set_title('Velocity Field', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Distance (m)', fontsize=10)
        ax3.set_ylabel('Depth (m)', fontsize=10)
        
        # Plot 4: Numerical Results Summary
        ax4.axis('off')
        
        # Calculate results
        seepage = self.solver.calculate_seepage_discharge()
        gradients = self.solver.calculate_exit_gradients()
        
        # Create text summary
        results_text = "NUMERICAL RESULTS\n" + "="*30 + "\n\n"
        results_text += "Seepage Flow:\n"
        
        if 'excavation_bottom' in seepage:
            results_text += f"  Through excavation: {seepage['excavation_bottom']:.2e} m³/s/m\n"
        results_text += f"  Left boundary: {seepage['left_boundary']:.2e} m³/s/m\n"
        results_text += f"  Right boundary: {seepage['right_boundary']:.2e} m³/s/m\n"
        results_text += f"  Mass balance error: {seepage['mass_balance_error']:.2f}%\n\n"
        
        results_text += "Exit Gradients:\n"
        for key, value in gradients.items():
            if 'gradient' in key or 'safety' in key:
                results_text += f"  {key.replace('_', ' ').title()}: {value:.3f}\n"
        
        results_text += f"\nHead Range: {h_min:.2f} to {h_max:.2f} m"
        results_text += f"\nMax Velocity: {np.max(velocity_magnitude):.2e} m/s"
        
        ax4.text(0.1, 0.9, results_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.suptitle('FLOW NET ANALYSIS DASHBOARD', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return fig
    
    def _add_geometry_to_plot(self, ax):
        """Add sheet piles, excavation, and soil layers to plot"""
        
        # Sheet piles
        for pile in self.domain.sheet_piles:
            pile_rect = patches.Rectangle(
                (pile.x_position - pile.thickness/2, -pile.bottom_depth),
                pile.thickness, pile.length,
                facecolor='black', edgecolor='black', linewidth=2, zorder=5
            )
            ax.add_patch(pile_rect)
        
        # Excavation
        if self.domain.excavation:
            excav_rect = patches.Rectangle(
                (self.domain.excavation.left_x, 0),
                self.domain.excavation.width, -self.domain.excavation.depth,
                facecolor='lightgray', edgecolor='darkgray',
                alpha=0.3, linewidth=2, hatch='//', zorder=2
            )
            ax.add_patch(excav_rect)
            
            # Water level in excavation
            if self.domain.excavation.water_level:
                ax.axhline(y=-self.domain.excavation.water_level,
                          xmin=self.domain.excavation.left_x/self.domain.width,
                          xmax=self.domain.excavation.right_x/self.domain.width,
                          color='lightblue', linestyle='--', linewidth=2,
                          label='Excavation water level')
        
        # Soil layer boundaries
        for layer in self.domain.soil_layers:
            if layer.depth_bottom < self.domain.depth:
                ax.axhline(y=-layer.depth_bottom, color='brown', 
                          linestyle=':', linewidth=1.5, alpha=0.5,
                          label=f'{layer.name} boundary')
        
        # Water table outside
        ax.axhline(y=-self.domain.water_level_left, color='cyan',
                  linestyle='-', linewidth=2, alpha=0.7,
                  label='Water table')
    
    def _add_velocity_vectors(self, ax, X, Y, density: int = 1):
        """Add velocity vectors to plot"""
        
        # Subsample for clarity
        skip = max(5, int(15 / density))
        
        X_vec = X[::skip, ::skip]
        Y_vec = Y[::skip, ::skip]
        qx_vec = self.solver.qx[::skip, ::skip]
        qy_vec = -self.solver.qy[::skip, ::skip]  # Negative for plotting
        
        # Scale vectors
        scale = np.max(np.sqrt(qx_vec**2 + qy_vec**2)) * 20
        
        ax.quiver(X_vec, Y_vec, qx_vec, qy_vec,
                 scale=scale, scale_units='xy', angles='xy',
                 color='white', alpha=0.7, width=0.003,
                 edgecolor='black', linewidth=0.5)
    
    def _highlight_critical_zones(self, ax):
        """Highlight zones with high exit gradients"""
        
        gradients = self.solver.calculate_exit_gradients()
        
        # Highlight sheet pile toes
        for pile in self.domain.sheet_piles:
            circle = patches.Circle(
                (pile.x_position, -pile.bottom_depth),
                radius=0.5, color='red', alpha=0.3,
                label='Critical zone'
            )
            ax.add_patch(circle)
        
        # Highlight excavation bottom center
        if self.domain.excavation:
            center_x = (self.domain.excavation.left_x + self.domain.excavation.right_x) / 2
            circle = patches.Circle(
                (center_x, -self.domain.excavation.depth),
                radius=0.5, color='orange', alpha=0.3
            )
            ax.add_patch(circle)
    
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
            
            # Write header
            writer.writerow(['Flow Net Analysis Results'])
            writer.writerow([])
            
            # Domain info
            writer.writerow(['Domain Information'])
            writer.writerow(['Width (m)', self.domain.width])
            writer.writerow(['Depth (m)', self.domain.depth])
            writer.writerow(['Grid points', f'{self.domain.nx} x {self.domain.ny}'])
            writer.writerow([])
            
            # Seepage results
            writer.writerow(['Seepage Flow Results'])
            for key, value in seepage.items():
                writer.writerow([key.replace('_', ' ').title(), f'{value:.6e}'])
            writer.writerow([])
            
            # Gradient results
            writer.writerow(['Exit Gradient Results'])
            for key, value in gradients.items():
                writer.writerow([key.replace('_', ' ').title(), f'{value:.6f}'])
            writer.writerow([])
            
            # Head statistics
            writer.writerow(['Hydraulic Head Statistics'])
            writer.writerow(['Minimum (m)', f'{np.min(self.solver.H):.3f}'])
            writer.writerow(['Maximum (m)', f'{np.max(self.solver.H):.3f}'])
            writer.writerow(['Mean (m)', f'{np.mean(self.solver.H):.3f}'])
            
        return filename
