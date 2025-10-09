#!/usr/bin/env python
"""
Quick test script to verify sectionproperties is working
"""

import sys

print("Testing sectionproperties library...")
print("-" * 40)

try:
    # Test basic import
    import sectionproperties
    print(f"✓ sectionproperties version: {sectionproperties.__version__}")
    
    # Test geometry import
    from sectionproperties.pre.library import rectangular_section
    print("✓ Geometry library imports successful")
    
    # Test material import
    from sectionproperties.pre import Material
    print("✓ Material class import successful")
    
    # Test section import
    from sectionproperties.analysis import Section
    print("✓ Section class import successful")
    
    # Create a simple test section
    print("\nCreating test section...")
    
    # Create material
    steel = Material(
        name="Steel",
        elastic_modulus=200000,
        poissons_ratio=0.3,
        yield_strength=250,
        density=7.85e-9,
        color="grey"
    )
    print("✓ Material created")
    
    # Create a simple rectangular section
    geom = rectangular_section(d=100, b=50, material=steel)
    print("✓ Geometry created")
    
    # Create mesh
    geom.create_mesh(mesh_sizes=[10])
    print("✓ Mesh generated")
    
    # Create section and analyze
    section = Section(geometry=geom)
    print("✓ Section object created")
    
    section.calculate_geometric_properties()
    print("✓ Geometric properties calculated")
    
    # Get some properties
    area = section.get_area()
    ixx = section.get_ic()[0]
    
    print(f"\nTest Results:")
    print(f"  Area: {area:.2f} mm²")
    print(f"  Ixx:  {ixx:.2e} mm⁴")
    
    print("\n" + "=" * 40)
    print("✓ All tests passed! sectionproperties is working correctly.")
    print("=" * 40)
    
except ImportError as e:
    print(f"\n✗ Import Error: {e}")
    print("\nPlease install sectionproperties:")
    print("  pip install sectionproperties")
    sys.exit(1)
    
except Exception as e:
    print(f"\n✗ Error during testing: {e}")
    print("\nThere may be an issue with the sectionproperties installation.")
    print("Try reinstalling:")
    print("  pip uninstall sectionproperties")
    print("  pip install sectionproperties")
    sys.exit(1)
