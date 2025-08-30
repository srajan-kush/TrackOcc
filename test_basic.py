import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

def test_imports():
    """Test if we can import the main modules"""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import models
        print("✓ models imported")
        
        import loaders
        print("✓ loaders imported")
        
        # Test specific components
        from models.detectors.trackocc import TrackOcc
        print("✓ TrackOcc imported")
        
        from loaders.waymo_occ_dataset import OccWaymoDataset
        print("✓ Dataset imported")
        
        print("\n✓ All imports successful!")
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_imports()