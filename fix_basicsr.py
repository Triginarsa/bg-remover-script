"""
Compatibility fix for basicsr with newer torchvision versions.
This patches the import issue with torchvision.transforms.functional_tensor
"""
import sys
import torchvision.transforms.functional as F

# Create a mock module for the old import path
class FunctionalTensorModule:
    @staticmethod
    def rgb_to_grayscale(img, num_output_channels=1):
        return F.rgb_to_grayscale(img, num_output_channels)

# Inject the compatibility module
sys.modules['torchvision.transforms.functional_tensor'] = FunctionalTensorModule()
