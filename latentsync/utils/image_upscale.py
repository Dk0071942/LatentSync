import os
import torch
import numpy as np
import torchvision
from PIL import Image

# Import RealESRGANer and RRDBNet architecture for RealESRGAN_x4plus.
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

class ImageUpscale:
    def __init__(self, model_name: str = "RealESRGAN_x4plus", tile: int = 0, tile_pad: int = 10, half: bool = False):
        """
        Initializes the RealESRGAN based upscaling model.
        
        Args:
            model_name (str): The model name; corresponds to the weight file (without extension).
            tile (int): Tile size for tiled inference (set to 0 to disable tiling).
            tile_pad (int): Padding for tiling.
            half (bool): Whether to use FP16 mode for performance.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.netscale = 4  # x4 upscaling factor for RealESRGAN
        self.model_name = model_name
        model_filename = f"{model_name}.pth"
        
        # Define the model architecture matching the defaults of RealESRGAN_x4plus
        self.model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                  num_block=23, num_grow_ch=32, scale=self.netscale)
        
        # Construct model path relative to this script.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, "upscale_models")
        self.model_path = os.path.join(model_dir, model_filename)
        print(f"Using model path: {self.model_path}")
        
        try:
            self.upsampler = RealESRGANer(
                scale=self.netscale,
                model_path=self.model_path,
                model=self.model_arch,
                tile=tile,
                tile_pad=tile_pad,
                pre_pad=0,
                half=half,
                device=self.device
            )
            print("RealESRGANer initialized successfully.")
        except Exception as e:
            print("Error initializing RealESRGANer:", e)
            print("Please ensure the model file exists and is compatible.")
            self.upsampler = None

    def upscale(self, input_image_path: str, outscale: int = None) -> Image.Image:
        """
        Upscales an input image (PIL-based) using the RealESRGANer pipeline.
        
        Args:
            input_image_path (str): Path to the input image.
            outscale (int, optional): The scaling factor for the output image.
            
        Returns:
            PIL.Image: The upscaled image.
        """
        if self.upsampler is None:
            print("Upsampler not initialized.")
            return None

        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, input_image_path)
        print(f"Loading image from: {image_path}")
        try:
            img_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            print("Error loading image:", e)
            return None

        # Convert from PIL (RGB) to NumPy array (BGR for RealESRGAN)
        img_np = np.array(img_pil)
        img_np_bgr = img_np[:, :, ::-1]
        
        if outscale is None:
            outscale = self.netscale
        try:
            output_np_bgr, _ = self.upsampler.enhance(img_np_bgr, outscale=outscale)
            output_np_rgb = output_np_bgr[:, :, ::-1]
            sr_image = Image.fromarray(output_np_rgb)
            print("Image enhancement successful.")
            return sr_image
        except Exception as e:
            print("An error occurred during enhancement:", e)
            return None

    def enhance_image(self, decoded_latents: torch.Tensor, sharpness_factor: float = 1.5) -> torch.Tensor:
        """
        Enhances the image quality using a sharpening filter.
        
        Expects an input tensor of shape (N, C, H, W) in the range [-1, 1] and returns a tensor of
        the same shape after enhancement.
        
        Args:
            decoded_latents (torch.Tensor): Input tensor with values in [-1, 1].
            sharpness_factor (float): Factor for adjusting image sharpness.
        
        Returns:
            torch.Tensor: Enhanced tensor with values in [-1, 1].
        """
        # Convert input from [-1, 1] to [0, 1]
        enhanced_latents = (decoded_latents / 2 + 0.5).clamp(0, 1)
        # Apply the sharpening filter using torchvision's adjust_sharpness.
        enhanced_latents = torchvision.transforms.functional.adjust_sharpness(enhanced_latents, sharpness_factor)
        # Convert back to [-1, 1]
        enhanced_latents = enhanced_latents * 2 - 1
        return enhanced_latents

    def enhance_and_upscale(self, input_tensor: torch.Tensor, sharpness_factor: float = 1.5) -> torch.Tensor:
        """
        Accepts an input image tensor, applies a sharpening filter and upscales it using RealESRGAN.
        The upscaled image is then resized back to the original spatial dimensions so that the output
        tensor has the same shape as the input.
        
        Assumptions:
          - The input tensor is of shape (N, C, H, W) with values in [-1, 1].
          - This function processes each image in the batch individually.
        
        Args:
            input_tensor (torch.Tensor): Input image tensor with shape (N, C, H, W) and range [-1, 1].
            sharpness_factor (float): The factor for the sharpening filter.
        
        Returns:
            torch.Tensor: The enhanced and upscaled tensor with the same shape as the input, on the same device.
        """
        if self.upsampler is None:
            print("Upsampler not initialized.")
            return None

        # Ensure the input tensor is on the proper device.
        input_tensor = input_tensor.to(self.device)
        # First, apply the sharpening enhancement.
        enhanced_tensor = self.enhance_image(input_tensor, sharpness_factor=sharpness_factor)
        N, C, H, W = enhanced_tensor.shape
        output_tensors = []

        for i in range(N):
            # Extract one image: shape (C, H, W), range [-1, 1]
            img_tensor = enhanced_tensor[i]
            # Convert to [0, 1] range.
            img_tensor_0_1 = (img_tensor + 1) / 2
            # Convert tensor (C, H, W) to PIL image.
            pil_image = torchvision.transforms.functional.to_pil_image(img_tensor_0_1.cpu())
            # Convert PIL image to numpy array (RGB) and then to BGR (RealESRGAN expects BGR).
            np_image = np.array(pil_image)
            np_image_bgr = np_image[:, :, ::-1]
            
            try:
                # Upscale using RealESRGANer. The output will be upscaled by self.netscale.
                upscaled_np_bgr, _ = self.upsampler.enhance(np_image_bgr, outscale=self.netscale)
            except Exception as e:
                print("Error during RealESRGAN upscaling:", e)
                return None
            
            # Convert the upscaled image from BGR to RGB.
            upscaled_np_rgb = upscaled_np_bgr[:, :, ::-1]
            upscaled_pil = Image.fromarray(upscaled_np_rgb)
            # Resize the upscaled image back to the original dimensions.
            downscaled_pil = upscaled_pil.resize((W, H), Image.BICUBIC)
            # Convert downscaled image to tensor; resulting tensor is in range [0, 1].
            downscaled_tensor = torchvision.transforms.functional.to_tensor(downscaled_pil)
            # Convert to [-1, 1] range.
            final_tensor = downscaled_tensor * 2 - 1
            # Add back a batch dimension.
            output_tensors.append(final_tensor.unsqueeze(0))
        
        # Concatenate along the batch dimension and move back to the device.
        return torch.cat(output_tensors, dim=0).to(self.device)
