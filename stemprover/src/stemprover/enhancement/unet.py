import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from typing import List, Tuple

class UNetWrapper(nn.Module):
    """
    A wrapper for the diffusers UNet2DConditionModel to make it compatible
    with the PhaseAwareControlNet architecture. This class exposes methods
    to get intermediate feature maps and to run the second half of the UNet
    with modified features.
    """
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5", subfolder: str = "unet"):
        super().__init__()
        # Load the pre-trained UNet model
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder=subfolder)

        # Define the points for feature injection (ControlNet)
        # These are the outputs of the down blocks and the mid block
        self.num_injection_points = len(self.unet.down_blocks) * 3 + 1 # 3 resnets per downblock + midblock

        # Store feature channel dimensions for the zero-convolutions in the ControlNet
        self._feature_channels = [b.resnets[-1].out_channels for b in self.unet.down_blocks]
        self._feature_channels.append(self.unet.mid_block.resnets[-1].out_channels)

    def get_feature_channels(self, i: int) -> int:
        """Returns the number of channels for the i-th injection point."""
        # This is a simplified mapping; a real implementation might need more detail
        if i < 9: # 3 resnets in each of the first 3 downblocks
            return self._feature_channels[i // 3]
        return self._feature_channels[-1] # Mid block

    def get_feature_pyramid(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Performs a partial forward pass through the UNet's down-sampling blocks
        and returns the intermediate feature maps.
        """
        features = []

        # Initial convolution
        x = self.unet.conv_in(sample)

        # Down-sampling blocks
        for down_block in self.unet.down_blocks:
            for resnet in down_block.resnets:
                x = resnet(x, timestep)
                features.append(x)
            if down_block.downsamplers:
                x = down_block.downsamplers[0](x)
                features.append(x)

        # Mid block
        x = self.unet.mid_block(x, timestep, encoder_hidden_states)
        features.append(x)
        return features

    def forward_with_features(
        self,
        x: torch.Tensor,
        features: List[torch.Tensor],
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Runs the second half of the UNet (up-sampling blocks) using the
        modified feature maps provided by the ControlNet.
        """
        # The `features` are the ControlNet-modified outputs of the down blocks + mid block
        # We need to feed these into the up-sampling blocks as skip connections

        # Start with the output of the mid-block, which is the last feature
        x = features.pop()

        # Up-sampling blocks
        for up_block in self.unet.up_blocks:
            # The number of skip connections needed for this up-block
            num_res_samples = len(up_block.resnets)

            # Get the skip connections from the end of the features list
            res_samples = features[-num_res_samples:]
            features = features[:-num_res_samples]

            if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
                x = up_block(
                    hidden_states=x,
                    temb=timestep,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                x = up_block(
                    hidden_states=x,
                    temb=timestep,
                    res_hidden_states_tuple=res_samples,
                )

        # Post-processing
        if self.unet.conv_norm_out:
            x = self.unet.conv_norm_out(x)
            x = self.unet.conv_act(x)
        x = self.unet.conv_out(x)

        return x

    def forward(self, *args, **kwargs):
        """The standard forward pass is just the original UNet's forward pass."""
        return self.unet(*args, **kwargs)
