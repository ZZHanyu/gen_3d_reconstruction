import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch import nn


class RandomPatchExtractor(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self, tensor: torch.Tensor, centers: torch.Tensor, patch_size: tuple[int, int]
    ):
        device = tensor.device
        dtype = tensor.dtype
        patch_width, patch_height = patch_size
        pad_width = patch_width // 2
        pad_height = patch_height // 2
        dtype = tensor.dtype

        # Pad input to avoid out-of-bounds
        tensor_padded = F.pad(
            tensor,
            (pad_width, pad_width, pad_height, pad_height),
            mode="constant",
            value=0.0,
        )

        # Adjust edge coordinates to account for padding
        centers_padded = centers + torch.tensor(
            [pad_height, pad_width], dtype=dtype, device=device
        ).reshape(1, 1, 2)

        output = ExtractPatchesFunction.apply(
            tensor_padded.float(), centers_padded.int(), patch_height, patch_width
        )
        return output.to(dtype)


class ExtractPatchesFunction(Function):
    @staticmethod
    def forward(ctx, input, centers, h, w):
        # Save variables for backward pass. inputs for shapes
        ctx.save_for_backward(input, centers)

        return RandomPatchExtraction.extract_patches_forward(input, centers, h, w)

    @staticmethod
    def backward(ctx, grad_output):
        input, centers = ctx.saved_tensors

        (grad_input,) = RandomPatchExtraction.extract_patches_backward(
            grad_output, centers, input.shape[2], input.shape[3]
        )
        # breakpoint()

        # Return gradients with respect to inputs only
        return grad_input, None, None, None


# Test
if __name__ == "__main__":
    RandomPatchExtraction = RandomPatchExtractor()
    B, C, H, W = 1, 1, 10, 10
    N = 2
    h, w = 3, 3
    input = torch.arange(
        B * C * H * W, device="cuda", dtype=torch.float32, requires_grad=True
    ).view(B, C, H, W)
    centers = torch.tensor([[[4, 4], [6, 6]]], device="cuda", dtype=torch.int32)
    output = ExtractPatchesFunction.apply(input, centers, h, w)
    output.mean().backward()
