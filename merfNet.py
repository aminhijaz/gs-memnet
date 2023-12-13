import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from scene.cameras import Camera as GSCamera
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from resmem import ResMem
import math
class MerfNet(nn.Module):
    def __init__(self, gaussians, renderer, camera, resmodel, pipeline, background, loss_fn=torch.nn.MSELoss(), device="cuda"):
        super().__init__()
        self.device = device
        self.Rt = None
        self.gaussians = gaussians
        self.renderer = renderer
        self.transform = transforms.Compose((
            transforms.Resize((256, 256)),
            # transforms.CenterCrop(227),
        ))
        self.camera = camera
        self.background = background
        self.pipeline = pipeline
        # optimize the camera translation
        self.camera_pos = nn.Parameter(
            torch.from_numpy(np.array([3.0,  6.9, +2.5], dtype=np.float32)).to(device))
        self.resmodel = resmodel.to(device)
        for param in self.resmodel.parameters():
            param.requires_grad = False
        self.resmodel.eval()
        self.loss_fn = loss_fn
        self.rasterizer = GaussianRasterizer(raster_settings=None)
        self.rasterizer.eval()
        for param in self.rasterizer.parameters():
            param.requires_grad = False






    def forward(self):
        tanfovx = math.tan(self.camera.FoVx * 0.5)
        tanfovy = math.tan(self.camera.FoVy * 0.5)
        R = look_at_rotation(self.camera_pos[None, :], device=self.device)
        T = -torch.bmm(R.transpose(1, 2), self.camera_pos[None, :, None])[:, :, 0]
        R = R.squeeze(0)
        Rt_top_left = R.transpose(0, 1)
        Rt_bottom = torch.tensor([[0., 0., 0., 1.]], device=self.device, dtype=torch.float32)
        Rt_top_right = T.view(3, 1)  # Reshape T to [3, 1] if it's not already
        Rt = torch.cat([torch.cat([Rt_top_left, Rt_top_right], dim=1), Rt_bottom], dim=0)
        # Now set requires_grad to True
        pc = self.gaussians
        self.raster_settings = GaussianRasterizationSettings(
        image_height=int(self.camera.image_height),
        image_width=int(self.camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=self.background,
        scale_modifier=1.0,
        viewmatrix=Rt,
        projmatrix=self.camera.full_proj_transform,
        sh_degree=self.gaussians.active_sh_degree,
        campos=self.camera_pos.world_view_transform,
        prefiltered=False,
        debug=self.pipeline.debug
    )
        self.rasterizer.raster_settings = self.raster_settings
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if self.pipeline.compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(3)
        else:
            scales = self.gaussians.get_scaling
            rotations = self.gaussians.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        shs = self.gaussians.get_features
        rendered_image, _, _, _ = self.rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp
            viewMatrix = Rt)
        i = self.transform(rendered_image.unsqueeze(0))
        prediction = self.resmodel.forward(i)
        loss = self.loss_fn(prediction, torch.ones(1, 1, dtype=torch.float32).to(self.device))
        return loss, prediction

def look_at_rotation(
    camera_position, at=((0, 0, 0),), up=((0, 1, 0),), device = "cpu"
) -> torch.Tensor:
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.

    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.

    Args:
        camera_position: position of the camera in world coordinates
        at: position of the object in world coordinates
        up: vector specifying the up direction in the world coordinate frame.

    The inputs camera_position, at and up can each be a
        - 3 element tuple/list
        - torch tensor of shape (1, 3)
        - torch tensor of shape (N, 3)

    The vectors are broadcast against each other so they all have shape (N, 3).

    Returns:
        R: (N, 3, 3) batched rotation matrices
    """
    # Format input and broadcast
    broadcasted_args = convert_to_tensors_and_broadcast(
        camera_position, at, up, device=device
    )
    camera_position, at, up = broadcasted_args
    for t, n in zip([camera_position, at, up], ["camera_position", "at", "up"]):
        if t.shape[-1] != 3:
            msg = "Expected arg %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))
    z_axis = F.normalize(at - camera_position, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(
        dim=1, keepdim=True
    )
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    return R.transpose(1, 2)

def convert_to_tensors_and_broadcast(
    *args,
    dtype = torch.float32,
    device = "cpu",
):
    # Convert all inputs to tensors with a batch dimension
    args_1d = [format_tensor(c, dtype, device) for c in args]

    # Find broadcast size
    sizes = [c.shape[0] for c in args_1d]
    N = max(sizes)

    args_Nd = []
    for c in args_1d:
        if c.shape[0] != 1 and c.shape[0] != N:
            msg = "Got non-broadcastable sizes %r" % sizes
            raise ValueError(msg)

        # Expand broadcast dim and keep non broadcast dims the same size
        expand_sizes = (N,) + (-1,) * len(c.shape[1:])
        args_Nd.append(c.expand(*expand_sizes))

    return args_Nd

def format_tensor(
    input,
    dtype = torch.float32,
    device = "cpu",
) -> torch.Tensor:
    if not torch.is_tensor(input):
        input = torch.tensor(input, dtype=dtype, device=device)

    if input.dim() == 0:
        input = input.view(1)

    if input.device == device:
        return input

    input = input.to(device=device)
    return input
