import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from gaussian_splatting.scene.cameras import Camera as GSCamera


class MerfNet(nn.Module):
    def __init__(self, gaussians, renderer, camera, resmodel, loss_fn=torch.nn.MSELoss(), device="cuda"):
        super().__init__()
        self.device = device
        self.gaussians = gaussians
        self.renderer = renderer
        self.transform = transforms.Compose((
            transforms.Resize((256, 256)),
            # transforms.CenterCrop(227),
        ))
        self.camera = camera
        # optimize the camera translation
        self.camera_pos = nn.Parameter(torch.from_numpy(np.array(camera.T, dtype=np.float32)).to(device))
        self.resmodel = resmodel.to(device)
        for param in self.resmodel.parameters():
            param.requires_grad = False
        self.resmodel.eval()
        self.loss_fn = loss_fn

    def forward(self):
        self.update_camera()
        image = self.renderer(self.camera, self.gaussians)
        i = image
        # i = image[0, :, :, :3]
        # i = i.permute(2, 0, 1)
        i = self.transform(i.unsqueeze(0))
        prediction = self.resmodel.forward(i.to(self.device))
        loss = self.loss_fn(prediction, torch.ones(1, 1).to(self.device))

        return loss, image, prediction

    def update_camera(self):
        R = look_at_rotation(self.camera_pos[None, :], device=self.device)
        T = -torch.bmm(R.transpose(1, 2), self.camera_pos[None, :, None])[:, :, 0]
        R = R.squeeze(0)
        new_R = R.clone().detach().cpu().numpy()
        new_T = T.clone().detach().cpu().numpy()
        prev_camera = self.camera
        new_camera = GSCamera(
            prev_camera.colmap_id, new_R, new_T,
            prev_camera.FoVx, prev_camera.FoVy,
            prev_camera.original_image, None,
            prev_camera.image_name, prev_camera.uid
        )
        self.camera = new_camera


def look_at_rotation(
    camera_position, 
    at=((0, 0, 0),), 
    up=((0, 1, 0),),
    device="cuda"):

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
