import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from gaussian_splatting.scene.cameras import Camera as GSCamera
from pytorch3d.renderer import (
    look_at_view_transform, look_at_rotation,
)


class MerfNet(nn.Module):
    def __init__(self, gaussians, renderer, camera, resmodel=None, device="cuda"):
        super(MerfNet, self).__init__()
        self.device = device
        self.gaussians = gaussians
        self.renderer = renderer
        self.transform = transforms.Compose((
            transforms.Resize((256, 256)),
            transforms.CenterCrop(227),
        ))
        self.camera = camera
        # optimize the camera pos and camera lookat
        self.camera_pos = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], device=device))
        self.camera_lookat = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], device=device))
        self.resmodel = resmodel
        for param in self.resmodel.parameters():
            param.requires_grad = False
        self.resmodel.eval()

    def forward(self):
        image = self.renderer(self.camera, self.gaussians)
        i = image[0, :, :, :3]
        i = i.permute(2, 0, 1)
        i = self.transform(i.unsqueeze(0))
        prediction = self.resmodel(i)
        print(prediction.item())

        return image, prediction

    def update_camera(self):
        R = look_at_rotation(self.camera_pos[None, :], self.camera_lookat[None, :], device=self.device)
        T = -torch.bmm(R.transpose(1, 2), self.camera_pos[None, :, None])[:, :, 0]

        prev_camera = self.camera
        new_camera = GSCamera(
            prev_camera.colmap_id, R, T, prev_camera.FoVx, prev_camera.FoVy,
            prev_camera.original_image, None,
            prev_camera.image_name, prev_camera.uid
        )
        self.camera = new_camera
