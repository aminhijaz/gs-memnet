import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from scene.cameras import Camera as GSCamera


class MerfNet(nn.Module):
    def __init__(self, gaussians, renderer, camera, resmodel, device="cuda"):
        super(MerfNet, self).__init__()
        self.device = device
        self.gaussians = gaussians
        self.renderer = renderer
        self.transform = transforms.Compose((
            transforms.Resize((256, 256)),
            # transforms.CenterCrop(227),
        ))
        camR = camera.R
        camT = camera.T
        # randomize the camera rotation and translation
        print("Camera R and T")
        print(camR)
        print(camT)
        camR = np.random.rand(3, 3)
        camT = np.random.rand(3)
        print("Randomized Camera R and T")
        print(camR)
        print(camT)
        self.camera = camera
        # position and lookat from R and T ?
        # optimize the camera pos and camera lookat
        self.camera_R = nn.Parameter(torch.from_numpy(camR)).to(device)
        self.camera_T = nn.Parameter(torch.from_numpy(camT)).to(device)
        # update the camera with the new R and T
        self.update_camera()
        self.resmodel = resmodel.to(device)
        for param in self.resmodel.parameters():
            param.requires_grad = False
        self.resmodel.eval()

    def forward(self):
        image = self.renderer(self.camera, self.gaussians)
        # print(image.shape)
        i = image
        # i = image[0, :, :, :3]
        # i = i.permute(2, 0, 1)
        i = self.transform(i.unsqueeze(0))
        print(i.shape)
        prediction = self.resmodel.forward(i.to(self.device))
        print(prediction.item())

        return image, prediction

    def update_camera(self):
        # R = look_at_rotation(self.camera_pos[None, :], self.camera_lookat[None, :], device=self.device)
        # T = -torch.bmm(R.transpose(1, 2), self.camera_pos[None, :, None])[:, :, 0]
        print(self.camera_R)
        print(self.camera_T)
        with torch.no_grad():
            new_R = self.camera_R.cpu().numpy()
            new_T = self.camera_T.cpu().numpy()
            print(new_T)
            prev_camera = self.camera
            new_camera = GSCamera(
                prev_camera.colmap_id, new_R, new_T,
                prev_camera.FoVx, prev_camera.FoVy,
                prev_camera.original_image, None,
                prev_camera.image_name, prev_camera.uid
            )
            self.camera = new_camera


# def look_at_rotation(camera_position, at, up=torch.from_numpy(np.array([0, 1, 0])), device="cuda"):
#     up = up.to(device)
#     for t, n in zip([camera_position, at, up], ["camera_position", "at", "up"]):
#         if t.shape[-1] != 3:
#             msg = "Expected arg %s to have shape (N, 3); got %r"
#             raise ValueError(msg % (n, t.shape))
#     z_axis = F.normalize(at - camera_position, eps=1e-5)
#     x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
#     y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
#     is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(
#         dim=1, keepdim=True
#     )
#     if is_close.any():
#         replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
#         x_axis = torch.where(is_close, replacement, x_axis)
#     R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    
#     return R.transpose(1, 2)
