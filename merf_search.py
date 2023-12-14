import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from resmem.model import ResMem
from torchvision import transforms
from scene.cameras import Camera, MiniCam
import torch.nn.functional as F
from utils.graphics_utils import getProjectionMatrix
import numpy as np


def merf_search(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    makedirs("merf_outputs", exist_ok=True)
    gaussians = GaussianModel(dataset.sh_degree)
    # gaussians.train()
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()

    all_cameras = train_cameras + test_cameras
    new_random_cameras = []

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # the step size for the camera movements
    delta = 0.1
    # the number of new cameras to generate along a direction
    num_extents = 10
    # how far to move the orginal camera to generate new ones along that movement
    extension_range = num_extents * delta
    num_original_cameras_used = 1

    transform = transforms.Compose((
        transforms.Resize((256, 256)),
    ))

    loop = tqdm(range(1))
    print("duplicating cameras")
    for camera in all_cameras[:num_original_cameras_used]:
        original_pos = torch.from_numpy(camera.T).to("cuda")
        # create a new camera instance
        mini_cam = fromGS2MiniCam(camera)
        # print("original pos: ", original_pos)
        for i in loop:
            # print("random camera: ", i)
            # generate a random direction in the range of the extension range
            rand_dir = (torch.rand(3) - 0.5) * 2 * extension_range
            rand_dir = rand_dir.to("cuda")
            # move the camera in that random direction
            new_pos = original_pos + rand_dir
            distance = torch.norm(new_pos - original_pos)
            # print("distance: ", distance)
            # based on distance, tune the number of extents
            h = distance / delta
            # print("h: ", h)
            # linspace from original pos to new pos for each dimension
            # basically, create h more cameras along the movement of the camera
            # from original position to new position, uniformly generate cameras
            new_positions = torch.empty((3, int(h)), dtype=torch.float32, device="cuda")
            # movement is in all 3 directions, so put them together
            for j in range(3):
                new_positions[j] = torch.linspace(original_pos[j], new_pos[j], int(h))
            new_positions = torch.stack((new_positions[0], new_positions[1], new_positions[2]), dim=1)
            # print("new positions: ", new_positions)
            for new_pos in new_positions:
                # for each new pos we created, create a new cam instance
                # update its parameters based on new pos
                new_cam = fromGS2MiniCam(mini_cam)
                update_mini_camera(new_pos, new_cam)
                new_random_cameras.append((new_cam, new_pos))

    print("done duplicating cameras")
    print(len(new_random_cameras))

    resmem = ResMem(pretrained=True)
    resmem.to("cuda")

    original_scores = []
    print("rendering original cameras")
    for camera in all_cameras[:num_original_cameras_used]:
        rendering = render(camera, gaussians, pipeline, background)["render"]
        image = transform(rendering.unsqueeze(0))
        score = resmem.forward(image.to("cuda")).cpu().item()
        original_scores.append(score)

    new_scores = []
    print("rendering new cameras")
    for camera, pos in new_random_cameras:
        rendering = render(camera, gaussians, pipeline, background)["render"]
        image = transform(rendering.unsqueeze(0))
        score = resmem.forward(image.to("cuda")).cpu().item()
        new_scores.append(score)

    print("finding best camera")
    original_scores = np.array(original_scores)
    new_scores = np.array(new_scores)
    best_original_idx = np.argmax(original_scores)
    best_new_idx = np.argmax(new_scores)
    best_original_camera = all_cameras[best_original_idx]
    best_new_camera, best_new_pos = new_random_cameras[best_new_idx]
    best_original_score = original_scores[best_original_idx]
    best_new_score = new_scores[best_new_idx]

    print("best original camera score: ", best_original_score)
    print("best new camera score: ", best_new_score)
    
    print("saving best cameras")
    torchvision.utils.save_image(
        render(best_original_camera, gaussians, pipeline, background)["render"], 
        os.path.join("merf_outputs", "best_original_camera.png"))
    torchvision.utils.save_image(
        render(best_new_camera, gaussians, pipeline, background)["render"], 
        os.path.join("merf_outputs", "best_new_camera.png"))

def update_mini_camera(new_pos, mini_cam):
    R = look_at_rotation(new_pos[None, :], device="cuda")
    T = -torch.bmm(R.transpose(1, 2), new_pos[None, :, None])[:, :, 0]
    R = R.squeeze(0)
    world_view_transform = getWorld2View2Torch(R, T).transpose(0, 1)
    projection_matrix = getProjectionMatrix(
        znear=mini_cam.znear, zfar=mini_cam.zfar, 
        fovX=mini_cam.FoVx, fovY=mini_cam.FoVy).transpose(0,1).to("cuda")
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    view_inv = torch.inverse(world_view_transform)
    mini_cam.full_proj_transform = full_proj_transform
    mini_cam.world_view_transform = world_view_transform
    mini_cam.camera_center = view_inv[3][:3]


def look_at_rotation(
    camera_position, 
    at=((0, 1, 0),), 
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

def fromGS2MiniCam(camera):
    new_mini_cam = MiniCam(
        camera.image_width,
        camera.image_height,
        camera.FoVx,
        camera.FoVy,
        camera.znear,
        camera.zfar,
        camera.world_view_transform,
        camera.full_proj_transform
    )
    return new_mini_cam

def getWorld2View2Torch(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0, device="cuda"):
    translate = translate.to(device)
    Rt = torch.zeros((4, 4), dtype=torch.float32, device=device)
    Rt[:3, :3] = R.transpose(0, 1)  
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.inverse(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.inverse(C2W)
    return Rt


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    gs_model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    merf_search(gs_model.extract(args), args.iteration, pipeline.extract(args))
