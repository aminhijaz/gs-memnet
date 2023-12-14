from merfNet import MerfNet
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_RT
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from resmem import ResMem
from torchvision import transforms

import numpy as np

def train_merf(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    makedirs("merf_outputs", exist_ok=True)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()

        # for our purposes, we only want to render one camera
        # later on we can play with this
        all_cameras = train_cameras + test_cameras
        single_camera = all_cameras[0:1]

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    def render_merf(viewpoint_camera, gaussians):
        rendering = render(viewpoint_camera, gaussians, pipeline, background)["render"]
        return rendering

    merf_eval_model = ResMem(pretrained=True)
    merf = MerfNet(gaussians, render_merf, single_camera[0], merf_eval_model, pipeline, background)
    merf.train()
    optimizer = torch.optim.AdamW([merf.camera_pos], lr=0.0001)
    best_render_image = render_merf(single_camera[0], gaussians)
    best_pred = 0
    loop = range(1000)
    for i in loop:
        optimizer.zero_grad()
        loss, rendered_image, prediction = merf()
        if(prediction > best_pred):
            best_pred = prediction
            best_render_image = rendered_image
        loss.backward(retain_graph=True)
        optimizer.step()
    torchvision.utils.save_image(best_render_image, os.path.join("merf_outputs", "best_new_camera.png"))
    print("rendering original cameras")
    resmem = ResMem(pretrained=True)
    resmem.eval()
    transform = transforms.Compose((
        transforms.Resize((256, 256)),
    ))
    original_scores =[]
    for camera in all_cameras[:1]:
        rendering = render(camera, gaussians, pipeline, background)["render"]
        image = transform(rendering.unsqueeze(0))
        score = resmem.forward(image.to("cuda")).cpu().item()
        original_scores.append(score)
    original_scores = np.array(original_scores)
    best_original_idx = np.argmax(original_scores)
    best_original_score = original_scores[best_original_idx]
    best_original_camera = all_cameras[best_original_idx]
    rendered = render_merf(best_original_camera, gaussians)
    print("original score")
    print(best_original_score)
    torchvision.utils.save_image(best_render_image, os.path.join("merf_outputs", "best_orginal_camera.png"))
    
    





if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    gs_model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    train_merf(gs_model.extract(args), args.iteration, pipeline.extract(args))
