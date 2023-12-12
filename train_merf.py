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

    def render_merf(R, T, viewpoint_camera, gaussians):
      
        rendering = render_RT(R,T,viewpoint_camera, gaussians, pipeline, background)["render"]
        return rendering

    merf_eval_model = ResMem(pretrained=True)
    merf = MerfNet(gaussians, render_merf, single_camera[0], merf_eval_model, pipeline, background)
    merf.train()

    optimizer = torch.optim.Adam([merf.camera_pos], lr=0.1)
    prev_loss = 0
    for i in 1000:
        optimizer.zero_grad()
        loss, _ = merf()
        loss.backward(retain_graph=True)
        optimizer.step()

        # if i % 100 == 0:
        #     for g in optimizer.param_groups:
        #         g['lr'] += 0.02
        #     torchvision.utils.save_image(rendering, os.path.join("merf_outputs", '{0:05d}'.format(i) + ".png"))


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
