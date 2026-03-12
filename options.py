from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

def str2bool(v):
     if isinstance(v, bool):
          return v
     if v.lower() in ('yes', 'true', 't', 'y', '1'):
          return True
     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
          return False
     else:
          raise argparse.ArgumentTypeError('Boolean value expected.')

class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the dataset",
                                 default=os.path.join(file_dir, "endovis_data"))

        # Model options
        self.parser.add_argument("--pretrained_path",
                                 type=str,
                                 help="pretrained weights path",
                                 default=os.path.join(file_dir, "pretrained_model"))
        self.parser.add_argument("--lora_type",
                                 type=str,
                                 help="which lora type use for the model",
                                 choices=["lora", "dvlora", "none"],
                                 default="dvlora")
        self.parser.add_argument("--lora_rank",
                                 type=int,
                                 help="the rank of lora",
                                 default=4)
        self.parser.add_argument("--residual_block_indexes",
                                 nargs="*",
                                 type=int,
                                 help="indexes for residual blocks in vitendodepth encoder",
                                 default=[2,5,8,11])
        self.parser.add_argument("--include_cls_token",
                                 type=str2bool,
                                 help="includes the cls token in the transformer blocks",
                                 default=True)
        self.parser.add_argument("--learn_intrinsics",
                                 type=str2bool,
                                 help="learn the camera intrinsics with a separate decoder",
                                 default=True)
        self.parser.add_argument("--use_isa",
                                 type=str2bool,
                                 help="use Illumination-Semantic Attention module",
                                 default=True)

        # FPN options
        self.parser.add_argument("--use_bifpn",
                                 type=str2bool,
                                 help="use BiFPN for multi-scale feature fusion",
                                 default=True)
        self.parser.add_argument("--bifpn_num_blocks",
                                 type=int,
                                 help="number of BiFPN blocks (repeat times)",
                                 default=2)
        self.parser.add_argument("--use_geo_injection",
                                 type=str2bool,
                                 help="enable geometry information injection in BiFPN",
                                 default=True)
        self.parser.add_argument("--geo_type",
                                 type=str,
                                 choices=["gradient", "normal"],
                                 help="geometry information type: gradient or normal",
                                 default="gradient")
        self.parser.add_argument("--geo_fusion_type",
                                 type=str,
                                 choices=["add", "gated"],
                                 help="geometry fusion type: add or gated (sigmoid)",
                                 default="gated")

        # Speed optimization options
        self.parser.add_argument("--use_lora_fusion",
                                 type=str2bool,
                                 help="use LoRA weight fusion at inference time",
                                 default=True)
        self.parser.add_argument("--use_conv_bn_fusion",
                                 type=str2bool,
                                 help="use Conv-BN fusion at inference time",
                                 default=True)

        # Dataset options
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=256)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=320)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the model",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=150.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=8)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["position_encoder", "position"])

        # EVALUATION options
        self.parser.add_argument("--model_type",
                                 type=str,
                                 help="which model type to evaluate",
                                 choices=["endodac", "afsfm"],
                                 default="endodac")
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="endovis",
                                 choices=["hamlyn", "c3vd", "endovis"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--visualize_depth",
                                 help="if set saves visualized depth map",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        self.parser.add_argument("--save_recon",
                                 help="if set saves reconstruction files",
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
