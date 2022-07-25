import argparse
import os
import time
import torch

# from torchvision import models as torchvision_models
import utils
from models.vision_transformer import DINOHead
from models import build_model
from config import config
from config import update_config
from datasets import build_dataloader
import cv2
import json


def get_args_parser():
    parser = argparse.ArgumentParser("EsViT", add_help=False)

    parser.add_argument(
        "--cfg",
        default="experiments/imagenet/swin/swin_small_patch4_window14_224.yaml",
        help="experiment configure file name",
        type=str,
    )

    # Model parameters
    parser.add_argument(
        "--arch",
        default="swin_small",
        type=str,
        choices=["swin_tiny", "swin_small", "swin_base", "swin_large", "swin"],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using deit_tiny or deit_small.""",
    )
    parser.add_argument(
        "--out_dim",
        default=65536,
        type=int,
        help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""",
    )
    parser.add_argument(
        "--norm_last_layer",
        default=True,
        type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with deit_small and True with vit_base.""",
    )
    parser.add_argument(
        "--use_bn_in_head",
        default=False,
        type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)",
    )
    parser.add_argument(
        "--use_dense_prediction",
        default=True,
        type=utils.bool_flag,
        help="Whether to use dense prediction in projection head (Default: False)",
    )

    # Training/Optimization parameters
    parser.add_argument(
        "--batch_size_per_gpu",
        default=1,
        type=int,
        help="Per-GPU batch-size : number of distinct images loaded on one GPU.",
    )

    # Dataset
    parser.add_argument(
        "--dataset", default="imagenet1k", type=str, help="Pre-training dataset."
    )
    parser.add_argument(
        "--zip_mode",
        type=utils.bool_flag,
        default=False,
        help="""Whether or not to use zip file.""",
    )
    parser.add_argument(
        "--tsv_mode",
        type=utils.bool_flag,
        default=False,
        help="""Whether or not to use tsv file.""",
    )

    # Misc
    parser.add_argument(
        "--data_path",
        default="./tmp",
        type=str,
        help="Please specify path to the test data.",
    )
    parser.add_argument(
        "--pretrained_weights_ckpt",
        default="params/checkpoint_best.pth",
        type=str,
        help="Path to pretrained weights to evaluate.",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def eval_esvit(args):
    print("Evaluating esvit")
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )

    # ============ preparing data ... ============
    data_loader = build_dataloader(args)

    # ============ building student and teacher networks ... ============
    if "swin" in args.arch:
        update_config(config, args)
        student = build_model(config, use_dense_prediction=args.use_dense_prediction)
        teacher = build_model(
            config, is_teacher=True, use_dense_prediction=args.use_dense_prediction
        )
        print(args.norm_last_layer)
        student.head = DINOHead(
            student.num_features,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
        teacher.head = DINOHead(teacher.num_features, args.out_dim, args.use_bn_in_head)

        if args.use_dense_prediction:
            student.head_dense = DINOHead(
                student.num_features,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            )
            teacher.head_dense = DINOHead(
                teacher.num_features, args.out_dim, args.use_bn_in_head
            )

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ optionally resume training ... ============
    if args.pretrained_weights_ckpt:
        utils.restart_from_checkpoint(
            os.path.join(args.pretrained_weights_ckpt),
            student=student,
            teacher=teacher,
        )
        print(f"Resumed from {args.pretrained_weights_ckpt}")

    # imgs = []
    outs = []
    # labels = []

    t0 = time.time_ns()

    print(len(data_loader))

    for i, (img, label) in enumerate(data_loader):

        out = teacher(img)[-1]
        outs.append(out)

    tf = time.time_ns()
    print(f"Time spend ns: {tf - t0}")
    outs = torch.cat(outs, dim=0)
    return out, outs
