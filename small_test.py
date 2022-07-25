import inference_utils
import argparse

args = inference_utils.get_args_parser()
parser = argparse.ArgumentParser("EsViT", parents=[args])
args = parser.parse_args()

_, o = inference_utils.eval_esvit(args)
print(_[0][0])
