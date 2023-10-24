import os
import sys
import json
import logging
import warnings
import argparse
from time import time
from uuid import uuid4

import torch
from diffusers import DiffusionPipeline


from src.backend.lcmdiffusion.pipelines.openvino.lcm_ov_pipeline import (
    OVLatentConsistencyModelPipeline,
)

from src.backend.lcmdiffusion.pipelines.openvino.lcm_scheduler import LCMScheduler

# Constants
RESULTS_DIRECTORY = "results"
CONFIG_FILE = 'config.json'

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if not os.path.exists(RESULTS_DIRECTORY):
    os.makedirs(RESULTS_DIRECTORY)


def set_seed(use_seed=True, seed_value=None):
    if use_seed:
        if seed_value is None:
            seed_value = torch.randint(0, 1 << 31, (1,)).item()
        torch.manual_seed(seed_value)
        logger.info(f"Using seed value: {seed_value}")
        return seed_value
    else:
        return None


def load_config(config_file=CONFIG_FILE):
    defaults = {
        'show_image': False,
        'guidance_scale': None,
        'img_width': None,
        'img_height': None,
        'inference_steps': None,
        'lcm_model_id': None,
        'use_openvino': None,
        'use_seed': None,
        'use_safety_checker': None
    }
    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
            # Merge the dictionaries, but prioritize the config's values over defaults
            merged_config = {**defaults, **config}
            return merged_config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_file} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(
            f"Failed to parse configuration file {config_file}. Ensure it contains valid JSON.")
        sys.exit(1)

    # def load_config(config_file='config.json'):
    #     defaults = {
    #         'show_image': False,
    #         'guidance_scale': None,
    #         'img_width': None,
    #         'img_height': None,
    #         'inference_steps': None,
    #         'lcm_model_id': None,
    #         'use_openvino': None,
    #         'use_seed': None,
    #         'use_safety_checker': None
    #     }
    #     try:
    #         with open(config_file, 'r') as file:
    #             config = json.load(file)
    #             return {**defaults, **config}
    #     except FileNotFoundError:
    #         logger.error(f"Configuration file {config_file} not found.")
    #         sys.exit(1)
    #     except json.JSONDecodeError:
    #         logger.error(
    #             f"Failed to parse configuration file {config_file}. Ensure it contains valid JSON.")
    #         sys.exit(1)


config = load_config()
# default to False if not in config
show_image = config.get('show_image', False)


def get_results_path():
    app_dir = os.path.dirname(__file__)
    config_path = os.path.join(app_dir, RESULTS_DIRECTORY)
    return config_path


def generate_image(prompt, guidance_scale, img_width, img_height, num_inference_steps, use_openvino, model_id, use_seed, seed_value, safety_checker):
    output_path = get_results_path()

    if use_openvino:
        logger.info("Using OpenVINO for image generation.")
        scheduler = LCMScheduler.from_pretrained(
            model_id, subfolder="scheduler")
        pipeline = OVLatentConsistencyModelPipeline.from_pretrained(
            "deinferno/LCM_Dreamshaper_v7-openvino",
            scheduler=scheduler,
            compile=False,
        )
    else:
        logger.info("Not using OpenVINO for image generation.")
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            custom_pipeline="latent_consistency_txt2img",
            custom_revision="main",
        )
        pipeline.to(torch_device="cpu", torch_dtype=torch.float32)

    logger.info(f"Prompt : {prompt}")
    logger.info(f"Resolution : {img_width} x {img_height}")
    logger.info(f"Guidance Scale : {guidance_scale}")
    logger.info(f"Inference_steps  : {num_inference_steps}")

    if use_seed:
        logger.info(f"Seed: {seed_value}")

    tick = time()

    if not safety_checker:
        pipeline.safety_checker = None

    images = pipeline(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        lcm_origin_steps=50,
        width=img_width,
        height=img_height,
        output_type="pil",
    ).images

    elapsed = time() - tick
    logger.info(f"Elapsed time : {elapsed:.2f} sec")
    image_id = uuid4()

    # Save the image
    image_path = os.path.join(output_path, f"{image_id}.png")
    images[0].save(image_path)
    logger.info(f"Image {image_id}.png saved")

    # Show image if true
    if show_image:
        logger.info(f"show_image: true - Displaying {image_id}.png")
        images[0].show()

    # Save the parameters used in a JSON file with the same UUID
    with open(os.path.join(output_path, f"{image_id}.json"), 'w') as json_file:
        json.dump({
            'prompt': prompt,
            'guidance_scale': guidance_scale,
            'img_width': img_width,
            'img_height': img_height,
            'num_inference_steps': num_inference_steps,
            'use_openvino': use_openvino,
            'model_id': model_id,
            'use_seed': use_seed,
            'seed_value': seed_value,
            'safety_checker': safety_checker,
            'elapsed_time_sec': elapsed
        }, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image generation script.')

    parser.add_argument('--prompt', type=str, required=True,
                        help='Image prompt.')
    parser.add_argument('--guidance-scale', type=float, default=config['guidance_scale'],
                        help='Guidance scale.')
    parser.add_argument('--img-width', type=int, default=config['img_width'],
                        help='Image width.')
    parser.add_argument('--img-height', type=int, default=config['img_height'],
                        help='Image height.')
    parser.add_argument('--inference-steps', type=int, default=config['inference_steps'],
                        help='Number of inference steps.')
    parser.add_argument('--lcm-model-id', type=str, default=config['lcm_model_id'],
                        help='LCM model ID.')
    parser.add_argument('--use-openvino', action='store_true', default=config['use_openvino'],
                        help='Use OpenVINO.')
    parser.add_argument('--use-seed', action='store_true', default=config['use_seed'],
                        help='Use seed.')
    parser.add_argument('--use-safety-checker', action='store_true', default=config['use_safety_checker'],
                        help='Use safety checker.')
    parser.add_argument('--show-image', action='store_true', default=config.get('show_image', False),
                        help='Show generated image.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Specific seed for reproducibility. If not provided, a random seed will be used.')

    args = parser.parse_args()

    # Set the seed here and use the returned value throughout the script
    seed_value = set_seed(args.use_seed, args.seed)

    generate_image(args.prompt, args.guidance_scale, args.img_width, args.img_height, args.inference_steps,
                   args.use_openvino, args.lcm_model_id, args.use_seed, seed_value, args.use_safety_checker)
