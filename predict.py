from cog import BasePredictor, Input, Path
import torch
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, get_agnostic_mask_hd, get_agnostic_mask_dc
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
import os

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Create directories
        os.makedirs("./ckpts", exist_ok=True)
        
        # Download checkpoints from HuggingFace
        snapshot_download(
            repo_id="franciszzj/Leffa",
            local_dir="./ckpts",
            ignore_patterns=["*.md", "*.txt", "*.json", "assets/*", "*.py"]
        )
        
        self.mask_predictor = AutoMasker(
            densepose_path="./ckpts/densepose",
            schp_path="./ckpts/schp",
        )

        self.densepose_predictor = DensePosePredictor(
            config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="./ckpts/densepose/model_final_162be9.pkl",
        )

        self.parsing = Parsing(
            atr_path="./ckpts/humanparsing/parsing_atr.onnx",
            lip_path="./ckpts/humanparsing/parsing_lip.onnx",
        )

        self.openpose = OpenPose(
            body_model_path="./ckpts/openpose/body_pose_model.pth",
        )

        # Initialize models
        self.vt_model_hd = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon.pth",
            dtype="float16",
        )
        self.vt_inference_hd = LeffaInference(model=self.vt_model_hd)

        self.vt_model_dc = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon_dc.pth",
            dtype="float16",
        )
        self.vt_inference_dc = LeffaInference(model=self.vt_model_dc)

        self.transform = LeffaTransform()

    def predict(
        self,
        src_image: Path = Input(description="Source person image"),
        ref_image: Path = Input(description="Reference garment image"),
        model_type: str = Input(
            description="Model type to use",
            choices=["viton_hd", "dress_code"],
            default="viton_hd"
        ),
        garment_type: str = Input(
            description="Type of garment",
            choices=["upper_body", "lower_body", "dresses"],
            default="upper_body"
        ),
        ref_acceleration: bool = Input(
            description="Accelerate Reference UNet (may slightly reduce performance)",
            default=False
        ),
        inference_steps: int = Input(
            description="Number of inference steps",
            default=30,
            ge=30,
            le=100
        ),
        guidance_scale: float = Input(
            description="Guidance scale",
            default=2.5,
            ge=0.1,
            le=5.0
        ),
        seed: int = Input(
            description="Random seed",
            default=42
        )
    ) -> Path:
        """Run virtual try-on prediction"""
        # Load and preprocess images
        src_image = Image.open(src_image)
        ref_image = Image.open(ref_image)
        src_image = resize_and_center(src_image, 768, 1024)
        ref_image = resize_and_center(ref_image, 768, 1024)

        src_image_array = np.array(src_image)

        # Generate mask
        src_image = src_image.convert("RGB")
        model_parse, _ = self.parsing(src_image.resize((384, 512)))
        keypoints = self.openpose(src_image.resize((384, 512)))
        
        if model_type == "viton_hd":
            mask = get_agnostic_mask_hd(model_parse, keypoints, garment_type)
        else:
            mask = get_agnostic_mask_dc(model_parse, keypoints, garment_type)
        mask = mask.resize((768, 1024))

        # Generate DensePose
        if model_type == "viton_hd":
            src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
            densepose = Image.fromarray(src_image_seg_array)
        else:
            src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)
            src_image_seg_array = src_image_iuv_array[:, :, 0:1]
            src_image_seg_array = np.concatenate([src_image_seg_array] * 3, axis=-1)
            densepose = Image.fromarray(src_image_seg_array)

        # Prepare data for inference
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = self.transform(data)

        # Run inference
        inference = self.vt_inference_hd if model_type == "viton_hd" else self.vt_inference_dc
        output = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )

        # Save and return result
        output_path = Path("output.png")
        output["generated_image"][0].save(str(output_path))
        return output_path