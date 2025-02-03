from modal import Image, App, web_endpoint
import os

def download_models():
    from huggingface_hub import snapshot_download
    
    # Create necessary directories
    os.makedirs("/root/models", exist_ok=True)
    os.makedirs("/root/ckpts", exist_ok=True)
    
    # Download Leffa models and assets
    snapshot_download(
        repo_id="franciszzj/Leffa",
        local_dir="/root/ckpts",
        ignore_patterns=["*.md", "*.git*"]
    )

# Set up Modal image with all dependencies
image = (
    Image.debian_slim(python_version="3.10")
    # First install system dependencies
    .apt_install(
        "git",
        "ffmpeg", 
        "libsm6", 
        "libxext6",
        "libgl1-mesa-glx"
    )
    # Then install Python packages
    .pip_install(
        # Core ML packages
        "torch==2.0.1",
        "torchvision==0.15.2",
        "diffusers>=0.21.4",
        "transformers==4.31.0",
        "xformers",
        # Image processing
        "Pillow",
        "opencv-python",
        "albumentations",
        # Utilities
        "numpy",
        "scipy",
        "einops",
        "timm",
        "omegaconf",
        "huggingface_hub==0.16.4",
        "gradio==4.19.2",
        "pytorch-lightning==1.5.0",
        "test-tube>=0.7.5",
        "basicsr==1.4.2",
        "safetensors>=0.3.1",
    )
    # Install detectron2 separately after other dependencies
    .pip_install("detectron2 @ git+https://github.com/facebookresearch/detectron2.git")
    .run_function(download_models)
)

# Create Modal app instead of Stub
app = App("leffa", image=image)

@app.cls(gpu="A100", container_idle_timeout=60*60)  # Keep container alive for 1 hour when idle
class Leffa:
    def __enter__(self):
        """Initialize the Leffa predictor and models"""
        from app import LeffaPredictor
        
        # Initialize predictor
        self.predictor = LeffaPredictor()
        
        # Set up example images
        self.example_dir = "/root/ckpts/examples"  # Updated path to match download location
        self.person1_images = self.list_dir(f"{self.example_dir}/person1")
        self.person2_images = self.list_dir(f"{self.example_dir}/person2") 
        self.garment_images = self.list_dir(f"{self.example_dir}/garment")

    def list_dir(self, path):
        """Helper to list files in directory"""
        return [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    @web_endpoint()
    def index(self):
        """Create and launch the Gradio interface"""
        import gradio as gr
        
        title = "## Leffa: Learning Flow Fields in Attention for Controllable Person Image Generation"
        link = """[📚 Paper](https://arxiv.org/abs/2412.08486) - [🤖 Code](https://github.com/franciszzj/Leffa) - [🔥 Demo](https://huggingface.co/spaces/franciszzj/Leffa) - [🤗 Model](https://huggingface.co/franciszzj/Leffa)
               
               Star ⭐ us if you like it!
               """
        description = "Leffa is a unified framework for controllable person image generation that enables precise manipulation of both appearance (i.e., virtual try-on) and pose (i.e., pose transfer)."
        note = "Note: The models used in the demo are trained solely on academic datasets. Virtual try-on uses VITON-HD/DressCode, and pose transfer uses DeepFashion."

        with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.pink, secondary_hue=gr.themes.colors.red)).queue() as demo:
            gr.Markdown(title)
            gr.Markdown(link)
            gr.Markdown(description)

            with gr.Tab("Control Appearance (Virtual Try-on)"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Person Image")
                        vt_src_image = gr.Image(
                            sources=["upload"],
                            type="filepath",
                            label="Person Image", 
                            width=512,
                            height=512,
                        )
                        gr.Examples(
                            inputs=vt_src_image,
                            examples=self.person1_images,
                            examples_per_page=10
                        )

                    with gr.Column():
                        gr.Markdown("#### Garment Image")
                        vt_ref_image = gr.Image(
                            sources=["upload"],
                            type="filepath",
                            label="Garment Image",
                            width=512,
                            height=512,
                        )
                        gr.Examples(
                            inputs=vt_ref_image,
                            examples=self.garment_images,
                            examples_per_page=10
                        )

                    with gr.Column():
                        gr.Markdown("#### Generated Image")
                        vt_gen_image = gr.Image(
                            label="Generated Image",
                            width=512,
                            height=512,
                        )

                        with gr.Row():
                            vt_gen_button = gr.Button("Generate")

                        with gr.Accordion("Advanced Options", open=False):
                            vt_model_type = gr.Radio(
                                label="Model Type",
                                choices=[("VITON-HD (Recommended)", "viton_hd"),
                                         ("DressCode (Experimental)", "dress_code")],
                                value="viton_hd",
                            )
                            vt_garment_type = gr.Radio(
                                label="Garment Type", 
                                choices=[("Upper", "upper_body"),
                                         ("Lower", "lower_body"),
                                         ("Dress", "dresses")],
                                value="upper_body",
                            )
                            vt_ref_acceleration = gr.Radio(
                                label="Accelerate Reference UNet",
                                choices=[("True", True), ("False", False)],
                                value=False,
                            )
                            vt_step = gr.Number(
                                label="Inference Steps", minimum=30, maximum=100, step=1, value=30)
                            vt_scale = gr.Number(
                                label="Guidance Scale", minimum=0.1, maximum=5.0, step=0.1, value=2.5)
                            vt_seed = gr.Number(
                                label="Random Seed", minimum=-1, maximum=2147483647, step=1, value=42)

                vt_gen_button.click(
                    fn=self.predictor.leffa_predict_vt,
                    inputs=[vt_src_image, vt_ref_image, vt_ref_acceleration, vt_step, vt_scale, vt_seed, vt_model_type, vt_garment_type],
                    outputs=[vt_gen_image]
                )

            gr.Markdown(note)

            # Configure for running in Modal
            return demo.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                show_error=True,
                quiet=True
            ) 