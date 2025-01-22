from modal import Image, Stub, web_endpoint
import os

def download_models():
    from huggingface_hub import snapshot_download
    
    # Download Leffa models
    snapshot_download(
        repo_id="franciszzj/Leffa",
        local_dir="/root/models/Leffa"
    )
    
    # Download SCHP model for mask generation
    snapshot_download(
        repo_id="franciszzj/SCHP",
        local_dir="/root/models/SCHP"
    )

# Set up Modal image with all dependencies
image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "diffusers",
        "transformers",
        "huggingface_hub",
        "gradio==4.19.2",
        "numpy",
        "opencv-python",
        "Pillow",
        "torchvision",
        "scipy",
        "einops",
        "timm",
        "omegaconf"
    )
    .apt_install("git")
    .run_function(download_models)
)

stub = Stub("leffa", image=image)

@stub.cls(gpu="A100", container_idle_timeout=60*60)  # Keep container alive for 1 hour when idle
class Leffa:
    def __enter__(self):
        import sys
        import os
        import torch
        from omegaconf import OmegaConf
        
        # Set paths
        os.makedirs("/root/models", exist_ok=True)
        
        # Import required modules
        sys.path.append("/root/Leffa")
        from models import create_model
        from utils import get_transform
        
        # Initialize model and transforms
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = create_model().to(self.device)
        self.transform = get_transform()

    @web_endpoint()
    def index(self):
        """Create and launch the Gradio interface"""
        import gradio as gr
        
        def process_images(src_img, pose_img):
            src_tensor = self.transform(src_img).unsqueeze(0).to(self.device)
            pose_tensor = self.transform(pose_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(src_tensor, pose_tensor)
                
            return output[0]
        
        demo = gr.Interface(
            fn=process_images,
            inputs=[
                gr.Image(label="Source Image", type="pil"),
                gr.Image(label="Target Pose", type="pil")
            ],
            outputs=gr.Image(label="Generated Result"),
            title="Leffa: Person Image Generation",
            description="Upload a source person image and target pose to generate a new image."
        )
        
        # Configure for running in Modal
        return demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=True
        )

@stub.function(schedule=modal.Period(days=1))  # Restart daily to ensure freshness
def run_server():
    leffa = Leffa()
    leffa.run.remote()

if __name__ == "__main__":
    stub.run()