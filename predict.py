import cog
import os
import shutil
from tempfile import TemporaryDirectory
from cog import Path

class AMTModel(cog.BasePredictor):
    def setup(self):
        # Load the model into memory for efficient multiple predictions
        if not os.path.exists('AMT'):
            os.system('git clone https://github.com/MCG-NKU/AMT.git')
        
        os.makedirs('pretrained', exist_ok=True)
        if not os.path.exists('pretrained/amt-s.pth'):
            os.system('gdown 1WmOKmQmd6pnLpID8EpUe-TddFpJuavrL -O pretrained/amt-s.pth')
        if not os.path.exists('pretrained/amt-l.pth'):
            os.system('gdown 1UyhYpAQLXMjFA55rlFZ0kdiSVTL7oU-z -O pretrained/amt-l.pth')
        if not os.path.exists('pretrained/amt-g.pth'):
            os.system('gdown 1yieLtKh4ei3gOrLN1LhKSP_9157Q-mtP -O pretrained/amt-g.pth')

    def predict(
        self, 
        video: Path,  # Input video file
        model_type: str = "amt-l",  # Model type, default "amt-l"
        recursive_interpolation_passes: int = 2,  # Number of recursive interpolation passes
        output_video_fps: int = 16, # Output video FPS,
    ) -> Path:
        with TemporaryDirectory() as tmpdir:
            inputs_dir = os.path.join(tmpdir, "frames")
            outputs_dir = os.path.join(tmpdir, "results")
            os.makedirs(inputs_dir, exist_ok=True)

            # Extract frames from the video
            fps_filter = " -vf mpdecimate,setpts=N/FRAME_RATE/TB"
            os.system(f'ffmpeg -i "{video}" {fps_filter} {inputs_dir}/frame%04d.png')

            # Run the AMT interpolation model
            model_type_upper = model_type.upper()
            # change directory to AMT
            os.chdir('/src/AMT')
            os.system(f'python demos/demo_2x.py --config cfgs/{model_type_upper}.yaml --ckpt ../pretrained/{model_type}.pth --niters {recursive_interpolation_passes} --input {inputs_dir} --out_path {outputs_dir} --frame_rate {output_video_fps}')

            # Return the output video
            output_video = os.path.join(outputs_dir, "demo_0000.mp4")

            # create a unique filename mp4 in /tmp
            import uuid
            out_file_1 = f'/tmp/{uuid.uuid4()}.mp4'
            out_file_2 = f'/tmp/{uuid.uuid4()}.mp4'
            shutil.copy(output_video, out_file_1)  # Copy to a non-temporary location
            
            # convert video to an mp4 format that web browsers can play
            os.system(f'ffmpeg -i "{out_file_1}" -vcodec libx264 "{out_file_2}"')
            
            return Path(out_file_2)
