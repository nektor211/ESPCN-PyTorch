
# coding: utf-8

# In[1]:


# limitations under the License.
# ==============================================================================
import argparse
import os
import sys

import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from espcn_pytorch import ESPCN



# In[9]:


# config

args_cuda = True
args_scale_factor = 2
args_weights = f'weights/espcn_{args_scale_factor}x.pth'

division_factor = 0.5

if sys.argv[1] == '-f':
    video_name = 'sequences/buh_00_01_00.mp4'
else:
    video_name = sys.argv[1]
args_view = False
args_compare = False

cudnn.benchmark = True

# rest of code

if torch.cuda.is_available() and not args_cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

torch.set_num_threads(8)

device = torch.device("cuda:"+str(int(os.environ.get('index', '0'))%4) if args_cuda else "cpu")
print(video_name, '-', device)

# create model
model = ESPCN(scale_factor=args_scale_factor).to(device)

# Load state dicts
model.load_state_dict(torch.load(args_weights, map_location=device))

# Set model eval mode
model.eval()

# img preprocessing operation
pil2tensor = transforms.ToTensor()
tensor2pil = transforms.ToPILImage()

# Open video file

print(f"Reading `{os.path.basename(video_name)}`...")
video_capture = cv2.VideoCapture(video_name)
# Prepare to write the processed image into the video.
fps = video_capture.get(cv2.CAP_PROP_FPS)
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
# Set video size
size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
sr_size = (size[0] * args_scale_factor, size[1] * args_scale_factor)
pare_size = (sr_size[0] * 2 + 10, sr_size[1] + 10 + sr_size[0] // 5 - 9)
# Video write loader.
fn = f"out/{os.path.basename(video_name).strip('.mp4')}_{args_scale_factor}x"
print(fn)
srgan_writer = cv2.VideoWriter(f"{fn}_espcn.mp4",
                               cv2.VideoWriter_fourcc(*"MPEG"), fps, sr_size)
compare_writer = cv2.VideoWriter(f"{fn}_compare.mp4",
                                 cv2.VideoWriter_fourcc(*"MPEG"), fps, pare_size)
sidebyside_writer = cv2.VideoWriter(f"{fn}_side_by_side.mp4",
                               cv2.VideoWriter_fourcc(*"MPEG"), fps, sr_size)

def merge(a, b, split=0.5):
    h = int(a.shape[1]*split)
    a[:,h:,:] = b[:,h:,:]
    a[:,h-1:h,:] = 255

# read frame
success, raw_frame = video_capture.read()
print('success', success)
progress_bar = tqdm(range(total_frames), desc="[processing video and saving/view result videos]")
for index in progress_bar:
    if success:
        img = Image.fromarray(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)).convert("YCbCr")
        y, cb, cr = img.split()
        img = pil2tensor(y).view(1, -1, y.size[1], y.size[0])
        img = img.to(device)

        with torch.no_grad():
            prediction = model(img)

        prediction = prediction.cpu()
        sr_frame_y = prediction[0].detach().numpy()
        sr_frame_y *= 255.0
        sr_frame_y = sr_frame_y.clip(0, 255)
        sr_frame_y = Image.fromarray(np.uint8(sr_frame_y[0]), mode="L")

        sr_frame_cb = cb.resize(sr_frame_y.size, Image.BICUBIC)
        sr_frame_cr = cr.resize(sr_frame_y.size, Image.BICUBIC)
        sr_frame = Image.merge("YCbCr", [sr_frame_y, sr_frame_cb, sr_frame_cr]).convert("RGB")
        # before converting the result in RGB
        sr_frame = cv2.cvtColor(np.asarray(sr_frame), cv2.COLOR_RGB2BGR)
        # save sr video
        srgan_writer.write(sr_frame)

        upscaled_img = cv2.resize(raw_frame, sr_size)
        #merge(upscaled_img, sr_frame, np.sin(index/40)/5+0.5)
        merge(upscaled_img, sr_frame, division_factor)
        sidebyside_writer.write(upscaled_img)
        if args_compare:
        
            # make compared video and crop shot of left top\right top\center\left bottom\right bottom
            sr_frame = tensor2pil(sr_frame)
            # Five areas are selected as the bottom contrast map.
            crop_sr_imgs = transforms.FiveCrop(size=sr_frame.width // 5 - 9)(sr_frame)
            crop_sr_imgs = [np.asarray(transforms.Pad(padding=(10, 5, 0, 0))(img)) for img in crop_sr_imgs]
            sr_frame = transforms.Pad(padding=(5, 0, 0, 5))(sr_frame)
            # Five areas in the contrast map are selected as the bottom contrast map
            compare_img = transforms.Resize((sr_size[1], sr_size[0]), interpolation=Image.BICUBIC)(tensor2pil(raw_frame))
            crop_compare_imgs = transforms.FiveCrop(size=compare_img.width // 5 - 9)(compare_img)
            crop_compare_imgs = [np.asarray(transforms.Pad(padding=(0, 5, 10, 0))(img)) for img in crop_compare_imgs]
            compare_img = transforms.Pad(padding=(0, 0, 5, 5))(compare_img)
            # concatenate all the pictures to one single picture
            # 1. Mosaic the left and right images of the video.
            top_img = np.concatenate((np.asarray(compare_img), np.asarray(sr_frame)), axis=1)
            # 2. Mosaic the bottom left and bottom right images of the video.
            bottom_img = np.concatenate(crop_compare_imgs + crop_sr_imgs, axis=1)
            bottom_img_height = int(top_img.shape[1] / bottom_img.shape[1] * bottom_img.shape[0])
            bottom_img_width = top_img.shape[1]
            # 3. Adjust to the right size.
            bottom_img = np.asarray(transforms.Resize((bottom_img_height, bottom_img_width))(tensor2pil(bottom_img)))


            # 4. Combine the bottom zone with the upper zone.
            final_image = np.concatenate((top_img, bottom_img))

            compare_writer.write(final_image)

        if args_view:
            # display video
            cv2.imshow("LR video convert HR video ", final_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    # next frame
    success, raw_frame = video_capture.read()
sidebyside_writer.release()
compare_writer.release()
srgan_writer.release()
print('done')

