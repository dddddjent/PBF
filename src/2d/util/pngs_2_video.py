"""
python util/pngs_2_video.py png_dir png_prefix [fps] [video_name]
python util/pngs_2_video.py logs/leapfrog-ed/vort w
"""

import os
import sys


def concatenate_pngs_to_video(png_dir, png_prefix, fps=10, video_name="output.mp4"):
    import imageio

    images = []
    for png in os.listdir(png_dir):
        if png.startswith(png_prefix):
            images.append(os.path.join(png_dir, png))
    images = sorted(images)

    video_name = os.path.join(png_dir, video_name)
    i = 0
    with imageio.get_writer(video_name, fps=fps) as writer:
        for img_path in images:
            i += 1
            if i % fps == 0:
                print(f"processing {i}th images")
            image = imageio.imread(img_path)
            writer.append_data(image)


png_dir = sys.argv[1]
png_prefix = sys.argv[2]

fps = 10
if len(sys.argv) >= 4:
    fps = int(sys.argv[3])
video_name = "output.mp4"
if len(sys.argv) >= 5:
    video_name = sys.argv[4]

concatenate_pngs_to_video(png_dir, png_prefix, fps, video_name)
