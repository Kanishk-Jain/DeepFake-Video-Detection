import os
import shutil

# Source folders - "Real" and "Fake"
source_base = '.'
categories = ['Real', 'Fake']

# Destination path
dest_base = 'frames'

for category in categories:
    source_path = os.path.join(source_base, category)
    dest_path = os.path.join(dest_base, category.lower())  # real, fake in lowercase

    os.makedirs(dest_path, exist_ok=True)

    print(f"📂 Processing {category}...")

    for video_folder in os.listdir(source_path):
        video_folder_path = os.path.join(source_path, video_folder)

        if os.path.isdir(video_folder_path):
            for frame_file in os.listdir(video_folder_path):
                if frame_file.lower().endswith('.png'):  # <- Corrected here
                    src = os.path.join(video_folder_path, frame_file)
                    dest_name = f"{video_folder}_{frame_file}"
                    dest = os.path.join(dest_path, dest_name)
                    shutil.copyfile(src, dest)

print("\n✅ Flattening complete. Images saved in frames/real and frames/fake.")
