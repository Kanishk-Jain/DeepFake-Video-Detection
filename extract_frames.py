import cv2
import os

# Base directories
input_dir = 'videos'
output_dir = 'frames'

# Categories
categories = ['real', 'deepfake']

# Frames to extract per video
frames_per_video = 10

for category in categories:
    input_path = os.path.join(input_dir, category)
    output_path = os.path.join(output_dir, category)

    # Ensure output folder exists
    os.makedirs(output_path, exist_ok=True)

    print(f"\n🔍 Processing category: {category}")
    
    for video_file in os.listdir(input_path):
        video_path = os.path.join(input_path, video_file)
        print(f"📁 Found video: {video_file}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Could not open video: {video_file}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(total_frames // frames_per_video, 1)

        print(f"🎞️ Total frames in {video_file}: {total_frames}")
        print(f"📸 Capturing every {frame_interval} frames")

        count = 0
        saved = 0

        while cap.isOpened() and saved < frames_per_video:
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ Finished reading early: {video_file}")
                break

            if count % frame_interval == 0:
                frame_name = f"{os.path.splitext(video_file)[0]}_frame{saved}.jpg"
                frame_path = os.path.join(output_path, frame_name)
                cv2.imwrite(frame_path, frame)
                print(f"✅ Saved frame {saved + 1} → {frame_path}")
                saved += 1

            count += 1

        cap.release()
        print(f"🎉 Done processing {video_file}")

print("\n✅ All frame extraction complete!")
