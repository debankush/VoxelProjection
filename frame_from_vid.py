import cv2
import os

def extract_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    
    frame_rate = 24
    frame_count = 0
    
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    
    output_directory = f"vid4_frames"
    os.makedirs(output_directory, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % int(cap.get(5) / frame_rate) == 0:
            output_file = f"{output_directory}/frame_{frame_count}.jpg"
            cv2.imwrite(output_file, frame)
            print(f"Frame {frame_count} has been extracted and saved as {output_file}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file = r"vid4.mp4"  
    
    extract_frames(video_file)