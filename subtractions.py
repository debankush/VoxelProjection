import cv2
import os

output_directory = f"vid4_subtractions"
os.makedirs(output_directory, exist_ok=True)

frame_counter = 1
counter = 1
img1_name = str("C:\\Users\\deban\\Documents\\Projects\\VoxelProjection\\vid4_frames\\frame_"+str(counter)+".jpg")
img1 = cv2.imread(img1_name)

counter += 1

img2_name = str("C:\\Users\\deban\\Documents\\Projects\\VoxelProjection\\vid4_frames\\frame_"+str(counter)+".jpg")
img2 = cv2.imread(img2_name)

output_file = f"{output_directory}/frame_{frame_counter}.jpg"
cv2.imwrite(output_file, cv2.cvtColor(cv2.subtract(img1,img2), cv2.COLOR_BGR2GRAY))
frame_counter += 1

final_img = cv2.subtract(img1, img2)

for i in range (2, 240):
    img1_name = str("C:\\Users\\deban\\Documents\\Projects\\VoxelProjection\\vid4_frames\\frame_"+str(counter)+".jpg")
    img1 = cv2.imread(img1_name)
    counter += 1
    img2_name = str("C:\\Users\\deban\\Documents\\Projects\\VoxelProjection\\vid4_frames\\frame_"+str(counter)+".jpg")
    img2 = cv2.imread(img2_name)
    output_file = f"{output_directory}/frame_{frame_counter}.jpg"
    cv2.imwrite(output_file, cv2.cvtColor(cv2.subtract(img1,img2), cv2.COLOR_BGR2GRAY))
    frame_counter += 1
    print(f"Frame {frame_counter} has been extracted and saved as {output_file}")
    final_img = cv2.add(final_img, cv2.subtract(img1, img2))

cv2.imwrite(output_file, cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY))