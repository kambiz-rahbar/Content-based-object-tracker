import cv2
import Alex_CBComp as cb
import torch
from torchvision.transforms import transforms
import numpy as np

# Function to select the region of interest (ROI)
def select_roi(image):
    roi = cv2.selectROI("Select Area", image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    return roi

# Load the video file
video_path = 'big_buck_bunny_720p_5mb.mp4'
cap = cv2.VideoCapture(video_path)

# define preprocessing functions
transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ConvertImageDtype(torch.float),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])                                
device = cb.select_device()
feature_extractor = cb.FeatureExtractor().to(device)                                               

# Set region radius for the search area
region_radius = 10

frame_counter = 0
while True:
    ret, frame = cap.read()  # Read video frames
    if not ret:
        break  # Exit the loop if no more frames

    # Convert BGR (OpenCV default) to RGB for displaying using matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_counter == 0:
        roi = select_roi(frame_rgb)   # Get user rectangle area

        # Extract the region of interest (ROI) from the frame
        x, y, w, h = roi
        selected_area = frame_rgb[y:y+h, x:x+w]

        # Save the selected area
        cv2.imwrite('frames/selected_area.png', cv2.cvtColor(selected_area, cv2.COLOR_RGB2BGR))

        # Extract the feature vector from ROI
        query_gain = np.transpose(selected_area, (2, 0, 1))
        query_gain = torch.from_numpy(query_gain).to(torch.uint8)

        query_gain = transform(query_gain)
        query_batch = torch.reshape(query_gain,[1,3,224,224])
        query_fv = feature_extractor.extract(query_batch.to(device))

        # Calculate Pascal VOC format (xmin, ymin, xmax, ymax)
        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h

        # Print Pascal VOC coordinates
        print(f"Pascal VOC format: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

        # Save the Pascal VOC coordinates to a text file
        with open('frames/coordinates.txt', 'w') as f:
                f.write(f"Pascal VOC format: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}\n")
    else:
        search_area = (max(xmin-region_radius,0), max(ymin-region_radius,0),
                       min(xmin+region_radius, frame.shape[0]), min(ymin+region_radius,frame.shape[1]))

        batched_win = []
        for r in range(search_area[0],search_area[2], 2):
             for c in range(search_area[1],search_area[3], 2):
                  win = frame_rgb[r:r+h,c:c+w]
                  batched_win.append(win)

        preprocess_win = np.transpose(batched_win, (0, 3, 1, 2))
        preprocess_win = torch.from_numpy(preprocess_win).to(torch.uint8)
        stacked_win = transform(preprocess_win)

        win_fv = feature_extractor.extract(stacked_win.to(device))
        
        # update query for the next frame
        gain_idx, gain_dist = cb.distance(query_fv, win_fv)
        query_batch = batched_win[gain_idx]
        query_fv = win_fv[gain_idx]
        
        # Save the query
        cv2.imwrite(f'frames/{frame_counter}-{gain_idx}-{gain_dist:.2f}.png', cv2.cvtColor(batched_win[gain_idx], cv2.COLOR_RGB2BGR))
        
        # Save the Pascal VOC coordinates to a text file
        with open('frames/coordinates.txt', 'a') as f:
             f.write(f"Pascal VOC format: xmin={max(xmin-region_radius,0)+region_radius*gain_idx}, ymin={max(ymin-region_radius,0)+region_radius*gain_idx}, xmax={max(xmin-region_radius,0)+region_radius*gain_idx+w}, ymax={max(ymin-region_radius,0)+region_radius*gain_idx+h}\n")
        
    print(f'frame {frame_counter} completed.')
    frame_counter += 1   # Increment frame counter

# Release the video
cap.release()
