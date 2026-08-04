import cv2
import numpy as np
import os

# Create folder for saving palettes if it doesn't exist
output_folder = "saved_palettes"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Global variables for state management
captured_frame = None
display_data = False
saved_counter = 1
unique_colors_list = []

# Open the webcam (0 is default built-in camera)
cap = cv2.VideoCapture(0)

while True:
    if not display_data:
        # ----------------------------------------------------
        # 1. LIVE CAMERA MODE (WITH MIRROR EFFECT)
        # ----------------------------------------------------
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot access the webcam.")
            break
            
        # MIRROR EFFECT: Flip the frame horizontally for natural movement
        frame = cv2.flip(frame, 1)
        
        display_frame = frame.copy()
        height, width, _ = frame.shape
        
        # Maximize the box size (using 80% of the shortest screen dimension)
        box_size = int(min(width, height) * 0.8)
        start_x = (width - box_size) // 2
        start_y = (height - box_size) // 2
        end_x = start_x + box_size
        end_y = start_y + box_size
        
        # Draw target box (Red) with corner accents
        cv2.rectangle(display_frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
        leng = 20
        cv2.line(display_frame, (start_x, start_y), (start_x + leng, start_y), (0, 255, 0), 4)
        cv2.line(display_frame, (start_x, start_y), (start_x, start_y + leng), (0, 255, 0), 4)
        cv2.line(display_frame, (end_x, end_y), (end_x - leng, end_y), (0, 255, 0), 4)
        cv2.line(display_frame, (end_x, end_y), (end_x, end_y - leng), (0, 255, 0), 4)
        
        # ON-SCREEN TEXT IN GREEN (Live View)
        green_color = (0, 255, 0)
        cv2.putText(display_frame, "[S] - Capture & Analyze Box", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, green_color, 2, cv2.LINE_AA)
        cv2.putText(display_frame, "[P] - Save Palette Image", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, green_color, 2, cv2.LINE_AA)
        cv2.putText(display_frame, "[R] - Reset & Live Camera", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, green_color, 2, cv2.LINE_AA)
        cv2.putText(display_frame, "[Q] - Quit Application", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, green_color, 2, cv2.LINE_AA)
        
        cv2.imshow("Color Detector Ultra Pro", display_frame)
        
    else:
        # ----------------------------------------------------
        # 2. CAPTURED & DATA DISPLAY MODE (UI PANEL)
        # ----------------------------------------------------
        h, w, c = captured_frame.shape
        panel_width = 420
        canvas = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
        
        canvas[0:h, 0:w] = captured_frame
        canvas[0:h, w:w+panel_width] = (24, 20, 18)  # Premium Dark Background
        
        # UPDATED NEW APP TITLE HERE
        cv2.putText(canvas, "Smart-Color-Palette-Generator", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(canvas, (w + 20, 55), (w + panel_width - 20, 55), (50, 50, 50), 1)
        
        y_offset = 95
        for color_info in unique_colors_list: 
            r, g, b = color_info['rgb']
            hx = color_info['hex']
            h_v, s, v = color_info['hsv']
            pct = color_info['percentage']
            
            # Draw Color Block
            cv2.rectangle(canvas, (w + 20, y_offset - 20), (w + 55, y_offset + 10), (b, g, r), -1)
            cv2.rectangle(canvas, (w + 20, y_offset - 20), (w + 55, y_offset + 10), (100, 100, 100), 1)
            
            # Percentage Bar
            bar_start_x = w + 70
            bar_max_w = 100
            bar_end_x = bar_start_x + int((pct / 100) * bar_max_w)
            cv2.rectangle(canvas, (bar_start_x, y_offset - 15), (bar_start_x + bar_max_w, y_offset - 5), (50, 50, 50), -1)
            cv2.rectangle(canvas, (bar_start_x, y_offset - 15), (bar_end_x, y_offset - 5), (0, 255, 0), -1)
            
            # Labels
            pct_str = f"{pct:.1f}%"
            metrics_str = f"HEX: {hx} | RGB: ({r},{g},{b}) | HSV: ({h_v},{s},{v})"
            
            cv2.putText(canvas, pct_str, (w + 180, y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, metrics_str, (w + 70, y_offset + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
            
            cv2.line(canvas, (w + 20, y_offset + 20), (w + panel_width - 20, y_offset + 20), (35, 35, 35), 1)
            y_offset += 50
            
        # Footer Menu
        cv2.line(canvas, (w + 20, h - 50), (w + panel_width - 20, h - 50), (50, 50, 50), 1)
        cv2.putText(canvas, "[P] Save  |  [R] Live Cam  |  [Q] Quit", (w + 20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        cv2.imshow("Color Detector Ultra Pro", canvas)

    # Key event listener
    key = cv2.waitKey(1) & 0xFF
    
    # [S] - High-Accuracy K-Means Color Extraction
    if key == ord('s') and not display_data:
        captured_frame = frame.copy()
        roi = frame[start_y:end_y, start_x:end_x]
        
        # Reshape ROI to pixel list
        pixels = roi.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        # Define K-Means criteria (Find up to 7 dominant exact colors inside the box)
        n_colors = 7
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        
        # Apply K-Means clustering to get mathematically accurate color groups
        _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        
        # Calculate precise percentages for each color group
        labels = labels.flatten()
        counts = np.bincount(labels)
        total_pixels = len(pixels)
        
        # Sort centers by frequency
        sorted_indices = np.argsort(-counts)
        centers = centers[sorted_indices]
        counts = counts[sorted_indices]
        
        unique_colors_list = []
        for center, count in zip(centers, counts):
            # FIXED HERE: Correctly unpack the B, G, R array values from the center object
            b_val = int(center[0])
            g_val = int(center[1])
            r_val = int(center[2])
            
            percentage = (count / total_pixels) * 100
            
            # Skip color clusters that are negligible (under 1.5%)
            if percentage < 1.5:
                continue
                
            # Convert to HSV and Hex
            pixel_hsv = cv2.cvtColor(np.uint8([[ [b_val, g_val, r_val] ]]), cv2.COLOR_BGR2HSV)
            h_val = int(pixel_hsv[0][0][0])
            s_val = int(pixel_hsv[0][0][1])
            v_val = int(pixel_hsv[0][0][2])
            
            hex_val = '#{:02x}{:02x}{:02x}'.format(r_val, g_val, b_val)
            
            unique_colors_list.append({
                'rgb': (r_val, g_val, b_val),
                'hsv': (h_val, s_val, v_val),
                'hex': hex_val,
                'percentage': percentage
            })
            
        display_data = True
        print(f"[Captured] K-Means exact analysis done.")
        
    # [P] - Save the dashboard view
    elif key == ord('p') and display_data:
        file_name = f"{output_folder}/palette_{saved_counter}.png"
        cv2.imwrite(file_name, canvas)
        print(f"[Saved] Screen saved to {file_name}")
        saved_counter += 1
        
    # [R] - Clear dashboard and reset to live webcam feed
    elif key == ord('r'):
        display_data = False
        captured_frame = None
        unique_colors_list = []
        print("[Reset] Camera is live.")
        
    # [Q] - Safe close
    elif key == ord('q'):
        print("[Exit] Closing color detection environment.")
        break

cap.release()
cv2.destroyAllWindows()
