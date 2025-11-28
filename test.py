import cv2

def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: x={x}, y={y}")

image_path = "video_frame/00000.jpg"
image = cv2.imread(image_path)
cv2.imshow("Image", image)
cv2.setMouseCallback("Image", get_coordinates)
cv2.waitKey(0)
cv2.destroyAllWindows()