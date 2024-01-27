import cv2
def get_area(input_video):
    cap = cv2.VideoCapture(input_video)

    ret, frame = cap.read()
    if not ret:
        print('No frame found')

    roi = cv2.selectROI('roi_window', frame, showCrosshair=True, fromCenter=False)

    cap.release()
    cv2.destroyAllWindows()
    return roi