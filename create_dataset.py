import cv2
import imutils
import numpy as np
import os

# global variables
bg = None


def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return None
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def main():
    aWeight = 0.5
    camera = cv2.VideoCapture(0)

    # ROI coordinates
    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0
    image_num = 0
    start_recording = False

    save_path = "D:/Hafsah/cnn dataset/flipTest"
    os.makedirs(save_path, exist_ok=True)

    while True:
        (grabbed, frame) = camera.read()

        if not grabbed:
            print("[Warning!] Error input, please check your camera.")
            break

        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        roi = frame[top:bottom, right:left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 30:
            run_avg(gray, aWeight)
            print(f"Calibrating background... Frame {num_frames}")
        else:
            hand = segment(gray)

            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                if start_recording:
                    filename = os.path.join(save_path, f"fist_{image_num}.png")
                    cv2.imwrite(filename, thresholded)
                    print(f"[INFO] Saved image {image_num} to {filename}")
                    image_num += 1
                else:
                    print("[INFO] Hand detected, but not recording yet. Press 's' to start.")

                cv2.imshow("Thresholded", thresholded)
            else:
                print("[INFO] No hand detected in this frame.")

        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
        num_frames += 1
        cv2.imshow("Video Feed", clone)

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q") or image_num > 100:
            print("[INFO] Exiting program.")
            break

        if keypress == ord("s"):
            start_recording = True
            print("[INFO] Recording started.")

    camera.release()
    cv2.destroyAllWindows()


main()
