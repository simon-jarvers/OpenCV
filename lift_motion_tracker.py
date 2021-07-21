import cv2
import numpy as np

file_path = "venv/Resources/deadlift.mp4"
cap = cv2.VideoCapture(file_path)

tracker = cv2.legacy_TrackerMOSSE.create()
# tracker = cv2.legacy_TrackerCSRT.create()
success, img = cap.read()
bbox = cv2.selectROI("Deadlift", img, False)
tracker.init(img, bbox)


def draw_box(vid, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(vid, (x, y), (x+w, y+h), (255, 0, 255), 3, 1)
    cv2.putText(vid, "Tracking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


while True:

    success_vid, vid = cap.read()

    if success_vid:

        # timer = cv2.getTickCount()
        # fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
        # cv2.putText(vid, str(int(fps)), (50, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        success_tracker, bbox = tracker.update(vid)

        if success_tracker:
            draw_box(vid, bbox)
        else:
            cv2.putText(vid, "Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Deadlift", vid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
