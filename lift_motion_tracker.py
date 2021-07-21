import cv2
import numpy as np

file_path = "venv/Resources/deadlift.mp4"
cap = cv2.VideoCapture(file_path)

tracker = cv2.legacy_TrackerMOSSE.create()
# tracker = cv2.legacy_TrackerCSRT.create()
success, img = cap.read()
bbox = cv2.selectROI("Deadlift", img, False)
tracker.init(img, bbox)

tracked_points = []


def draw_tracked_points(tracked_points):
    for point in tracked_points:
        cv2.circle(vid, (point[0], point[1]), 5, (255, 0, 255), cv2.FILLED)


def draw_box(vid, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(vid, (x, y), (x+w, y+h), (255, 0, 255), 3, 1)
    cv2.putText(vid, "Tracking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return int(x+w/2), int(y+h/2)


while True:

    success_vid, vid = cap.read()

    if success_vid:

        # timer = cv2.getTickCount()
        # fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
        # cv2.putText(vid, str(int(fps)), (50, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        success_tracker, bbox = tracker.update(vid)

        if success_tracker:
            x, y = draw_box(vid, bbox)
            cv2.circle(vid, (x, y), 5, (255, 0, 255), cv2.FILLED)
            tracked_points.append([x, y])
            draw_tracked_points(tracked_points)
            print(x, y)

        else:
            cv2.putText(vid, "Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Deadlift", vid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
