import cv2
import numpy as np

file_path = "venv/Resources/deadlift.mp4"
cap = cv2.VideoCapture(file_path)

save_width = int(cap.get(3))
save_height = int(cap.get(4))
save_size = (save_width, save_height)
save_video = cv2.VideoWriter('venv/Resources/tracked_deadlift.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, save_size)

tracker = cv2.legacy_TrackerMOSSE.create()
# tracker = cv2.legacy_TrackerCSRT.create()
success, img = cap.read()
bbox = cv2.selectROI("Deadlift", img, False)
tracker.init(img, bbox)

tracked_points = []
tracked_lifts = []


def draw_tracked_points(tracked_points):
    if len(tracked_points) < 5:
        return
    else:
        for point in tracked_points:
            if point[2] == -1: # going down
                cv2.circle(vid, (point[0], point[1]), 5, (0, 0, 255), cv2.FILLED)
            elif point[2] == 1: # going up
                cv2.circle(vid, (point[0], point[1]), 5, (0, 255, 0), cv2.FILLED)
            else: # neutral
                cv2.circle(vid, (point[0], point[1]), 5, (255, 0, 255), cv2.FILLED)


def append_tracked_points(x, y):
    threshhold = 2
    if not tracked_points:
        tracked_points.append([x, y, 0])
    else:
        diff = y - tracked_points[-1][1]
        if diff > threshhold: # going down, assign -1
            tracked_points.append([x, y, -1])
        elif diff < -threshhold: # going up, assign 1
            tracked_points.append([x, y, 1])
        else: # neutral, assign 0
            tracked_points.append([x, y, 0])


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
            append_tracked_points(x, y)
            draw_tracked_points(tracked_points)
            # find_one_lift(tracked_points)
            # print(x, y)
            save_video.write(vid)

        else:
            cv2.putText(vid, "Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Deadlift", vid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
