import cv2
import numpy as np

file_path = "venv/Resources/deadlift.mp4"
cap = cv2.VideoCapture(file_path)


save_size = (int(cap.get(3)), int(cap.get(4)))
save_video = cv2.VideoWriter('venv/Resources/tracked_deadlift.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, save_size)

tracker = cv2.legacy_TrackerMOSSE.create()
# tracker = cv2.legacy_TrackerCSRT.create()
success, img = cap.read()
bbox = cv2.selectROI("Deadlift", img, False)
tracker.init(img, bbox)

tracked_points = []
tracked_lifts = [[]]
current_lift = 0


def draw_tracked_points(vid, tracked_points):
    if len(tracked_points) < 5:
        return
    else:
        for point in tracked_points:
            if point[2] == -1:  # going down
                cv2.circle(vid, (point[0], point[1]), 5, (0, 0, 255), cv2.FILLED)
            elif point[2] == 1:  # going up
                cv2.circle(vid, (point[0], point[1]), 5, (0, 255, 0), cv2.FILLED)
            else:  # neutral
                cv2.circle(vid, (point[0], point[1]), 5, (255, 0, 255), cv2.FILLED)


def append_tracked_points(x, y):
    global current_lift
    threshold = 2
    if not tracked_points:
        tracked_points.append([x, y, 0])
    else:
        diff = y - tracked_points[-1][1]
        if diff > threshold:  # going down, assign -1
            tracked_points.append([x, y, -1])
            tracked_lifts[current_lift].append([x, y, -1])
        elif diff < -threshold:  # going up, assign 1
            tracked_points.append([x, y, 1])
            tracked_lifts[current_lift].append([x, y, 1])
        else:  # neutral, assign 0
            tracked_points.append([x, y, 0])
        if tracked_lifts[current_lift]:
            if tracked_points[-1][2] != tracked_lifts[-1][-1][2]:
                current_lift += 1
                tracked_lifts.append([])


def draw_box(vid, bbox):  # draw the given boung box and return the center
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(vid, (x, y), (x+w, y+h), (255, 0, 255), 3, 1)
    cv2.putText(vid, "Tracking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return int(x+w/2), int(y+h/2)


def lift_offset(vid, tracked_lifts):
    if not tracked_lifts[current_lift]:
        return
    else:
        current_offset = tracked_lifts[current_lift][-1][0] - tracked_lifts[current_lift][0][0]
        cv2.putText(vid, "Offset: " + str(current_offset), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        score = 0
        for lift in tracked_lifts[current_lift]:
            score += (lift[0] - tracked_lifts[current_lift][0][0])**2/len(tracked_lifts[current_lift])
        score = round(score, 2)
        cv2.putText(vid, "Score: " + str(score), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


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
            draw_tracked_points(vid, tracked_points)
            lift_offset(vid, tracked_lifts)

        else:
            cv2.putText(vid, "Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Deadlift", vid)
        save_video.write(vid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap = cv2.VideoCapture(file_path)
success, img = cap.read()
draw_tracked_points(img, tracked_points)
cv2.imshow("Lifts", img)
cv2.waitKey(0)

# cap.release()
# cv2.destroyAllWindows()
