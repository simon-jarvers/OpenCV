import cv2
import numpy as np
import argparse
import os


class Colors:
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    pink = (255, 0, 255)
    white = (255, 255, 255)


def draw_tracked_points(vid, tracked_points):
    if len(tracked_points) < 1:
        return
    else:
        for point in tracked_points:
            if point[2] == -1:  # going down
                cv2.circle(vid, (point[0], point[1]), 5, Colors.red, cv2.FILLED)
            elif point[2] == 1:  # going up
                cv2.circle(vid, (point[0], point[1]), 5, Colors.green, cv2.FILLED)
            else:  # neutral
                cv2.circle(vid, (point[0], point[1]), 5, Colors.pink, cv2.FILLED)


def append_tracked_points(x, y, tracked_points, tracked_lifts):
    current_lift = len(tracked_lifts) - 1
    threshold = 2
    if not tracked_points:
        tracked_points.append([x, y, 0])
    else:
        try:
            diff = y - tracked_points[-2][1]
        except IndexError:
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

    return tracked_points, tracked_lifts


def draw_box(vid, bbox):  # draw the given bounding box and return the center
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(vid, (x, y), (x+w, y+h), Colors.pink, 3, 1)
    cv2.putText(vid, "Tracking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return int(x+w/2), int(y+h/2)


def write_offset_score(vid, tracked_lift):  # display Offset and Score
    if not tracked_lift:
        return 0
    else:
        current_offset = tracked_lift[-1][0] - tracked_lift[0][0]
        cv2.putText(vid, "Offset: " + str(current_offset), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Colors.blue, 2)

        score = lift_score(tracked_lift)
        cv2.putText(vid, "Score: " + str(score), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Colors.blue, 2)
        return score


def lift_score(tracked_lift):
    score = 0
    mean = np.mean(tracked_lift, axis=0)
    for current in tracked_lift:
        score += (100 - (mean[0] - current[0])**2) / (len(tracked_lift))
    score = round(score, 2)
    return score


def track_lift(path_to_video, save_dir):

    video_dir, video_name = os.path.split(path_to_video)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = video_dir
    save_name = f'tracked_{video_name}'

    cap = cv2.VideoCapture(path_to_video)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    save_size = (frame_width, frame_height)
    save_video = cv2.VideoWriter(os.path.join(save_dir, save_name), cv2.VideoWriter_fourcc(*'mp4v'), 30, save_size)

    tracker = cv2.legacy_TrackerMOSSE.create()
    # tracker = cv2.legacy_TrackerCSRT.create()
    success, img = cap.read()  # read the first image for bbox
    bbox = cv2.selectROI("Please select the plates", img, False)  # get bbox from user input
    tracker.init(img, bbox)

    tracked_points = []
    tracked_lifts = [[]]

    while True:

        success_vid, vid = cap.read()

        if success_vid:

            success_tracker, bbox = tracker.update(vid)

            if success_tracker:
                x, y = draw_box(vid, bbox)
                tracked_points, tracked_lifts = append_tracked_points(x, y, tracked_points, tracked_lifts)
                draw_tracked_points(vid, tracked_points)
                write_offset_score(vid, tracked_lifts[-1])

            else:
                cv2.putText(vid, "Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Colors.red, 2)

            cv2.imshow("Tracking Lift", vid)
            img = vid
            save_video.write(vid)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    draw_tracked_points(img, tracked_points)
    temp_tracked_lifts = []
    for tracked_lift in tracked_lifts:
        if len(tracked_lift) > 10:
            temp_tracked_lifts.append(tracked_lift)
    tracked_lifts = temp_tracked_lifts
    del temp_tracked_lifts
    cv2.rectangle(img, (10, 10), (250, 30*len(tracked_lifts) + 25), Colors.white, cv2.FILLED)
    for i, tracked_lift in enumerate(tracked_lifts):
        score = lift_score(tracked_lift)
        cv2.putText(img, "Score " + str(i + 1) + ": " + str(score), (30, 30*(i+1) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, Colors.blue, 2)
    cv2.imshow("Tracked Life", img)
    save_video.write(img)
    cv2.waitKey(0)

    # cap.release()
    # cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Select the plates to track your lift')
    parser.add_argument('--video', help='Path to video', type=str, dest='path_to_video', required=True)
    parser.add_argument('--save_dir', help='Directory where results should be saved', type=str, dest='save_dir',
                        required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    track_lift(**vars(args))
