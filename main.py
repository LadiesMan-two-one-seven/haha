import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
)

mp_draw = mp.solutions.drawing_utils
video_path = "cat.mp4"
photo_path = "photo.jpeg"
cap_cat = None
cap = cv2.VideoCapture(0)
is_video_playing = False
is_photo_showing = False
photo = cv2.imread(photo_path)

finger_tip_base_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]


def is_fist_closed(lm):
    return all(lm[tip].y > lm[base].y for tip, base in finger_tip_base_pairs)


def is_palm_open(lm):
    return all(lm[tip].y < lm[base].y for tip, base in finger_tip_base_pairs)


def safe_destroy_window(name):
    try:
        cv2.destroyWindow(name)
    except cv2.error:
        pass


while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    open_palms = 0
    is_fist_present = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark

            if is_fist_closed(lm):
                is_fist_present = True
            elif is_palm_open(lm):
                open_palms += 1

    should_show_photo = (open_palms == 2 and not is_fist_present)

    if should_show_photo and not is_photo_showing:
        is_photo_showing = True
        print("photo start showing")
    elif not should_show_photo and is_photo_showing:
        is_photo_showing = False
        safe_destroy_window("Photo")
        print("photo stop showing")

    if is_fist_present and not is_video_playing:
        cap_cat = cv2.VideoCapture(video_path)
        if cap_cat.isOpened():
            is_video_playing = True
            print("cat start dancing")
    elif not is_fist_present and is_video_playing:
        is_video_playing = False
        if cap_cat:
            cap_cat.release()
            cap_cat = None
        safe_destroy_window("Scuba-cat")
        print("cat stop dancing")

    if is_photo_showing and photo is not None:
        photo_resized = cv2.resize(photo, (400, 400))
        cv2.imshow("Photo", photo_resized)
    elif is_video_playing and cap_cat is not None:
        ret_v, frame_cat = cap_cat.read()
        if not ret_v:
            cap_cat.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_v, frame_cat = cap_cat.read()
        if ret_v:
            frame_cat = cv2.resize(frame_cat, (400, 400))
            cv2.imshow("Scuba-cat", frame_cat)
    cv2.imshow("camera", frame)

    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
if cap_cat: cap_cat.release()
cv2.destroyAllWindows()