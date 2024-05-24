import mediapipe as mp
import numpy as np
import pygame
import random
import pyautogui
import cv2

def main():
    pygame.init()
    cap = cv2.VideoCapture(0)
    ret = True
    width = 640
    height = 480
    gameWidth = 800
    gameHeight = 600

    screen = pygame.display.set_mode((gameWidth, gameHeight))
    pygame.display.set_caption("Drag and Drop")
    # Create 5 draggable objects
    boxes = []
    for i in range(5):
        x = random.randint(50, 700)
        y = random.randint(50, 350)
        w = random.randint(35, 65)
        h = random.randint(35, 65)
        box = pygame.Rect(x, y, w, h)
        boxes.append(box)
    active_box = None

    hands = mp.solutions.hands
    hands_mesh = hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # This is to detect 2 hands
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    draw = mp.solutions.drawing_utils

    while ret:
        ret, frm = cap.read()
        if frm is not None:
            frm = cv2.resize(frm, (width, height))
            rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
            op = hands_mesh.process(rgb)

            if op.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(op.multi_hand_landmarks):
                    # 绘制手部标记
                    draw.draw_landmarks(frm, hand_landmarks, hands.HAND_CONNECTIONS)

                    # 检查手的类型（左手或右手）
                    handedness = op.multi_handedness[idx]
                    hand_type = handedness.classification[0].label  # 获取手的类型（'Left' 或 'Right'）
                    if hand_type == 'Left':
                        hand_type = 'Right'
                    else:
                        hand_type = 'Left'
                    print(hand_type)  # 打印手的类型
                    if hand_type == 'Right':
                        move_mouse(hand_landmarks)

                    # 可以在视频帧上显示手的类型
                    cv2.putText(frm, hand_type, (10, 50 * (idx + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)

                # Clear the screen
                screen.fill("white")

                # Draw the objects
                for box in boxes:
                    pygame.draw.rect(screen, (255, 0, 0), box)

                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            for num, box in enumerate(boxes):
                                if box.collidepoint(event.pos):
                                    active_box = num
                    if event.type == pygame.MOUSEMOTION:
                        if active_box != None:
                            boxes[active_box].move_ip(event.rel)
                    if event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:
                            active_box = None

                # Update the display
                pygame.display.flip()

                # Update the display

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                cap.release()
                pygame.quit()
                break
        else:
            print("Frame is empty. Check if the webcam is connected and working properly.")
            break


# def move_mouse(hand_landmarks):
#     screen_width, screen_height = pyautogui.size()
#     # Assuming landmark 4 is the tip of the thumb and landmark 2 is the base of the thumb
#     thumb_tip_left = hand_landmarks.landmark[8]
#     thumb_base_left = hand_landmarks.landmark[7]
#     thumb_tip_right = hand_landmarks.landmark[12]
#     thumb_base_right = hand_landmarks.landmark[11]
#     # Convert the hand position to screen position for the index fingertip
#     index_finger_tip = hand_landmarks.landmark[9]
#     screen_x = np.interp(index_finger_tip.x, [0, 1], [0, screen_width])
#     screen_y = np.interp(index_finger_tip.y, [0, 1], [0, screen_height])
#     # Move the mouse cursor
#     pyautogui.moveTo(screen_x, screen_y)
#     # Check if the thumb is bent downwards and click if it is
#     if thumb_tip_left.y > thumb_base_left.y:
#         pyautogui.click(button='left')
#     if thumb_tip_right.y > thumb_base_right.y:
#         pyautogui.click(button='right')

def move_mouse(hand_landmarks):
    screen_width, screen_height = pyautogui.size()
    # Assuming landmark 4 is the tip of the thumb and landmark 2 is the base of the thumb
    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[8]
    # Convert the hand position to screen position for the index fingertip
    index_finger_tip = hand_landmarks.landmark[9]
    screen_x = np.interp(index_finger_tip.x, [0, 1], [0, screen_width])
    screen_y = np.interp(index_finger_tip.y, [0, 1], [0, screen_height])
    # Move the mouse cursor
    pyautogui.moveTo(screen_x, screen_y)
    # Check if the thumb is bent downwards and click if it is
    distance = np.sqrt((thumb_tip.x - thumb_base.x) ** 2 + (thumb_tip.y - thumb_base.y) ** 2)
    if distance < 0.1:
        pyautogui.mouseDown()
    else:
        pyautogui.mouseUp()


if __name__ == "__main__":
    main()
