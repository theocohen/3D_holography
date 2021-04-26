import cv2
import numpy as np

def generate_video_from_focal_stack(focal_stack, image_shape, file):

    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fourcc = cv2.VideoWriter_fourcc(*'PIM1')
    out = cv2.VideoWriter(file, fourcc, 20, image_shape, False)

    for focal_plane in range(focal_stack.shape[2]):
        frame = np.uint8(255 * focal_stack[:, :, focal_plane])
        out.write(frame)
    out.release()


def read_video(file, delay_per_frame=1):
    while True:
        # This is to check whether to break the first loop
        isclosed = 0
        cap = cv2.VideoCapture(file)
        tot_nbr_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        while (True):

            ret, frame = cap.read()
            # It should only show the frame when the ret is true
            if ret == True:

                cv2.imshow('frame', frame)

                # display fram nbr fixme not working
                """
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_pos = (0, 0)
                fontScale = 20
                fontColor = (255, 255, 255)
                caption = 'Frame {}/{}'.format(cap.get(cv2.CAP_PROP_POS_FRAMES), tot_nbr_frames)
                cv2.putText(frame, caption, text_pos, font, fontScale, fontColor, bottomLeftOrigin=True)
                """
                if cv2.waitKey(delay_per_frame) == ord(' '):
                    # paused
                    if cv2.waitKey(0) == 27:
                        isclosed = 1
                        break
            else:
                break
        # To break the loop if it is closed manually
        if isclosed:
            break

    cap.release()
    cv2.destroyAllWindows()