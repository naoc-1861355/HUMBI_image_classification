import os
import cv2
import glob


def hand_label(dir1, sub_list):
    for sub in os.listdir(dir1):
        if sub in sub_list:
            sub = dir1 + sub
            if os.path.isdir(sub):
                hand_dir = sub + '/hand/'
                for d in os.listdir(hand_dir):
                    if int(d) >= 240:
                        in_dir = hand_dir + d
                        if os.path.isdir(in_dir):
                            print(in_dir[-5:])
                            generate_label(in_dir, True)
                            generate_label(in_dir, False)


def generate_label(in_dir, is_left):
    if is_left:
        in_path = in_dir + '/image_cropped/left/'
    else:
        in_path = in_dir + '/image_cropped/right/'
    images = glob.glob(in_path + '*.png')
    if len(images) > 0:
        k = 0
        while k != 13:
            for image in images:
                img = cv2.imread(image)
                cv2.imshow(in_dir[-4:], img)
                cv2.waitKey(100)
            k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        text = input("prompt")
        with open(in_path + 'label.txt', 'w') as file:
            file.write(text)
        print(text)


def main():
    dir1 = 'D:\Hand-data/HUMBI/Hand_1_80_updat/'
    sublist = ['subject_3']
    hand_label(dir1, sublist)


if __name__ == '__main__':
    main()
