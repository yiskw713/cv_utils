import argparse
import glob
import os

import cv2


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="")
    parser.add_argument("--img_dir", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="./result")
    return parser.parse_args()


def crop_bbox(image_path: str, save_dir: str) -> None:
    img = cv2.imread(image_path)
    name = os.path.splitext(os.path.basename(image_path))[0]

    # count the number of saved images from the image
    cnt = len(glob.glob(os.path.join(save_dir, f"{name}*.jpg")))

    while True:
        # select region of interest (bounding box)
        ROI = cv2.selectROI(
            "Please draw a bounding box", img, fromCenter=False, showCrosshair=False
        )

        left, top, width, height = map(int, ROI)

        # check if a bbox is valid or not
        if width == 0 or height == 0:
            break

        # cropping image
        cropped_img = img[top : top + height, left : left + width]
        cv2.imshow("crop", cropped_img)

        while True:
            k = cv2.waitKey(0)
            if k == ord("r"):
                # retry selecting bboxes
                cv2.destroyWindow("crop")
                break
            elif k == ord("s"):
                # save cropped images
                cnt += 1
                save_path = os.path.join(save_dir, f"{name}{cnt:0>3}.jpg")
                cv2.imwrite(save_path, cropped_img)
                cv2.destroyWindow("crop")
                break


def main():
    args = get_arguments()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.img_path:
        crop_bbox(args.img_path, args.save_dir)
    elif args.img_dir:
        image_paths = glob.glob(os.path.join(args.img_dir, "*.jpg"))
        image_paths += glob.glob(os.path.join(args.img_dir, "*.JPG"))
        image_paths += glob.glob(os.path.join(args.img_dir, "*.jpeg"))
        image_paths += glob.glob(os.path.join(args.img_dir, "*.png"))

        for path in image_paths:
            crop_bbox(path, args.save_dir)
    else:
        raise ValueError("Either --img_path or --img_dir must be specified.")


if __name__ == "__main__":
    main()
