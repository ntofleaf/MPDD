import os
import glob
import cv2


class CopyImage:

    def __init__(self, root_dir, img_type="bmp", num_img=10000):
        self.root_dir = root_dir
        self.img_type = img_type
        self.num_img = num_img
    
    def select_images(self, set_name):
        data_split = os.path.join(self.root_dir, set_name)
        img_dirs = os.listdir(data_split)
        for img_dir in img_dirs:
            if "aligned" in img_dir:
                print(img_dir)
                img_files = glob.glob(os.path.join(data_split, img_dir, f"*.{self.img_type}"))
                img_files.sort()
                sel_imgs = img_files[: self.num_img]
                if len(sel_imgs) != self.num_img:
                    # raise ValueError(f"Number of images is not enough: {len(sel_imgs)}")
                    print(f"img_dir: {img_dir}")
                    print(f"Number of images is not enough: {len(sel_imgs)}")

                for img_file in sel_imgs:
                    img_arr = cv2.imread(img_file)
                    save_to = img_file.replace("avec", "avec_2013").replace(".bmp", ".jpg")
                    # print(save_to)
                    save_dir = os.path.dirname(save_to)
                    os.makedirs(save_dir, exist_ok=True)
                    cv2.imwrite(save_to, img_arr)

    def check(self, set_name):
        data_split = os.path.join(self.root_dir, set_name)
        img_dirs = os.listdir(data_split)
        print(f"Number of img_dirs: {len(img_dirs)}")
        for img_dir in img_dirs:
            if "aligned" in img_dir:
                print(img_dir)
                img_files = glob.glob(os.path.join(data_split, img_dir, f"*.{self.img_type}"))
                img_files.sort()
                print(f"Number of images: {len(img_files)}")

        


if __name__ == "__main__":
    handler = CopyImage("data/avec_2013", img_type="jpg")
    # handler.select_images("Training")
    # handler.select_images("Testing")
    # handler.select_images("Development")
    # handler.check("Training")
    # handler.check("Testing")
    handler.check("Development")
 