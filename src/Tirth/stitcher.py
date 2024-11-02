import cv2
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt


# References
# - https://github.com/opencv/opencv/blob/4.x/samples/python/stitching_detailed.py
# - https://docs.opencv.org/2.4.13.7/modules/stitching/doc/high_level.html?highlight=stitcher#stitcher-stitch
# - https://docs.opencv.org/2.4.13.7/modules/stitching/doc/introduction.html
# - https://www.morethantechnical.com/blog/2018/10/30/cylindrical-image-warping-for-panorama-stitching/
# - https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html


class PanoramaStitcher:
    idx = 0
    size_threshold = 1000

    def __init__(self):
        # Use SIFT as feature detector
        self.sift = cv2.SIFT_create()
        # Initialize FLANN based matcher
        index_params = dict(algorithm=1, trees=5)  # KDTree algorithm
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def cylindricalWarp(self, img):
        """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
        # auto determine K
        h_, w_ = img.shape[:2]

        # i have tried to find optimal way to calculate focal length
        # using sigmoid function to calculate alpha
        # f_ = 0.9 * w_  # guess for focal length
        alpha = 2 / (1.1 + np.exp(-0.0008 * w_))
        # f_ = np.tanh(w_) * w_
        f_ = alpha * w_
        # print(f"focal length: {f_}, width: {w_}, alpha: {alpha}")
        K = np.array([[f_, 0, w_ / 2], [0, f_, h_ / 2], [0, 0, 1]])
        # pixel coordinates
        y_i, x_i = np.indices((h_, w_))
        X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(
            h_ * w_, 3
        )  # to homog
        Kinv = np.linalg.inv(K)
        X = Kinv.dot(X.T).T  # normalized coords
        # calculate cylindrical coords (sin\theta, h, cos\theta)
        A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(
            w_ * h_, 3
        )
        B = K.dot(A.T).T  # project back to image-pixels plane
        # back from homog coords
        B = B[:, :-1] / B[:, [-1]]
        # make sure warp coords only within image bounds
        B[(B[:, 0] < 0) | (B[:, 0] >= w_) | (B[:, 1] < 0) | (B[:, 1] >= h_)] = -1
        B = B.reshape(h_, w_, -1)

        # img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # for transparent borders...
        # # warp the image according to cylindrical coords
        # return cv2.remap(
        #     img_rgba,
        #     B[:, :, 0].astype(np.float32),
        #     B[:, :, 1].astype(np.float32),
        #     cv2.INTER_AREA,
        #     borderMode=cv2.BORDER_TRANSPARENT,
        # )
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # return cv2.remap(
        #     img_rgb,
        #     B[:, :, 0].astype(np.float32),
        #     B[:, :, 1].astype(np.float32),
        #     cv2.INTER_AREA,
        #     borderMode=cv2.BORDER_CONSTANT,
        # )
        return cv2.remap(
            img,
            B[:, :, 0].astype(np.float32),
            B[:, :, 1].astype(np.float32),
            cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def stitch_images(self, images):
        # # Initialize the panorama as the first image
        # panorama = images[0]
        # # Calculate homographies and warp each image
        # for i in range(1, len(images)):
        #     print(f"Stitching image {i} to the panorama.")
        #     panorama = self.stitch_pair(panorama, images[i])
        # return panorama

        homography_matrix_list = []

        mid = len(images) // 2
        panorama = self.cylindricalWarp(images[mid])
        # plt.imshow(panorama)
        # plt.show()
        for i in range(1, mid + 1):
            try:
                print(f"Stitching image {mid - i} to the panorama.")
                panorama, H = self.stitch_pair(
                    panorama, self.cylindricalWarp(images[mid - i])
                )
                homography_matrix_list.append(H)
            except IndexError:
                pass
            try:
                print(f"Stitching image {mid + i} to the panorama.")
                panorama, H = self.stitch_pair(
                    panorama, self.cylindricalWarp(images[mid + i])
                )
                homography_matrix_list.append(H)
            except IndexError:
                pass

        return panorama, homography_matrix_list

    def resize_if_needed(self, img):
        resized_img = img
        if img.shape[0] > self.size_threshold:
            x = self.size_threshold / img.shape[0]
            resized_img = cv2.resize(img, None, fx=x, fy=x)

        return resized_img

    def stitch_pair(self, image1, image2):
        self.idx += 1

        # Resize the images if needed
        image1 = self.resize_if_needed(image1)
        image2 = self.resize_if_needed(image2)

        # Find keypoints and descriptors for both images
        kp1, des1 = self.sift.detectAndCompute(image1, None)
        kp2, des2 = self.sift.detectAndCompute(image2, None)

        # Match descriptors between images
        matches = self.matcher.knnMatch(des1, des2, k=2)
        # print(len(matches))
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Check if enough matches are found
        MIN_MATCH_COUNT = 10
        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )
            # Find homography matrix
            H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        else:
            print("Not enough matches are found.")
            return image1  # Return the base image if homography fails

        # Warp the second image onto the panorama
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        # Calculate the size of the final panorama
        corners_image2 = np.array(
            [[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype="float32"
        ).reshape(-1, 1, 2)
        warped_corners_image2 = cv2.perspectiveTransform(corners_image2, H)
        corners = np.vstack(
            ([[0, 0], [0, h1], [w1, h1], [w1, 0]], warped_corners_image2.reshape(-1, 2))
        )

        # Calculate the bounds of the new image
        xmin, ymin = np.int32(corners.min(axis=0))
        xmax, ymax = np.int32(corners.max(axis=0))
        translation_dist = [-xmin, -ymin]

        # Adjust homography matrix to place the image correctly
        translation_matrix = np.array(
            [[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]]
        )
        panorama_size = (xmax - xmin, ymax - ymin)

        # # Warp image2 onto the canvas
        # panorama = cv2.warpPerspective(image2, translation_matrix @ H, panorama_size)
        # plt.imshow(panorama)
        # plt.show()
        # # Place the original image1 on the panorama canvas
        # panorama[
        #     translation_dist[1] : h1 + translation_dist[1],
        #     translation_dist[0] : w1 + translation_dist[0],
        # ] = image1
        # plt.imshow(panorama)
        # plt.show()

        warped_image2 = cv2.warpPerspective(
            image2, translation_matrix @ H, panorama_size
        )

        # Place the original image1 on the panorama canvas
        panorama = np.zeros((ymax - ymin, xmax - xmin, 3), dtype=np.uint8)
        # panorama = warped_image2
        panorama[
            translation_dist[1] : h1 + translation_dist[1],
            translation_dist[0] : w1 + translation_dist[0],
        ] = image1

        mask = np.all(panorama == 0, axis=2)
        # plt.imshow(mask)
        # plt.show()
        panorama = np.where(mask[:, :, None], warped_image2, panorama)
        # plt.imshow(panorama)
        # plt.show()

        # cv2.imwrite(f"panorama_result{self.idx}.jpg", panorama)
        return panorama, H

    def make_panaroma_for_images_in(self, path):
        images = [cv2.imread(file) for file in sorted(glob.glob(path + "/*.JPG"))]
        panorama, homography_matrix_list = self.stitch_images(images)
        return panorama, homography_matrix_list


# # Load images for stitching
# path = f"Images/I{sys.argv[1]}"
# images = [cv2.imread(file) for file in sorted(glob.glob(path + "/*.JPG"))]

# # Create the stitcher and process the images
# stitcher = PanoramaStitcher()
# panorama = stitcher.stitch_images(images)

# # Show or save the result
# # cv2.imshow("Panorama", panorama)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# cv2.imwrite("panorama_result.jpg", panorama)
