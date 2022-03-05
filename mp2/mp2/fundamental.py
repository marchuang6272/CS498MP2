'''
Questions 2-4. Fundamental matrix estimation

Question 2. Eight-point Estimation
For this question, your task is to implement normalized and unnormalized eight-point algorithms to find out the fundamental matrix between two cameras.
We've provided a method to compute the average geometric distance, which is the distance between each projected keypoint from one image to its corresponding epipolar line in the other image.
You might consider reading that code below as a reminder for how we can use the fundamental matrix.
For more information on the normalized eight-point algorithm, please see this link: https://en.wikipedia.org/wiki/Eight-point_algorithm#Normalized_algorithm

Question 3. RANSAC
Your task is to implement RANSAC to find out the fundamental matrix between two cameras if the correspondences are noisy.

Please report the average geometric distance based on your estimated fundamental matrix, given 1, 100, and 10000 iterations of RANSAC.
Please also visualize the inliers with your best estimated fundamental matrix in your solution for both images (we provide a visualization function).
In your PDF, please also explain why we do not perform SVD or do a least-square over all the matched key points.

Question 4. Visualizing Epipolar Lines
Please visualize the epipolar line for both images for your estimated F in Q2 and Q3.

To draw it on images, cv2.line, cv2.circle are useful to plot lines and circles.
Check our Lecture 4, Epipolar Geometry, to learn more about equation of epipolar line.
Our Lecture 4 and 5 cover most of the concepts here.
This link also gives a thorough review of epipolar geometry:
    https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf
'''

import os
import cv2 # our tested version is 4.5.5
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
import random
from pathlib import Path

basedir= Path('assets/fountain')
img1 = cv2.imread(str(basedir / 'images/0000.png'), 0)
img2 = cv2.imread(str(basedir /'images/0005.png'), 0)

f, axarr = plt.subplots(2, 1)
axarr[0].imshow(img1, cmap='gray')
axarr[1].imshow(img2, cmap='gray')
plt.show()

# --------------------- Question 2

def calculate_geometric_distance(all_matches, F):
    """
    Calculate average geomtric distance from each projected keypoint from one image to its corresponding epipolar line in another image.
    Note that you should take the average of the geometric distance in two direction (image 1 to 2, and image 2 to 1)
    Arguments:
        all_matches: all matched keypoint pairs that loaded from disk (#all_matches, 4).
        F: estimated fundamental matrix, (3, 3)
    Returns:
        average geomtric distance.
    """
    ones = np.ones((all_matches.shape[0], 1))
    all_p1 = np.concatenate((all_matches[:, 0:2], ones), axis=1)
    all_p2 = np.concatenate((all_matches[:, 2:4], ones), axis=1)
    # Epipolar lines.
    F_p1 = np.dot(F, all_p1.T).T  # F*p1, dims [#points, 3].
    F_p2 = np.dot(F.T, all_p2.T).T  # (F^T)*p2, dims [#points, 3].
    # Geometric distances.
    p1_line2 = np.sum(all_p1 * F_p2, axis=1)[:, np.newaxis]
    p2_line1 = np.sum(all_p2 * F_p1, axis=1)[:, np.newaxis]
    d1 = np.absolute(p1_line2) / np.linalg.norm(F_p2, axis=1)[:, np.newaxis]
    d2 = np.absolute(p2_line1) / np.linalg.norm(F_p1, axis=1)[:, np.newaxis]

    # Final distance.
    dist1 = d1.sum() / all_matches.shape[0]
    dist2 = d2.sum() / all_matches.shape[0]

    dist = (dist1 + dist2)/2
    return dist

# Coords of matched keypoint pairs in image 1 and 2, dims (#matches, 4). Same pair of images as before
# For each row, it consists (k1_x, k1_y, k2_x, k2_y).
# If necessary, you can convert float to int to get the integer coordinate
eight_good_matches = np.load('assets/eight_good_matches.npy')
all_good_matches = np.load('assets/all_good_matches.npy')

def estimate_fundamental_matrix(matches, normalize=False):
    """
    Arguments:
        matches: Coords of matched keypoint pairs in image 1 and 2, dims (#matches, 4).
        normalize: Boolean flag for using normalized or unnormalized alg.
    Returns:
        F: Fundamental matrix, dims (3, 3).
    """
    F = np.eye(3)
    # --------------------------- Begin your code here ---------------------------------------------
    Ta = np.eye(3)
    Tb = np.eye(3)
    if normalize == True:
        mean = np.mean(matches, axis=0)
        centroid = np.zeros((matches.shape[0], matches.shape[1]))
        for i in range(matches.shape[0]):
            centroid[i] = mean
        matches = matches - centroid
        
        mean_of_distances_from_origin_0 = 0
        for match in matches:
            mean_of_distances_from_origin_0 = mean_of_distances_from_origin_0 + sqrt(match[0]**2 + match[1]**2)
        mean_of_distances_from_origin_0 = mean_of_distances_from_origin_0/matches.shape[0]
        scale_factor_0 = sqrt(2)/mean_of_distances_from_origin_0

        mean_of_distances_from_origin_1 = 0
        for match in matches:
            mean_of_distances_from_origin_1 = mean_of_distances_from_origin_1 + sqrt(match[2]**2 + match[3]**2)
        mean_of_distances_from_origin_1 = mean_of_distances_from_origin_1/matches.shape[0]
        scale_factor_1 = sqrt(2)/mean_of_distances_from_origin_1

        scaling = np.zeros((matches.shape[0], matches.shape[1]))
        for i in range(matches.shape[0]):
            scaling[i] = [scale_factor_0,scale_factor_0,scale_factor_1,scale_factor_1]
        matches = np.multiply(matches,scaling)

        Ta_scale = np.array([[scale_factor_0,0,0],[0,scale_factor_0,0],[0,0,1]])
        Ta_translate = np.array([[1,0,-1*mean[0]],[0,1,-1*mean[1]],[0,0,1]])

        Tb_scale = np.array([[scale_factor_1,0,0],[0,scale_factor_1,0],[0,0,1]])
        Tb_translate = np.array([[1,0,-1*mean[2]],[0,1,-1*mean[3]],[0,0,1]])

        Ta = np.matmul(Ta_scale,Ta_translate)
        Tb = np.matmul(Tb_scale,Tb_translate)
    else:
        Ta = np.eye(3)
        Tb = np.eye(3)
        
    A = np.zeros((matches.shape[0], 9))
    for i in range(matches.shape[0]):
        A[i] = [matches[i][2]*matches[i][0], matches[i][2]*matches[i][1], matches[i][2], matches[i][3]*matches[i][0], matches[i][3]*matches[i][1], matches[i][3], matches[i][0], matches[i][1] , 1]
    u, s, v = np.linalg.svd(A)
    F = v[-1].reshape(3,3)
    if normalize == True:
        F = np.transpose(Tb)@F@Ta

    # --------------------------- End your code here   ---------------------------------------------
    return F

F_with_normalization = estimate_fundamental_matrix(eight_good_matches, normalize=True)
F_without_normalization = estimate_fundamental_matrix(eight_good_matches, normalize=False)

# Evaluation (these numbers should be quite small)
print(f"F_with_normalization average geo distance: {calculate_geometric_distance(all_good_matches, F_with_normalization)}")
print(f"F_without_normalization average geo distance: {calculate_geometric_distance(all_good_matches, F_without_normalization)}")


# --------------------- Question 3

def ransac(all_matches, num_iteration, estimate_fundamental_matrix, inlier_threshold):
    """
    Arguments:
        all_matches: coords of matched keypoint pairs in image 1 and 2, dims (# matches, 4).
        num_iteration: total number of RANSAC iteration
        estimate_fundamental_matrix: your eight-point algorithm function but use normalized version
        inlier_threshold: threshold to decide if one point is inlier
    Returns:
        best_F: best Fundamental matrix, dims (3, 3).
        inlier_matches_with_best_F: (#inliers, 4)
        avg_geo_dis_with_best_F: float
    """

    best_F = np.eye(3)
    inlier_matches_with_best_F = None
    avg_geo_dis_with_best_F = 0.0

    ite = 0
    # --------------------------- Begin your code here ---------------------------------------------

    #while ite < num_iteration:

        # random sample correspondences

        # estimate the minimal fundamental estimation problem

        # compute # of inliers

        # update the current best solution
    most_amount_of_inliers  = 0
    while ite < num_iteration:
        inlier_matches = []
        matches = []
        for i in range(8):
            #Choose 8 random points
            random_index = random.randint(0, len(all_matches)-1)
            match = all_matches[random_index]
            matches.append([match[0],match[1],match[2],match[3]])
        matches = np.array(matches)
        F = estimate_fundamental_matrix(matches, normalize=True)

        ones = np.ones((all_matches.shape[0], 1))
        all_p1 = np.concatenate((all_matches[:, 0:2], ones), axis=1)
        all_p2 = np.concatenate((all_matches[:, 2:4], ones), axis=1)

        # Epipolar lines.
        F_p1 = np.dot(F, all_p1.T).T  # F*p1, dims [#points, 3].
        F_p2 = np.dot(F.T, all_p2.T).T  # (F^T)*p2, dims [#points, 3].

        # Geometric distances.
        p1_line2 = np.sum(all_p1 * F_p2, axis=1)[:, np.newaxis]
        p2_line1 = np.sum(all_p2 * F_p1, axis=1)[:, np.newaxis]
        d1 = np.absolute(p1_line2) / np.linalg.norm(F_p2, axis=1)[:, np.newaxis]
        d2 = np.absolute(p2_line1) / np.linalg.norm(F_p1, axis=1)[:, np.newaxis]

        # Final distance.
        dist1 = d1.shape
        dist2 = d2.shape 
        distances = (d1 + d2)/2

        #Find inliers
        number_of_inliers = 0
        for distance_index in range(len(distances)):
            if distances[distance_index] < inlier_threshold:
                number_of_inliers = number_of_inliers + 1
                match = all_matches[distance_index]
                inlier_matches.append([match[0],match[1],match[2],match[3]])

        #Update for best choice
        if number_of_inliers > most_amount_of_inliers:
            most_amount_of_inliers = number_of_inliers
            best_F = F
            inlier_matches_with_best_F = inlier_matches
            avg_geo_dis_with_best_F = calculate_geometric_distance(all_matches, best_F)
        ite = ite + 1




    # --------------------------- End your code here   ---------------------------------------------
    return best_F, inlier_matches_with_best_F, avg_geo_dis_with_best_F

def visualize_inliers(im1, im2, inlier_coords):
    for i, im in enumerate([im1, im2]):
        plt.subplot(1, 2, i+1)
        plt.imshow(im, cmap='gray')
        plt.scatter(inlier_coords[:, 2*i], inlier_coords[:, 2*i+1], marker="x", color="red", s=10)
    plt.show()

num_iterations = [1,100,1000]
inlier_threshold = 0.1 # TODO: change the inlier threshold by yourself
for num_iteration in num_iterations:
    best_F, inlier_matches_with_best_F, avg_geo_dis_with_best_F = ransac(all_good_matches, num_iteration, estimate_fundamental_matrix, inlier_threshold)
    if inlier_matches_with_best_F is not None:
        print(f"num_iterations: {num_iteration}; avg_geo_dis_with_best_F: {avg_geo_dis_with_best_F};")
        # visualize_inliers(img1, img2, inlier_matches_with_best_F)

# --------------------- Question 4

def visualize(estimated_F, img1, img2, kp1, kp2):
    # --------------------------- Begin your code here ---------------------------------------------
    image = img1
    image2 = img2
    for i in range(20):
        #Find random correspondence points
        random_index = random.randint(0, len(kp1)-1)
        x_prime = kp2[random_index]
        x = kp1[random_index]

        #Plot points
        image = cv2.circle(img1,(int(x[0]),int(x[1])),10,(255,255,255),4)
        image2 = cv2.circle(img2,(int(x_prime[0]),int(x_prime[1])),10,(255,255,255),4)

        #Find epipolar equation
        l_prime = (estimated_F@np.array([[x[0]],[x[1]],[1]]))
        l = (np.transpose(estimated_F)@np.array([[x_prime[0]],[x_prime[1]],[1]]))

        #Find two points on the epipolar line
        img1_point1_y = int((-1*0*l[0][0] - l[2][0])/(l[1][0]))
        img1_point2_y = int((-1*(len(img1[0])-1)*l[0][0] - l[2][0])/(l[1][0]))
        img2_point1_y = int((-1*0*l_prime[0][0] - l_prime[2][0])/(l_prime[1][0]))
        img2_point2_y = int((-1*(len(img2[0])-1)*l_prime[0][0] - l_prime[2][0])/(l_prime[1][0]))
        img1_point1 = (0,img1_point1_y)
        img1_point2 = (len(img1[0])-1,img1_point2_y)
        img2_point1 = (0,img2_point1_y)
        img2_point2 = (len(img2[0])-1,img2_point2_y)

        #Plot the epipolar lines
        image = cv2.line(img1,img1_point1,img1_point2,(255,255,255),5)
        image2 = cv2.line(img2,img2_point1,img2_point2,(255,255,255),5)
    cv2.imshow("image" + str(random.randint(0, len(kp1)-1)), image) 
    cv2.imshow("other_image"+ str(random.randint(0, len(kp1)-1)), image2) 
    cv2.waitKey()
    # --------------------------- End your code here   ---------------------------------------------
    pass

all_good_matches = np.load('assets/all_good_matches.npy')
F_Q2 = F_with_normalization # link to your estimated F in Q3
F_Q3 = best_F # link to your estimated F in Q3
visualize(F_Q2, img1, img2, all_good_matches[:, :2], all_good_matches[:, 2:])
visualize(F_Q3, img1, img2, all_good_matches[:, :2], all_good_matches[:, 2:])