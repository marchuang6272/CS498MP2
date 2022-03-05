'''
Question 1. Keypoint Matching - Putative Matches

Your task is to select putative matches between detected SIFT keypoints found in two images. 
The matches should be selected based on the Euclidean distance for the pairwise descriptor. 
In your implementation, make sure to filter your keypoint matches using Lowe's ratio test 
    A great explanation here: https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work 

For this question, you should implement your solution without using third-party functions e.g., cv2 knnmatch, which can significantly trivialize the problem. 

Meanwhile, please avoid solutions using for-loop which slows down the calculations. 
Hint: To avoid using for-loop, check out scipy.spatial.distance.cdist(X,Y,'sqeuclidean')
'''
import os
import cv2 # our tested version is 4.5.5
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from scipy import spatial

basedir= Path('assets/fountain')
img1 = cv2.imread(str(basedir / 'images/0000.png'), 0)
img2 = cv2.imread(str(basedir /'images/0005.png'), 0)

f, axarr = plt.subplots(2, 1)
axarr[0].imshow(img1, cmap='gray')
axarr[1].imshow(img2, cmap='gray')
plt.show()

# Initiate SIFT detector, The syntax of importing sift descriptor depends on your cv2 version. The following is for cv2 4.5.5, 
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None) # Hints: kp1 and des1 has the same order. kp1.pt gives the x-y coordinate
kp2, des2 = sift.detectAndCompute(img2,None) # Hints: kp2 and des2 has the same order. kp2.pt gives the x-y coordinate

def plot_matches(ax, img1, img2, kp_matches):
    """
    plot the match between two image according to the matched keypoints
    :param ax: plot handle
    :param img1: left image
    :param img2: right image
    :inliers: x,y in the first image and x,y in the second image (Nx4)
    """
    res = np.hstack([img1, img2])
    ax.set_aspect('equal')
    ax.imshow(res, cmap='gray')
    
    ax.plot(kp_matches[:,0], kp_matches[:,1], '+r')
    ax.plot(kp_matches[:,2] + img1.shape[1], kp_matches[:,3], '+r')
    ax.plot([kp_matches[:,0], kp_matches[:,2] + img1.shape[1]],
            [kp_matches[:,1], kp_matches[:,3]], 'r', linewidth=0.8)
    ax.axis('off')
    
def select_putative_matches(des1, des2):
    """
    Arguments:
        des1: cv2 SIFT descriptors extracted from image1 (None x 128 matrix)
        des2: cv2 SIFT descriptors extracted from image2 (None x 128 matrix)
    Returns:
        matches: List of tuples. Each tuple characterizes a match between descriptor in des1 and descriptor in des2. 
                 The first item in each tuple stores the descriptor index in des1, and the second stores index in des2.
                 For example, returning a tuple (8,10) means des1[8] and des2[10] are a match.
    """
    matches = []
    # --------------------------- Begin your code here ---------------------------------------------
    distance_calculations = spatial.distance.cdist(des1,des2)
    for des1_index in range(len(distance_calculations)):
        two_lowest_distances = distance_calculations[des1_index].argsort()[:2]
        matches.append((des1_index,two_lowest_distances[0]))
        matches.append((des1_index,two_lowest_distances[1]))
    # --------------------------- End your code here   ---------------------------------------------
    def lowe_ratio_test(matches):
        """
        run lowe ratio test to filter out 
        Arguments: 
            matches: output from select_putative_matches function
        Returns:
            matches: filter matches using lowe ratio test
        """
        # --------------------------- Begin your code here ---------------------------------------------
        filter_matches = [0,0]
        for match_index in range(len(matches)-1):
            if matches[match_index][0] == matches[match_index+1][0]:
                if distance_calculations[matches[match_index][0]][matches[match_index][1]]/distance_calculations[matches[match_index][0]][matches[match_index+1][1]] < 0.4:
                    filter_matches = np.vstack([filter_matches, matches[match_index]])
        matches = filter_matches
        matches = np.delete(matches, (0), axis=0)
        # --------------------------- End your code here   ---------------------------------------------
        return matches
    filtered_matches = lowe_ratio_test(matches)
    return filtered_matches

matches = select_putative_matches(des1, des2) # your code
match_indices = np.random.permutation(len(matches))[:20] # you can change it to visualize more points
kp_matches = np.array([[kp1[i].pt[0],kp1[i].pt[1],kp2[j].pt[0],kp2[j].pt[1]] for i,j in np.array(matches)[match_indices]])

fig, ax = plt.subplots(figsize=(10,5))
plot_matches(ax, img1, img2, kp_matches)
ax.set_title('Matches visualization')
plt.show()