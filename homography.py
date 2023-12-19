import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.spatial import distance
from matplotlib.patches import ConnectionPatch

def load_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    return image

def convert_to_grayscale(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        return image
    else:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_to_grayscale(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        return image
    else:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
# Load the images to look at 
print("Loading images... ",end="")
image_1 = load_image('graff/img1.png')
image_2 = load_image('graff/img2.png')
image_3 = load_image('graff/img3.png')  
image_4 = load_image('graff/img4.png')  
image_5 = load_image('graff/img5.png')    
image_6 = load_image('graff/img6.png')  
print("Done")

 
H12 = np.array([[8.7976964e-01,3.1245438e-01,-3.9430589e+01],
                [-1.8389418e-01,9.3847198e-01,1.5315784e+02],
                [1.9641425e-04,-1.6015275e-05,1.0000000e+00]])

H13= np.array([[7.6285898e-01,-2.9922929e-01,2.2567123e+02],
               [3.3443473e-01,1.0143901e+00,-7.6999973e+01],
               [3.4663091e-04,-1.4364524e-05,1.0000000e+00]])

H14= np.array([[6.6378505e-01,6.8003334e-01,-3.1230335e+01],
               [-1.4495500e-01,9.7128304e-01,1.4877420e+02],
               [4.2518504e-04,-1.3930359e-05,1.0000000e+00]])

H15= np.array([[6.2544644e-01,5.7759174e-02,2.2201217e+02],
               [2.2240536e-01,1.1652147e+00,-2.5605611e+01],
               [4.9212545e-04,-3.6542424e-05,1.0000000e+00]])

H16= np.array([[4.2714590e-01,-6.7181765e-01,4.5361534e+02],
               [4.4106579e-01,1.0133230e+00,-4.6534569e+01],
               [5.1887712e-04,-7.8853731e-05,1.0000000e+00]])

images=[image_1,image_2,image_3,image_4,image_5,image_6]
H=[H12,H13,H14,H15,H16]

#Second image index 2-6
n=2 

image_A = images[0]
image_B = images[n-1]

# Detect keypoints on both images using SIFT
print("Detecting keypoints with SIFT... ",end="")
sift = cv2.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(image_A,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(image_B,None)
print("Done")
print(f'Detected {len(keypoints_1)} features for image 1 and {len(keypoints_2)} features for image 2.')

# Match keypoints
print("Matching keypoints using BFMatcher... ",end="")
bf = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors_1,descriptors_2)
print("Done")
print(f"Matched {len(matches)} keypoints accross both images.")

# Sort best keypoints first for testing
#matches = sorted(matches, key = lambda x:(1/x.distance)) # Reverse
matches = sorted(matches, key = lambda x:x.distance)

# All-to-all check of feature matching
image_A = convert_to_grayscale(image_A)
image_B = convert_to_grayscale(image_B)

inliers_img_1=[]
inliers_img_2=[]
outliers_img_1=[]
outliers_img_2=[]

for i in range(0,len(matches)):
    #if i > 100 : break #### Cut short for testing 0-len(matches)
    x1=keypoints_1[matches[i].queryIdx].pt[0]
    y1=keypoints_1[matches[i].queryIdx].pt[1]

    x2=keypoints_2[matches[i].trainIdx].pt[0]
    y2=keypoints_2[matches[i].trainIdx].pt[1]

    p1=np.array([[x1],
                 [y1],
                 [1]])
    p2=np.array([[x2],
                 [y2],
                 [1]])
    
    p2_H=np.dot(H[n-2],p1)
    p2_H=p2_H/p2_H[2]

    p_img1=[x1,y1]
    p_img2=[x2,y2]

    dist = np.linalg.norm(p2_H - p2)

    if dist < 5:
        inliers_img_1.append(p_img1)
        inliers_img_2.append(p_img2)
    else:
        outliers_img_1.append(p_img1)
        outliers_img_2.append(p_img2)

inliers_img_1 = np.array(inliers_img_1)
inliers_img_2 = np.array(inliers_img_2)
outliers_img_1 = np.array(outliers_img_1)
outliers_img_2 = np.array(outliers_img_2)


fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.scatter(inliers_img_1[:,0],inliers_img_1[:,1],marker="o",color="#aeff00") 
ax1.scatter(outliers_img_1[:,0],outliers_img_1[:,1],marker="o",color="red")  
ax1.imshow(image_A, cmap='gray')

ax2.scatter(inliers_img_2[:,0],inliers_img_2[:,1],marker="o",color="#aeff00")
ax2.scatter(outliers_img_2[:,0],outliers_img_2[:,1],marker="o",color="red")   
ax2.imshow(image_B, cmap='gray')

n_matches=len(inliers_img_1)+len(outliers_img_1)
n_inliers=len(inliers_img_1)
n_outliers=len(outliers_img_1)

print("Done")
print(f'Detected {n_inliers} inliers ({round((n_inliers/n_matches)*100,3)}%).')
print(f'Detected {n_outliers} outliers ({round((n_outliers/n_matches)*100,3)}%).')

for j in range(0,len(inliers_img_1)):
    xy1 = (inliers_img_1[j])
    xy2 = (inliers_img_2[j])
    con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data",
                        axesA=ax1, axesB=ax2, color="#aeff00")
    ax2.add_artist(con)

for j in range(0,len(outliers_img_1)):
    xy1 = (outliers_img_1[j])
    xy2 = (outliers_img_2[j])
    con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data",
                        axesA=ax1, axesB=ax2, color="red")
    ax2.add_artist(con)

plt.show()


