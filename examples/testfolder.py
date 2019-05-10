import face_alignment
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io

# Run the 3D face alignment on a test image, without CUDA.
#fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda:0', flip_input=True)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

#input = io.imread('../test/assets/yates/')
preds = fa.get_landmarks_from_directory('../test/assets/yates/')
print(preds)
#TODO: Make this nice
"""
fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(input)
ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=6,linestyle='-',color='black',lw=2)
ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=6,linestyle='-',color='black',lw=2)
ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=6,linestyle='-',color='black',lw=2)
ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=6,linestyle='-',color='black',lw=2)
ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=6,linestyle='-',color='black',lw=2)
ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=6,linestyle='-',color='black',lw=2)
ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=6,linestyle='-',color='black',lw=2)
ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=6,linestyle='-',color='black',lw=2)
ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=6,linestyle='-',color='black',lw=2)
ax.axis('off')
plt.show()
"""
