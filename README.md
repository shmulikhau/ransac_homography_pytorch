# ransac_homography_pytorch
Implementation of cv2.find_homography by RANSAC (RANdom SAmple Consensus) function on pytorch for gpu support. 
###### on the algorithm: [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus)

## How to run
Take the too sets of matching points that every element in first set corresponding to same element in the second set,
the sets need to be in shape of ```[batch_size],[xy|2]```.

#### How to use:
```python
from pytorch_homography import find_homography
# kpt1 - first set of key-points
# kpt2 - second set of key-points
H = find_homography(kpt1, kpt2)
H.shape
# output:
# 3,3
```
#### Now you can use the homography to draw the picture:
```python
from matplotlib import pyplot as plt
transformed = cv.warpPerspective(img1, H.cpu().numpy(), (h, w))
plt.imshow(transformed, 'gray'),plt.show()
```
#### for full example you can read and use the [demo.ipynb](/demo.ipynb) notebook.
## Times on tesla-t4 ours implemention vs cv2.find_homography
| cv2.find_homography    | ours |
| -------- | ------- |
| 0.00531s  | **0.00350**    |
## On the implementation
- To find the homography, the program selected random 4 sets of key-points, make homography, check the loss between all key-points, and select the best homography.

- When every projects on numpy implements the homography by using SVD function, on torch implementation its work only when the 4 selection key-points has long distance, if it's short, the result will be very non deterministic, to improve this, i implemented a version of *Gaussian elimination* and that's works good.

- I optimize the algorithm to using 10000 iterations and select the best homography that has maximum key-point that has *distance < 6*.
