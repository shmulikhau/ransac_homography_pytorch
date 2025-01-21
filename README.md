# ransac_homography_pytorch
Implementation of cv2.find_homography by RANSAC (RANdom SAmple Consensus) function on pytorch for gpu support. 
###### on the algorithm: [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus)

## How to run
Takes two sets of matching points, So that every element in first set is corresponding to same element in the second set, this sets should be to be in shape of ```[batch_size],[xy|2]```.

#### Example of use:
```python
from pytorch_homography import find_homography
# kpt1 - first set of key-points in shape of N,2
# kpt2 - second set of key-points in shape of N,2
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
| 0.00531s  | **0.00350s**    |
## Implementation Overview
- To compute the homography, the program randomly selects four sets of key points. It then calculates the homography and assesses the loss across all key points, ultimately selecting the best homography based on this evaluation.
- In the NumPy implementations, homography is derived using the SVD function. However, in the Torch implementation, the algorithm performs effectively only when the selected key points are widely spaced. When the key points are close together, the results tend to be highly variable. To address this issue, I implemented a version of *Gaussian elimination*, which significantly improved the stability of the results.
    - #### How it works:
        First, this is the homography:
        $ \begin{bmatrix} {h_1} & {h_2} & {h_3} \\ {h_4} & {h_5} & {h_6} \\ {h_7} & {h_8} & {h_9} \end{bmatrix} $,

        $ \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} $

        the key-point before transforming: $[x_1,y_1,1]$,

        the output after transforming: $[x_2,y_2]$.

        The equation for $x_2$ is: 
        $\frac{h_1 \cdot x_1+ h_2 \cdot y_1+ h_3 \cdot 1}{h_7 \cdot x_1+ h_8 \cdot y_1+ h_9 \cdot 1} = x_2$,

        and the equation for $y_2$ is:
        $\frac{h_4\cdot x_1+h_5\cdot y_1+ h_6 \cdot 1}{h_7 \cdot x_1+ h_8 \cdot y_1+ h_9 \cdot 1} = y_2$.

        And this is equivalent of $x_2$ equation:

        $0 = \frac{h_1\cdot x_1+h_2\cdot y_1+h_3\cdot 1}{h_7\cdot x_1+h_8\cdot y_1+h_9\cdot 1}-x_2=h_4 x_1+h_5 y_1+h_6\cdot 1-x_2(h_7 x_1+h_8 y_1+h_9\cdot 1) = h_4 x_1+h_5 y_1+h_6\cdot 1-x_2 h_7 x_1-x_2 h_8 y_1-x_2 h_9\cdot 1=0$.

        So we got this vector for $x_2$: $[h_4 x_1, h_5 y_1, h_6 \cdot 1, -x_2 h_7 x_1, -x_2 h_8 y_1, -x_2 h_9 \cdot 1]=0$,

        and same for $y_2$: $[h_4 x_1, h_5 y_1, h_6 \cdot 1, -y_2 h_7 x_1, -y_2 h_8 y_1, -y_2 h_9 \cdot 1]=0$,

        and it's represented to this vectors: $[h_1,h_2,h_3,h_7,h_8,h_9]$ & $[h_4,h_5,h_6,h_7,h_8,h_9]$ from the homography.

        By 4 vectors of $x_1$ from 4 key-points, by *Gaussian elimination* - reduce the length of the vectors into this one vector: $[h_7,  h_8,h_9]$, and same for $y_1$,

        so now we have 2 vectors one from $x_1$ and one from $y_1$, from 4 key-points.

        Now reduce again by *Gaussian elimination* into one vector of $[h_8,h_9]$.

        Set $h_9$ to be $1$, and because $[h_8,h_9] = 0$, the $h_8$ from the homography equale to $-\frac{h_9}{h_8}$.

        Now we will go back step by step in revese order, and slove every value in the homography by *Gaussian elimination*.
- I optimized the algorithm to run 10,000 iterations, selecting the best homography based on the maximum number of key points with *distances < 6*.
