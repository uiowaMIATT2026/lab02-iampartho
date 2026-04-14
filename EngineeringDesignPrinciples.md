* use img1,png as fixed image and img2.png as the moving image

* since both image are circles so maybe rotation is not the transform I should be looking for, translation makes much more sense

* So, I will use gradient descent as the optimizer and MSE as the metric and optimize for the translation parameters

* After talking with Prof. Johnson we realized that the instructions are misleading, png file type does not have information regarding origin, spacing and direction cosine. So, we need to create nrrd file or nii.gz file for generating the circles and then register them 
