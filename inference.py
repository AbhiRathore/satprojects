from tensorflow.keras.models import load_model
from numpy.random import randint
import numpy as np 
from PIL import Image 
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

npArrHigh = np.load("/media/abhishek/r/satproject/hilowres/npArrHighNew.npy")
npArrLow = np.load('/media/abhishek/r/satproject/hilowres/npArrLowNew.npy')


lr_train, lr_test, hr_train, hr_test = train_test_split(npArrLow, npArrHigh, 
                                                      test_size=0.33, random_state=42)


[X1, X2] = [lr_test, hr_test]
# select random example
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
# generator = load_model('/media/abhishek/r/satproject/hilowres/gen_e_50.h5', compile=False)

src_imageFinal = Image.fromarray((src_image[0]).astype(np.uint8))
tar_imageFinal = Image.fromarray((tar_image[0]).astype(np.uint8))



# # generate image from source
# gen_image = generator.predict(src_image)
# gen_imageFinal = Image.fromarray((gen_image[0]).astype(np.uint8))
# gen_imageFinal.save(os.path.join("/media/abhishek/r/satproject/hilowres/",f"gen_imageFinal{0}.png"))

# plt.figure(figsize=(16, 8))
# plt.subplot(231)
# plt.title('LR Image')
# plt.imshow(src_imageFinal)
# plt.subplot(232)
# plt.title('Superresolution')
# plt.imshow(gen_imageFinal)
# plt.subplot(233)
# plt.title('Orig. HR image')
# plt.imshow(tar_imageFinal)

# plt.show()


plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('LR Image')
plt.imshow(src_imageFinal)
plt.subplot(232)
plt.title('Superresolution')
plt.imshow(src_imageFinal)
plt.subplot(233)
plt.title('Orig. HR image')
plt.imshow(tar_imageFinal)
plt.show()
