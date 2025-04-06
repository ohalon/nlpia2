from PIL import image
from PIL import Image
Image('manuscript/images/ch08/wet-paint-text-rainbow-black-stencil.png')
Image(open('manuscript/images/ch08/wet-paint-text-rainbow-black-stencil.png'. ))
Image(open('manuscript/images/ch08/wet-paint-text-rainbow-black-stencil.png','b'))
with open('manuscript/images/ch08/wet-paint-text-rainbow-black-stencil.png', 'rb') as fin:
    im = Image(fin)
im = Image.open('manuscript/images/ch08/wet-paint-text-rainbow-black-stencil.png')
im.read()
im.size
im.shape
import numpy as np
ima = np.array(im)
ima
from matplotlib import pyplot as plt
plt.imshow(ima)
plt.show()
im2 = ima[:100] + ima[100:]
im2 = ima[:-100] + ima[100:]
plt.imshow(im2)
plt.show()
im2 = ima[:-5] + ima[5:]
plt.imshow(ima[:-5] + ima[5:])
plt.show()
ima
ima[:-5]
ima[:-5].shape
ima.shape
plt.imshow(ima[:,:-5,:] + ima[:,5:,:]); plt.show()
plt.imshow((ima[:,:-5,:] + ima[:,5:,:]))/2; plt.show()
plt.imshow((ima[:,:-5,:] + ima[:,5:,:])/2); plt.show()
plt.imshow((ima[:,:-15,:] + ima[:,15:,:])/2); plt.show()
im.copy?
im.copy(10,10,20,20)
dir(im)
strip = im.copy()
strip.crop?
strip.crop(100,50,200,150)
strip.crop((100,50,200,150))
stripe = strip.crop((100,50,200,150))
strip.size
stripe.size
for x in range(0, None, 5):
    print(x)
for x in range(0, -1, 5):
    print(x)
for x in range(0, im.width, 5):
    print(x)
imnew = im.copy()
imnew.putalpha(16)
for x in range(0, im.width-5, 5):
    imnew.paste(im.crop((x,0,x+5,im.height)), (5, 0), im.crop((x,0,x+5,im.height)))
imnew.show()
im.show()
imnew = im.copy()
imnew.putalpha(16)
for x in range(0, im.width-5, 5):
    imnew.paste(im.crop((x,0,x+5,im.height)), (x, 0))
imnew.show()
imnew = im.copy()
imnew.putalpha(16)
for x in range(0, im.width-5, 5):
    strip = im.crop((x,0,x+5,im.height))
    offset = (x, 0)
    imnew.paste(strip, offset, strip)
imnew.show()
imnew.save('smudged.png')
imnew.show()
