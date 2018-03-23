import numpy as np
from PIL import Image 

# n = np.load('./data/images.npy')[640:,:,:,:]
# n = np.load('./data/masks.npy')[640:,:,:]
n = ((np.load("pred_all.npy")) ) 

print (np.min(n))
print (np.max(n))
n = ((n)).astype('uint8')
# super_threshold_indices = n < 240
# n[super_threshold_indices] = 0

print(n.shape)
print (np.min(n))
print(np.max(n))




for i in xrange(n.shape[0]):

	Image.fromarray(np.squeeze(n[i])).save('images/' + str(i) + '.png')
	#Image.fromarray(m[i]).save('testdata_ds_visual/color' + str(i) + '.png')	
	print(i)
	#Image.fromarray(n[i]).save('pic/' + str(i) + '.png')
	#Image.fromarray(n[i],'RGB').save('pic/' + str(i) + '.png')


#Image.fromarray(m[89]).show()
# Image.fromarray(m[6]).show()
# Image.fromarray(m[7]).show()
# Image.fromarray(m[8]).show()



# print(m[])



#Image.fromarray(n[89]).show()
# Image.fromarray(n[6]).show()
# Image.fromarray(n[7]).show()
# Image.fromarray(n[8]).show()
# Image.fromarray(m[80],'RGB').show()
# Image.fromarray(m[120],'RGB').show()
# Image.fromarray(m[80],'RGB').save('sample_result4.png')
# Image.fromarray(m[120],'RGB').save('sample_result4.png')

