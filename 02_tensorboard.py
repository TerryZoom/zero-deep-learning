from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs_02')

# Part1 draw line
for i in range(100):
    writer.add_scalar('y = 2x', 2 * i, i)
writer.close()

# Part2 put picture to board
img_path = r'data\hymenoptera_data\train\bees\39672681_1302d204d1.jpg'
img_pil = Image.open(img_path)
img_array = np.array(img_pil)
writer.add_image('bee_pic', img_array, dataformats = 'HWC') # image format is HWC
writer.close() # remember close

# Part3
# use 'python -m tensorboard.main --logdir=xxx' in terminal
# then use 'http://localhost:6006' to see results
# xxx is logs in line4