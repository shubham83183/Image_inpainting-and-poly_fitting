import numpy as np
import torchvision.transforms as transforms
import torch
from PIL import Image
import matplotlib.pyplot as plt
import functions as f
torch.manual_seed(0)

#  Reading the image and transforming it to tensor of  shape 3*256*256
image = Image.open('target_image.jpg')
transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(size=256)])
target_image = transform(image)
#  Creating mask of shape 256*256
mask = torch.full((target_image.shape[1], target_image.shape[2]), 1)
mask[30:100, 100:110] = 0
mask[220:230, 80:150] = 0
mask[120:180, 150:160] = 0
mask[200:250, 210:220] = 0


#  Defining custom loss function
def myCustomLoss(estimated_y, y):
    return torch.sum(((estimated_y - y) * mask) ** 2) / np.prod(target_image.shape)


# Initialising the loss, Generator and optimizer
loss = myCustomLoss
generator = f.Generator(channels=1)
optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

#  Creating a 2D noise image tensor
noise = torch.randn(1, target_image.shape[1], target_image.shape[2])
noise = torch.unsqueeze(noise, 0)
total_loss = []

for epoch in range(300):
    optimizer.zero_grad()
    if epoch % 10 == 0:
        print(f'{epoch}th epoch started.')
    gen_imgs = generator(noise)
    g_loss = loss(gen_imgs, target_image)
    total_loss.append(g_loss.item())
    g_loss.backward()
    optimizer.step()

gen_imgs = (torch.squeeze(gen_imgs.detach())).permute(1, 2, 0)

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(target_image.permute(1, 2, 0))
ax1.set_title('Real Image')
ax2 = fig.add_subplot(2, 2, 2)
ax2.imshow((target_image*mask).permute(1, 2, 0))
ax2.set_title('Masked Image')
ax3 = fig.add_subplot(2, 2, 3)
ax3.imshow(gen_imgs)
ax3.set_title('Generated Image')
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(total_loss)
ax4.set_title('Loss vs epochs')
ax4.set_xlabel('Epochs')
ax4.set_ylabel('Loss')
plt.show()
