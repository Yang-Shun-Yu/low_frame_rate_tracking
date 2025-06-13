# import os
# import shutil
# import torch
# from PIL import Image
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms  # <-- Import transforms
# import veri776


# veri776_path = '/home/rhome/littleyang0807/MasterThesis/VeRi'
# # Now we define a simple transformation and run the dataset loader
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # Resize all images to 256x256
#     transforms.ToTensor()           # Convert images to tensor
# ])

# # Testing the get_veri776_train function
# train_loader = veri776.get_veri776_train(veri776_path, num_workers=2, batch_size=2, transform=transform, shuffle=True)
# # Now, iterate over the DataLoader and print output for testing
# print("Testing Veri776Train DataLoader output:")
# for batch_idx, (images, labels) in enumerate(train_loader):
#     print(f"Batch {batch_idx + 1}")
#     print(f"Images shape: {images.shape}")  # Should be [batch_size, 3, C, H, W]
#     print(f"Labels: {labels}")

# import torch
# model = torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=True)
# print(model)
# print(model.fc)  # Print the fully connected (final) layer of the model

import torch
print(torch.__version__)