# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import xml.etree.ElementTree as ET
# from model import Resnet101IbnA
# from model import make_model
# from argparse import ArgumentParser
# import os
# from Transforms import get_test_transform
# from PIL import Image
# import torch
# from tqdm import tqdm
# import numpy as np


# def get_class_imgs(xml_path, cls):

#     img_paths = []
#     xml_prefix, _ = os.path.split(xml_path)
#     with open(xml_path) as f:
#         et = ET.fromstring(f.read())

#         for item in et.iter('Item'):
#             if int(item.attrib['vehicleID']) == cls:
#                 img_paths.append(os.path.join(xml_prefix, 'image_test', item.attrib['imageName']))

#     return img_paths



# def visualize(X, y, out):
#     '''
#         X: a list of embedding features
#         y: the corresponding label
#     '''
#     X_tsne = TSNE(n_components=2, random_state=42, perplexity=50).fit_transform(X)

#     # Visualize the result
#     plt.figure(figsize=(8, 6))
#     plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, marker='o', s=30)
#     plt.title('t-SNE Visualization')
#     plt.xlabel('Dimension 1')
#     plt.ylabel('Dimension 2')
#     plt.savefig(out)

#     plt.show()


# @torch.no_grad()
# def get_data(model_params, vehicle_ids, xml_path):
#     imgs_ls = [(id, get_class_imgs(xml_path, id)) for id in vehicle_ids] # not the most efficient algo, but anyway
#     test_transform = get_test_transform()

#     net = make_model(backbone='resnet', num_classes=576)
#     # Step 2: Load the state dictionary into the instantiated model
#     # net.load_state_dict(torch.load(model_params, map_location='cpu')) 

#     # Step 3: Move the model to CPU
#     net = net.to('cpu')
#     net.eval()

#     X = []
#     y = []


#     for id, cls_imgs in imgs_ls:
#         for img in tqdm(cls_imgs, dynamic_ncols=True, desc=f'id: {id}'):
#             input = Image.open(img)
#             input = test_transform(input)
#             eu_feat, cos_feat, _ = net(input.unsqueeze(dim=0))
            
#             y.append(id)
#             X.append(eu_feat.squeeze().numpy())

#     return X, y


# def run_tsne(model_params, vehicle_ids, xml_path, out):
#     X, y = get_data(model_params, vehicle_ids, xml_path)    

    
#     visualize(np.array(X), np.array(y), out)





# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('--params', '-p', type=str, default='run3/9.pth')
#     parser.add_argument('--ids', nargs='+', default=[2, 5, 9, 14, 546, 653, 768, 38, 42, 402, 421, 281, 776, 150])
#     parser.add_argument('--xml_path', '-x', type=str, default='../VeRi/test_label.xml')
#     parser.add_argument('--out', type=str)


#     args = parser.parse_args()

#     ids = [int(id) for id in args.ids]

#     run_tsne(args.params, ids, args.xml_path, args.out)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from model import Resnet101IbnA
from argparse import ArgumentParser
import os
from Transforms import get_test_transform
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
from model import make_model

def get_class_imgs(xml_path, cls):

    img_paths = []
    xml_prefix, _ = os.path.split(xml_path)
    with open(xml_path) as f:
        et = ET.fromstring(f.read())

        for item in et.iter('Item'):
            if int(item.attrib['vehicleID']) == cls:
                img_paths.append(os.path.join(xml_prefix, 'image_test', item.attrib['imageName']))

    return img_paths




import matplotlib.cm as cm
import matplotlib.colors as mcolors

def visualize(X, y, out):
    '''
        X: a list of embedding features
        y: the corresponding label
        out: output file to save the visualization
    '''
    # Generate t-SNE embedding
    X_tsne = TSNE(n_components=2, random_state=42, perplexity=50).fit_transform(X)

    # Generate a list of unique labels (vehicle IDs)
    unique_labels = np.unique(y)
    
    # Create a color map with distinct colors
    num_classes = len(unique_labels)
    colors = cm.get_cmap('tab20', num_classes)  # Use the 'tab20' colormap, which provides 20 distinct colors
    
    # Create a dictionary that maps each label to a specific color
    label_to_color = {label: colors(i) for i, label in enumerate(unique_labels)}

    # Prepare a list of colors corresponding to the labels in 'y'
    point_colors = [label_to_color[label] for label in y]

    # Plot t-SNE visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=point_colors, marker='o', s=30, edgecolor='k', alpha=0.7)

    # Create a legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'ID {label}',
                          markersize=10, markerfacecolor=label_to_color[label]) for label in unique_labels]
    plt.legend(handles=handles, loc='best', title='Vehicle IDs', fontsize='small', title_fontsize='medium')

    # Add titles and labels
    plt.title('t-SNE Visualization with Custom Colors')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # Save the plot to file
    plt.savefig(out)

    # Show the plot
    plt.show()


# @torch.no_grad()
# def get_data(model_params, vehicle_ids, xml_path):
#     imgs_ls = [(id, get_class_imgs(xml_path, id)) for id in vehicle_ids] # not the most efficient algo, but anyway
#     test_transform = get_test_transform()

#     net = Resnet101IbnA()  # Instantiate the model
#     net.load_state_dict(torch.load(model_params, map_location=torch.device('cpu')))  # Load the state dictionary into the model
#     net = net.to('cpu')  # Move the model to the CPU
#     net.eval()

#     X = []
#     y = []


#     for id, cls_imgs in imgs_ls:
#         for img in tqdm(cls_imgs, dynamic_ncols=True, desc=f'id: {id}'):
#             input = Image.open(img)
#             input = test_transform(input)
#             eu_feat, cos_feat, _ = net(input.unsqueeze(dim=0))
            
#             y.append(id)
#             X.append(eu_feat.squeeze().numpy())

#     return X, y

@torch.no_grad()
def get_data(model_params, vehicle_ids, xml_path):
    imgs_ls = [(id, get_class_imgs(xml_path, id)) for id in vehicle_ids]  # Load images for given IDs
    test_transform = get_test_transform()  # Image transform (normalization, resizing, etc.)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Choose device
    # device = torch.device('cpu')
    # net = Resnet101IbnA()  # Instantiate the model architecture
    # net.load_state_dict(torch.load(model_params, map_location=device))  # Load model parameters

    net = make_model(backbone='swin', num_classes=576)
    state_dict = torch.load(model_params)
    net.load_state_dict(state_dict)
    net = net.to(device)  # Move model to selected device (CPU or GPU)
    net.eval()  # Set model to evaluation mode

    X = []
    y = []

    for id, cls_imgs in imgs_ls:
        for img in tqdm(cls_imgs, dynamic_ncols=True, desc=f'id: {id}'):
            input = Image.open(img)
            input = test_transform(input).to(device)  # Move the transformed image to the correct device
            eu_feat, cos_feat, _ = net(input.unsqueeze(dim=0))  # Make a batch of one image

            y.append(id)
            X.append(eu_feat.squeeze().cpu().numpy())  # Ensure that features are moved to CPU for further processing

    return X, y

def run_tsne(model_params, vehicle_ids, xml_path, out):
    X, y = get_data(model_params, vehicle_ids, xml_path)    

    
    visualize(np.array(X), np.array(y), out)





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--params', '-p', type=str, default='model_smoothing_0.1/swin_best.pth')
    parser.add_argument('--ids', nargs='+', default=[2, 5, 9, 14, 546, 653, 768,  42, 402, 421, 281, 150])
    parser.add_argument('--xml_path', '-x', type=str, default='../VeRi/test_label.xml')
    parser.add_argument('--out', type=str)


    args = parser.parse_args()

    ids = [int(id) for id in args.ids]

    run_tsne(args.params, ids, args.xml_path, args.out)

