import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict
# from model import make_model
import torch
class VehicleReIDDataset(Dataset):
    def __init__(self, image_directory, label_directory):
        self.image_directory = image_directory
        self.label_directory = label_directory
        self.image_file_paths = []
        self.label_file_paths = []

        # 遍歷圖像目錄，收集圖像路徑和對應的標籤
        for file_name in os.listdir(self.image_directory):
            if file_name.endswith('.jpg'):
                image_path = os.path.join(self.image_directory, file_name)
                label_path = os.path.join(self.label_directory, file_name.replace('.jpg', '.txt'))
                if os.path.exists(label_path):
                    self.image_file_paths.append(image_path)
                    self.label_file_paths.append(label_path)

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, index):
        image_path = self.image_file_paths[index]
        label_path = self.label_file_paths[index]
        return image_path, label_path


@torch.no_grad()
def create_label_feature_map(net,image_root_directory,label_root_directory,image_transform,batch_size,num_workers):
    
    # 建立標籤與特徵的映射字典
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    label_to_feature_map = defaultdict(list)

    # cont = 0

    # 遍歷資料夾
    for folder_name in os.listdir(image_root_directory):


        print(folder_name)
        image_subdirectory = os.path.join(image_root_directory, folder_name)
        label_subdirectory = os.path.join(label_root_directory, folder_name)

        # 創建資料集和數據加載器
        dataset = VehicleReIDDataset(image_directory=image_subdirectory, label_directory=label_subdirectory)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        for batch_image_paths, batch_label_paths in tqdm(data_loader, dynamic_ncols=True, desc=f'Loading images'):

            for image_path, label_path in zip(batch_image_paths, batch_label_paths):
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")

                # 讀取標籤檔案
                with open(label_path, 'r') as label_file:
                    label_lines = label_file.readlines()
                # print(label_path)
                if not label_lines:
                    label_to_feature_map[label_path] = []
                    continue
                

                for label_line in label_lines:
                    label_parts = label_line.strip().split()

                    # 解析標籤資訊
                    center_x_ratio = float(label_parts[1])
                    center_y_ratio = float(label_parts[2])
                    bbox_width_ratio = float(label_parts[3])
                    bbox_height_ratio = float(label_parts[4])

                    # 開啟並裁剪圖像
                    with Image.open(image_path) as img:
                        image_width, image_height = img.size

                        left = int((center_x_ratio - bbox_width_ratio / 2) * image_width)
                        right = int((center_x_ratio + bbox_width_ratio / 2) * image_width)
                        top = int((center_y_ratio - bbox_height_ratio / 2) * image_height)
                        bottom = int((center_y_ratio + bbox_height_ratio / 2) * image_height)

                        cropped_image = img.crop((left, top, right, bottom))

                        # 應用圖像預處理
                        if image_transform is not None:
                            cropped_image = image_transform(cropped_image)

                        # Convert cropped_image to a batch of size 1
                        cropped_image = cropped_image.unsqueeze(0).to(device)  # Adds a batch dimension: (1, C, H, W)

                        # 將裁剪後的圖像存入映射
                        
                        
                        _,_,f_i,_ = net(cropped_image,0,0)

                        # label_to_feature_map[label_path].append(f_i.cpu())
                        # 將特徵與中心座標資訊存入映射字典
                        feature = torch.cat(f_i, dim=1)
                        label_to_feature_map[label_path].append({
                            "feature": feature.cpu(),
                            "center_x_ratio": center_x_ratio,
                            "center_y_ratio": center_y_ratio
                        })

    sorted_label_to_feature_map = dict(
        sorted(
            label_to_feature_map.items(),
            key=lambda item:(
                item[0].split('/')[-2],
                item[0].split('/')[-1].split('.')[0]
            )
        )
    )   

    return sorted_label_to_feature_map

# if __name__ == '__main__':
#     image_root_directory = '/home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets/test/images'
#     label_root_directory = '/home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets/test/labels'

#     # 定義圖像預處理
#     image_transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     net = make_model('swin',576)
#     net.eval()
#     label_to_feature_map = create_label_feature_map(net,image_root_directory,label_root_directory,image_transform)

#     # Print the sorted keys for verification

#     for label_path ,objects in label_to_feature_map.items():
#         print(f"Label file: {label_path}")
#         if objects == [[]]:
#             continue

#         for obj in objects:
#             feature = obj["feature"]
#             center_x_ratio = obj['center_x_ratio']
#             center_y_ratio = obj['center_y_ratio']
#             print(f"Feature shape: {feature.shape}, Center: ({center_x_ratio}, {center_y_ratio})")

