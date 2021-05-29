# Imports
import torch
from PIL import Image
from torch.utils.data import Dataset

# Create custom dataset class
# Note : Using torch.Tensor() creates memory overhead, as it has grad_required = True
# Hence use torch.new_tensor() which has required_grad set to false
class Breakhis(Dataset):
    def __init__(self,img_list,transform=None):
        self.img_list=img_list
        self.transform=transform
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self,index):
        img_path=self.img_list[index]
        image=Image.open(img_path)
        temp=img_path.split('\\')
        if temp[3]=='benign':
            ylabel=torch.tensor(0,dtype=torch.float32,requires_grad=False)
        elif temp[3]=='malignant':
            ylabel=torch.tensor(1,dtype=torch.float32,requires_grad=False)
        else:
            print(temp,temp[7])
        if self.transform:
            image=self.transform(image)
        return [image,ylabel]