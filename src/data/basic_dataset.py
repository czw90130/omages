import os
import numpy as np

from torch.utils.data import DataLoader, Dataset

class NpzListDataset(Dataset):
    """
    用于加载npz格式数据的数据集类。
    
    参数:
    dset_list (str): 数据集列表文件路径
    duplicate (int): 数据重复次数
    keys (dict): 需要加载的数据键
    """
    def __init__(self, dset_list=None, duplicate = 1, keys=dict(), **kwargs):
        """ 
            The following is the structure of the dset folder:
            数据集文件夹结构如下:
            dset_root/
                dset_list.txt
                dset/
                    0000/
                        *.npy
                    0001/
                        *.npy
                    ...
        """
        super().__init__()
        self.__dict__.update(locals())
        self.shape_list = np.loadtxt(dset_list, dtype=str)
        self.list_dir = os.path.dirname(dset_list)
    
    def __len__(self):
        return len(self.shape_list) * self.duplicate
    
    def ind2name(self, ind):
        """
        将索引转换为对应的文件名。
        
        参数:
        ind (int): 数据索引
        
        返回:
        str: 对应的文件名
        """
        ind = ind % len(self.shape_list)
        return str(self.shape_list[ind]).split('/')[-1].split('.')[0]
    
    def ind2path(self, ind):
        """
        将索引转换为对应的文件路径。
        
        参数:
        ind (int): 数据索引
        
        返回:
        str: 对应的文件路径
        """
        ind = ind % len(self.shape_list)
        return os.path.join(self.list_dir, self.shape_list[ind])
    
    def __getitem__(self, ind, keys=None):
        """
        获取指定索引的数据项。
        
        参数:
        ind (int): 数据索引
        keys (list): 需要加载的数据键列表
        
        返回:
        dict: 包含指定键的数据字典
        """
        ind = ind % len(self.shape_list)
        shape_name = self.shape_list[ind]
        shape_path = os.path.join(self.list_dir, shape_name)
        ditem = {}
        if keys is None:
            keys = self.keys
        loaded = np.load(shape_path, allow_pickle=True)
        for key in keys:
            ditem[key] = np.array(loaded[key])
            if ditem[key].dtype == np.dtype('O'):
                ditem[key] = ditem[key].item()
        return ditem

class NpyListDataset(Dataset):
    """
    用于加载npy格式数据的数据集类。
    
    参数:
    dset_list (str): 数据集列表文件路径
    duplicate (int): 数据重复次数
    keys (dict): 需要加载的数据键
    """
    def __init__(self, dset_list=None, duplicate = 1, keys=dict(), **kwargs):
        """ 
            The following is the structure of the dset folder:
            数据集文件夹结构如下:
            dset_root/
                dset_list.txt
                dset/
                    0000/
                        *.npy
                    0001/
                        *.npy
                    ...
        """
        super().__init__()
        self.__dict__.update(locals())
        self.ditem_list = np.loadtxt(dset_list, dtype=str)
        self.list_dir = os.path.dirname(dset_list)
    
    def __len__(self):
        return len(self.ditem_list) * self.duplicate
    
    def ind2name(self, ind):
        """
        将索引转换为对应的文件名。
        
        参数:
        ind (int): 数据索引
        
        返回:
        str: 对应的文件名
        """
        ind = ind % len(self.ditem_list)
        return str(self.ditem_list[ind]).split('/')[-1].split('.')[0]
    
    def ind2path(self, ind):
        """
        将索引转换为对应的文件路径。
        
        参数:
        ind (int): 数据索引
        
        返回:
        str: 对应的文件路径
        """
        ind = ind % len(self.ditem_list)
        return os.path.join(self.list_dir, self.ditem_list[ind])
    
    def __getitem__(self, ind, keys=None):
        """
        获取指定索引的数据项。
        
        参数:
        ind (int): 数据索引
        keys (list): 需要加载的数据键列表
        
        返回:
        dict: 包含指定键的数据字典
        """
        ind = ind % len(self.ditem_list)
        ditem_name = self.ditem_list[ind] 
        ditem_path = os.path.join(self.list_dir, ditem_name)
        ditem = {}
        if keys is None:
            keys = self.keys
        for key in keys:
            dkey_path = os.path.join(ditem_path, key + '.npy')
            ditem[key] = np.load(dkey_path, allow_pickle=True)
            if ditem[key].dtype == np.dtype('O'):
                ditem[key] = ditem[key].item()
        return ditem
