import numpy as np 
from torch.utils.data._utils.collate import default_collate

class NP_collate_fn():
    """
    用于神经过程(Neural Process)的数据批处理函数。
    """
    def __init__(self, range_context, range_extra_target, sort=False):
        """ batching x,y for NP with random number contexts and targets
        context and target points have same distribution.
        初始化函数。

        Args:
            num_context ([int,int]): maximum num of context
            上下文点数量的范围
            num_extra_target ([int,int] or int): Note that context is contained in target. Take all points as target if =-1
            额外目标点数量的范围。如果为-1，则使用所有点作为目标
            sort (bool, optional): whether sort the *target* points to orinal order. Defaults to False.
            是否对目标点进行排序
        Returns:
            a dict containing batched x and y  # 一个包含批处理后的x和y的字典

        """
        self.range_context, self.range_extra_target, self.sort = range_context, range_extra_target, sort
    def __call__(self, batch): # the function formerly known as "bar"
        """
        批处理函数。

        参数:
        batch (list): 包含多个数据项的列表

        返回:
        dict: 包含批处理后的数据的字典
        """
        # collate_fn
        range_context, range_extra_target, sort = self.range_context, self.range_extra_target, self.sort
        batch = default_collate(batch)
        x = batch['x']
        y = batch['y']
        # Sample a subset of random size
        # 随机选择上下文点数量
        num_points = x.shape[1]
        num_context = np.random.randint(*range_context)
        
        if type(range_extra_target) is int:
            if range_extra_target == -1:
                num_extra_target = num_points - num_context
                num_total = num_points
            else:
                raise ValueError('%s should be -1' % range_extra_target)
        else:
            assert(len(range_context) == 2 and len(range_extra_target) == 2)
            num_extra_target = np.random.randint(*range_extra_target)
            num_total = min(num_context + num_extra_target, num_points)

        # 随机选择点的位置
        locations = np.random.choice( \
                        num_points,
                        size=num_total,
                        replace=False)
        # random select context, sort to original order if sort==True
        # 随机选择上下文点，如果sort==True，则按原始顺序排序
        context_locations = locations[:num_context] if sort==False else np.sort(locations[:num_context])
        batch['context_x'] = x[:, context_locations, :].float()
        batch['context_y'] = y[:, context_locations, :].float()
        
        # 选择目标点
        target_locations = locations if not sort else np.sort(locations)
        batch['target_x'] = x[:, target_locations, :].float()
        batch['target_y'] = y[:, target_locations, :].float()

        del batch['x']
        del batch['y']
        return batch

def ratio2int(percentage, max_val):
    """
    将百分比或整数转换为整数。

    参数:
    percentage (float或int): 百分比或整数
    max_val (int): 最大值

    返回:
    int: 转换后的整数
    """
    if 0 <= percentage <= 1 and type(percentage) is not int:
        out = percentage * max_val
    elif 1 <= percentage <= max_val:
        out = percentage
    elif max_val < percentage:
        out = max_val
    else:
        raise ValueError('percentage cannot be negative')
    return out

def parse_range(r, max_num):
    """
    解析范围参数。

    参数:
    r (int或tuple): 范围参数
    max_num (int): 最大数量

    返回:
    int: 解析后的数量
    """
    if type(r) is int:
        if r == -1:
            num = max_num
        else:
            raise ValueError('%s should be -1' % r)
    else:
        assert(len(r) == 2)
        r[0] = ratio2int(r[0], max_num)
        r[1] = ratio2int(r[1], max_num) + 1
        num = np.random.randint(*r)
    return num

def random_choice(total, choiceN, sort=False):
    """
    随机选择指定数量的项。

    参数:
    total (int): 总项数
    choiceN (int): 选择的数量
    sort (bool): 是否对选择的项进行排序

    返回:
    numpy.ndarray: 选择的项的索引
    """
    locations = np.random.choice( \
                    total,
                    size=choiceN,
                    replace=False)
    locations = locations if sort==False else np.sort(locations)
    return locations

class nnrecon_collate_fn():
    """
    用于神经网络重建的数据批处理函数。
    """
    def __init__(self, range_context, range_extra_target, sort=False):
        """ batching x,y for NP with random number contexts and targets
        context and target points have different distribution.
        初始化函数。
        Args:
            num_context ([int,int]): maximum num of context 
            最大上下文数量
            num_extra_target ([int,int] or int): Note that context is contained in target. Take all points as target if =-1
            额外目标点数量。注意上下文包含在目标中。如果为-1则将所有点作为目标
            sort (bool, optional): whether sort the *target* points to orinal order. Defaults to False.
            是否将*目标*点排序为原始顺序。默认为False
        Returns:
            [type]: [description]
            返回类型：描述
        """
        self.range_context, self.range_extra_target, self.sort = range_context, range_extra_target, sort
    def __call__(self, batch): # the function formerly known as "bar"
        """
        批处理函数。

        参数:
        batch (list): 包含多个数据项的列表

        返回:
        dict: 包含批处理后的数据的字典
        """
        # collate_fn
        range_context, range_extra_target, sort = self.range_context, self.range_extra_target, self.sort
        max_num_context, max_num_extra_target = np.inf, np.inf
        for i in range(len(batch)):
            max_num_context = min(max_num_context, batch[i]['context_x'].shape[0])
            max_num_extra_target = min(max_num_extra_target, batch[i]['target_x'].shape[0])
        num_context = parse_range(range_context, max_num_context)
        num_extra_target = parse_range(range_extra_target, max_num_extra_target)
        num_total = num_context + num_extra_target
        for i in range(len(batch)):
            numc = batch[i]['context_x']
            numt = batch[i]['target_x']
            nc_loc = random_choice(len(numc), num_context, sort=sort)
            nt_loc = random_choice(len(numt), num_extra_target, sort=sort)
            batch[i]['context_x'] = batch[i]['context_x'][nc_loc]
            batch[i]['context_y'] = batch[i]['context_y'][nc_loc]
            batch[i]['target_x'] = batch[i]['target_x'][nt_loc]
            batch[i]['target_y'] = batch[i]['target_y'][nt_loc]
            #del batch[i]['target_x']
            #del batch[i]['target_y']
        # for item in batch:
        #     for key in item:
        #         print(key,item[key].shape)
        batch = default_collate(batch)
        #print('success')
        for key in batch:
            batch[key] = batch[key].float()
        return batch


