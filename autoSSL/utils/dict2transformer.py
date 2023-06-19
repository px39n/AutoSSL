from torchvision import transforms
from lightly.transforms.multi_view_transform import MultiViewTransform
def dict2transformer(dict, view=1):
    # Define the desired augmentations
    transform_list = []
    res=None
    if dict:
        for aug_name, aug_params in dict.items():
            transform = getattr(transforms, aug_name)
            transform_list.append(transform(**aug_params))
        transform= transforms.Compose(transform_list)  
        if view !=1:    
            res=MultiViewTransform([transform for _ in range(view)])
        else:
            res=transform
        res.nview=view
        res.aug=transform
        
    return res


from torchvision import transforms
from lightly.transforms.multi_view_transform import MultiViewTransform
def trans2multi(transform, view=1):
    # Define the desired augmentations
 
    transform=transform
    
    if view !=1:    
        res=MultiViewTransform([transform for _ in range(view)])
    else:
        res=transform
    res.nview=view
    res.aug=transform
    return res