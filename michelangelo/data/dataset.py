import torch
import torch.utils.data as data
from .vipc_dataloader import ViPCDataLoader, ViPCDataLoaderMemory


def get_train_augmentation():
    return False # works better without augmentation
    return {
            "pc_augm_scale": 1.2,
            "pc_augm_rot": True,
            "pc_rot_scale": 90,
            "pc_augm_mirror_prob": 0.5,
            "pc_augm_jitter": False,
            "translation_magnitude": 0.1,
            "noise_magnitude_for_generated_samples": 0
        }

def get_dataset(
    data_dir,
    phase='train',
    category='plane',
    mini=True,
    image_size=224,
    view_align=True,
    
):
    debug = False
    if phase == "debug":
        phase = "train"
        augmentation = False
        debug = True

    if phase == 'train':
        augmentation = get_train_augmentation()
    else:
        assert phase in ['val', 'test', 'test_trainset']
        augmentation = False

    dataset = ViPCDataLoaderMemory(
        data_dir, phase, view_align=view_align, 
        category=category, mini=mini,
        augmentation=augmentation,
        return_augmentation_params=False,
        R=1,
        debug=debug,
        scale=1,
        image_size=image_size,
    )
    return dataset



