from pytorch_lightning import Callback
import os
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name=''):
        self.reset()
        # name is the name of the quantity that we want to record, used as tag in tensorboard
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class PointCloudSampler(Callback):
    """
    Callback to sample original and reconstructed point clouds after each epoch.
    """
    def __init__(
        self,
        save_dir: str = "points_sampled",
        max_samples: int = 2,
        every_n_epochs: int = 1,
    ):
        """
        Args:
            save_dir: Directory to save point cloud files
            max_samples: Maximum number of samples to save per epoch
            every_n_epochs: Save point clouds every N epochs
        """
        super().__init__()
        self.save_dir = save_dir
        self.max_samples = max_samples
        self.every_n_epochs = every_n_epochs
        os.makedirs(os.path.join(save_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "test"), exist_ok=True)
    
    def _save_xyz(self, points: np.ndarray, filename: str):
        """Save point cloud as .xyz file."""
        # Ensure points are 3D
        if points.shape[-1] > 3:
            points = points[..., :3]  # Only keep xyz coordinates
        
        # Save as space-separated xyz coordinates
        np.savetxt(filename, points.reshape(-1, 3), delimiter=' ', fmt='%.6f')
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        # pl_module.eval()
        self.save_batch(batch, pl_module, "train", trainer.current_epoch)
        # train_chamfer_distance, train_hausdorff_distance = self.calculate_metrics(pl_module, trainer.train_dataloader)
        # # log to logger
        # trainer.logger.log_metrics({
        #     "train_cd": train_chamfer_distance,
        #     "train_hd": train_hausdorff_distance,
        # }, step=trainer.current_epoch)
        # pl_module.train()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        # pl_module.eval()
        self.save_batch(batch, pl_module, "test", trainer.current_epoch)
        test_chamfer_distance, test_hausdorff_distance = self.calculate_metrics(pl_module, trainer.val_dataloaders[dataloader_idx])
        # log to logger
        trainer.logger.log_metrics({
            "test_cd": test_chamfer_distance,
            "test_hd": test_hausdorff_distance,
        }, step=trainer.current_epoch)
        # pl_module.train()

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.save_batch(batch, pl_module, "predict", trainer.current_epoch)
        test_chamfer_distance, test_hausdorff_distance = self.calculate_metrics(pl_module, trainer.predict_dataloaders[dataloader_idx])
        
    def save_batch(self, batch, pl_module, name, current_epoch):

        # Limit number of samples to save
        num_samples = min(self.max_samples, batch["surface"].shape[0])

        # limit to num_samples
        batch = {k: v[:num_samples] for k, v in batch.items()}
        
        # Move batch to device
        device = pl_module.device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Get model predictions
        with torch.no_grad():
            # incomplete_points = batch["incomplete_points"][..., :3]
            # surface = batch["surface"]
            # image = batch["image"]
            # text = batch["text"]
            outputs = pl_module.sample(batch)[-1]
        
        # Get original and reconstructed point clouds
        original_pcs = batch["surface"][..., :3].cpu().numpy()  # [B, N, 3]
        reconstructed_pcs = outputs[:, -pl_module.numpoints:, :].cpu().numpy()  # [B, N, 3]
        incomplete_points = batch["incomplete_points"][..., :3].cpu().numpy()
        
        # Create directory for this epoch
        epoch_dir = osp.join(self.save_dir, name)#, f"epoch_{current_epoch:04d}")
        os.makedirs(epoch_dir, exist_ok=True)
        sample_names = [name]*num_samples
        # Save point clouds for each sample
        for i in range(num_samples):
            # sample_dir = osp.join(epoch_dir, f"sample_{i:02d}")
            # os.makedirs(sample_dir, exist_ok=True)
            
            # Save original and reconstructed point clouds
            self._save_xyz(
                original_pcs[i], 
                osp.join(epoch_dir, f"orig_{sample_names[i]}_{i:02d}.xyz")
            )
            self._save_xyz(
                reconstructed_pcs[i], 
                osp.join(epoch_dir, f"sampled_{sample_names[i]}_{i:02d}.xyz")
            )
            self._save_xyz(
                incomplete_points[i], 
                osp.join(epoch_dir, f"partial_{sample_names[i]}_{i:02d}.xyz")
            )
            
            # # Also save as combined file for easier comparison
            # combined = np.concatenate([
            #     np.hstack([original_pcs[i], np.ones((original_pcs[i].shape[0], 1))]),  # Label 1 for original
            #     np.hstack([reconstructed_pcs[i], 2 * np.ones((reconstructed_pcs[i].shape[0], 1))])  # Label 2 for reconstructed
            # ], axis=0)
            
            # # Save with labels (x y z label)
            # np.savetxt(
            #     osp.join(epoch_dir, f"combined_{i:02d}.xyzc"),
            #     combined,
            #     delimiter=' ',
            #     header='x y z class',
            #     comments='',
            #     fmt='%.6f %.6f %.6f %d'
            # )
            
            # # Save a simple PLY file with colors for visualization
            # self._save_ply_with_colors(
            #     original_pcs[i],
            #     reconstructed_pcs[i],
            #     osp.join(epoch_dir, f"comparison_{i:02d}.ply")
            # )
    def move_to_device(self, batch, device):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        return batch
    
    def calculate_metrics(self, pl_module, dataloader):
        CD_meter = AverageMeter()
        for batch in tqdm(dataloader, desc="Calculating metrics"):
            batch = self.move_to_device(batch, pl_module.device)
            with torch.no_grad():
                outputs = pl_module.sample(batch)[-1]
            original_pcs = batch["surface"][..., :3].cpu()
            reconstructed_pcs = outputs[:, -pl_module.numpoints:, :].cpu()
            cd, _ = chamfer_distance(original_pcs, reconstructed_pcs)
            batch, numpoints, _ = original_pcs.shape
            CD_meter.update(cd.item(), n=batch)
        
        return CD_meter.avg, 0.0
    
    def _save_ply_with_colors(self, original: np.ndarray, reconstructed: np.ndarray, filename: str):
        """Save both point clouds as a single PLY file with different colors."""
        try:
            import open3d as o3d
            
            # Create point cloud objects
            pcd_orig = o3d.geometry.PointCloud()
            pcd_orig.points = o3d.utility.Vector3dVector(original[..., :3])
            pcd_orig.paint_uniform_color([0, 0, 1])  # Blue for original
            
            pcd_recon = o3d.geometry.PointCloud()
            pcd_recon.points = o3d.utility.Vector3dVector(reconstructed[..., :3])
            pcd_recon.paint_uniform_color([1, 0, 0])  # Red for reconstructed
            
            # Combine point clouds
            combined = pcd_orig + pcd_recon
            
            # Save to file
            o3d.io.write_point_cloud(filename, combined)
            
        except ImportError:
            print("Open3D not available, skipping PLY export")

