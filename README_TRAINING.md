# Training Michelangelo Model

This guide explains how to train the Michelangelo model on your point cloud dataset.

## Dataset Preparation

1. **Dataset Structure**:
   Organize your dataset in the following structure:
   ```
   your_dataset_directory/
   ├── train/
   │   ├── sample1_pc.npy
   │   ├── sample1_img.png
   │   ├── sample2_pc.npy
   │   └── sample2_img.png
   ├── val/
   │   └── ...
   └── test/
       └── ...
   ```

   - `*_pc.npy`: Point cloud data as numpy array of shape [N, 3] or [N, 6] (xyz + optional normals)
   - `*_img.png`: Optional corresponding RGB image (224x224)

2. **Data Format**:
   - Point clouds should be normalized to fit within a unit sphere
   - Images should be square and will be resized to 224x224

## Training

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Training**:
   ```bash
   python train.py \
     --config configs/image_cond_diffuser_asl/image-ASLDM-256.yaml \
     --data_dir /path/to/your/dataset \
     --batch_size 32 \
     --num_workers 8 \
     --max_epochs 1000 \
     --log_dir ./logs \
     --gpus 1  # Set to -1 to use all available GPUs
   ```

## Configuration

You can modify the following configurations:

1. **Model Configuration**:
   - Edit `configs/image_cond_diffuser_asl/image-ASLDM-256.yaml` for model architecture

2. **Data Configuration**:
   - Edit `configs/data/point_cloud.yaml` for data loading and augmentation

## Monitoring

Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir=./logs
```

## Resuming Training

To resume training from a checkpoint:
```bash
python train.py \
  --resume_from /path/to/checkpoint.ckpt \
  --data_dir /path/to/your/dataset
```

## Multi-GPU Training

For multi-GPU training, set the `--gpus` flag to the number of GPUs to use:
```bash
python train.py --gpus 4
```

## Troubleshooting

1. **Out of Memory**:
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Slow Training**:
   - Increase number of workers
   - Use a faster storage solution (e.g., SSD)
   - Enable mixed precision training
