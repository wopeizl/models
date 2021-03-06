##Training details
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

#ResNet50:
python -m paddle.distributed.launch train.py \
       --model=ResNet50 \
       --batch_size=256 \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --lr_strategy=piecewise_decay \
       --num_epochs=120 \
       --lr=0.1 \
       --reader_thread=4 \
       --l2_decay=1e-4
