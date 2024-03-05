export LIBTPU_INIT_ARGS='--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE'; 
python3 MaxText/train.py MaxText/configs/base.yml run_name=hosseins-v5e-16 \
        ici_fsdp_parallelism=4 ici_tensor_parallelism=4 steps=5 remat_policy=full per_device_batch_size=1 \
        reuse_example_batch=1 enable_profiler=true \
        base_output_directory=gs://maxtext-logs-dogfood-proj/maxtext-logs/ \
        dataset_path=gs://maxtext-logs-dogfood-proj/dataset
