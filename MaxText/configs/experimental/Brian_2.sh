export LIBTPU_INIT_ARGS='--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE'; 
python3 MaxText/train.py MaxText/configs/base.yml run_name=hosseins-maxtext-v4-32 \
        ici_fsdp_parallelism=4 ici_tensor_parallelism=2 steps=5 remat_policy=full per_device_batch_size=1 \
        reuse_example_batch=1 enable_profiler=true \
        base_output_directory=gs://maxtext-logs-northam-test/maxtext-logs/ \
        dataset_path=gs://northam-ce-mlai-tpu/vlp-2nic-maxtext-data/
