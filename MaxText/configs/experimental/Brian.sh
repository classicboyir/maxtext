export LIBTPU_INIT_ARGS='--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE'; 

python3 MaxText/train.py MaxText/configs/base.yml run_name=hosseins-ss-1024-baseconfig-profile-run2 \
        ici_fsdp_parallelism=16 ici_tensor_parallelism=16 steps=5 remat_policy=full per_device_batch_size=0.5 \
        base_emb_dim=8960 base_num_heads=32 base_mlp_dim=35840 base_num_decoder_layers=100 head_dim=128  \
        max_target_length=4096 max_eval_target_length=4096 reuse_example_batch=1 enable_profiler=true \
        base_output_directory=gs://maxtext-logs-northam-test/maxtext-logs/ \
        dataset_path=gs://northam-ce-mlai-tpu/vlp-2nic-maxtext-data/
