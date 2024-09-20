- Test
    `python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --use_env run.py -c configs/mobile/cls_emo -m test model.name=EMO_1M trainer.data.batch_size= model.model_kwargs.checkpoint_path=runs_emo/CLS_EMO_1M_DefaultCLS_20240905-213636/net.pth`
- Train
    `python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --use_env run.py -c configs/mobile/cls_emo -m train model.name=EMO_1M trainer.data.batch_size=32`