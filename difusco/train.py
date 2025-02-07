"""The handler for training and evaluation."""

import os
from argparse import ArgumentParser
import pprint as pp
import torch
import numpy as np
import random
import wandb
import sys
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
# from pytorch_lightning.strategies.fsdp import FSDPStrategy
# from pytorch_lightning.strategies import DeepSpeedStrategy
# from pytorch_lightning.strategies import Strategy
from pytorch_lightning.utilities import rank_zero_info

from de_conv_model_light import SC2STModel_light
import time


def arg_parser():
  parser = ArgumentParser(description='Train a Pytorch-Lightning diffusion model on a TSP dataset.')
  parser.add_argument('--task', type=str, required=True, default='tsp')
  parser.add_argument('--data_path', type=str, required=True)
  parser.add_argument('--storage_path', type=str, required=True)
  parser.add_argument('--training_split', type=str, default='train.pt')
  parser.add_argument('--training_split_label_dir', type=str, default=None,
                      help="Directory containing labels for training split (used for MIS).")
  parser.add_argument('--validation_split', type=str, default='val.pt')
  parser.add_argument('--test_split', type=str, default='test.pt')
  parser.add_argument('--test_file', type=str, default='test.pt')
  parser.add_argument('--validation_examples', type=int, default=2)

  parser.add_argument('--batch_size', type=int, default=1)
  parser.add_argument('--num_epochs', type=int, default=100)
  parser.add_argument('--learning_rate', type=float, default=1e-4)
  parser.add_argument('--weight_decay', type=float, default=1e-1)
  parser.add_argument('--lr_scheduler', type=str, default='cosine-decay')

  parser.add_argument('--num_workers', type=int, default=16)
  parser.add_argument('--fp16', action='store_true')
  parser.add_argument('--use_activation_checkpoint', action='store_true')

  parser.add_argument('--diffusion_type', type=str, default='categorical')
  parser.add_argument('--diffusion_schedule', type=str, default='linear')
  parser.add_argument('--diffusion_steps', type=int, default=1000)
  parser.add_argument('--inference_diffusion_steps', type=int, default=1000)
  parser.add_argument('--inference_schedule', type=str, default='linear')
  parser.add_argument('--inference_trick', type=str, default="ddim")
  parser.add_argument('--sequential_sampling', type=int, default=1)
  parser.add_argument('--parallel_sampling', type=int, default=1)
  parser.add_argument('--train_sampling', type=int, default=1)

  parser.add_argument('--n_layers', type=int, default=12)
  parser.add_argument('--hidden_dim', type=int, default=256)
  parser.add_argument('--sparse_factor', type=int, default=-1)
  parser.add_argument('--aggregation', type=str, default='sum')
  parser.add_argument('--two_opt_iterations', type=int, default=1000)
  parser.add_argument('--save_numpy_heatmap', action='store_true')
  parser.add_argument('--save_plt_loss', action='store_true')
  

  parser.add_argument('--project_name', type=str, default='tsp_diffusion')
  parser.add_argument('--wandb_entity', type=str, default=None)
  parser.add_argument('--wandb_logger_name', type=str, default='new')
  parser.add_argument("--resume_id", type=str, default=None, help="Resume training on wandb.")
  parser.add_argument('--ckpt_path', type=str, default=None)
  parser.add_argument('--resume_weight_only', action='store_true')

  parser.add_argument('--do_train', action='store_true')
  parser.add_argument('--do_test', action='store_true')
  parser.add_argument('--do_valid_only', action='store_true')
  parser.add_argument('--check_val', type=int, default=20)
  
  parser.add_argument('--save_A0_only', action='store_true')
  parser.add_argument('--path_A0', type=str, default='/media/imin/DATA/jjingliu/STDD/csv_data/de_conv/std2/')

  parser.add_argument('--N', type=int, default=256,
                    help="number of single cell")
  parser.add_argument('--S', type=int, default=32,
                    help="number of spot")
  parser.add_argument('--gen_num', type=int, default=2000,
                    help="number of genes")
  parser.add_argument('--cell_type', type=int, default=8,
                    help="number of cell_type")
  parser.add_argument('--cells_per_type', type=int, default=256,
                    help="cells_per_type")
  
  parser.add_argument('--self_graph', action='store_true',
                      help="sc2sc graph(cell_type) & st2st graph(location)")

  parser.add_argument('--st_k', type=int, default=4,
                    help="k_neighbor for location of spots")
  
  parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
  
  parser.add_argument('--source', type=str, default='real',
                      help="generate random st_data from real_dataset or simulation_dataset")
  parser.add_argument('--data', type=str, default='seqFISH',
                      help="generate random st_data from real_dataset")
  parser.add_argument('--resolution', type=int, default=50,
                      help="resolution for merfish")
                      
  parser.add_argument('--test_data', type=str, default='simulation',
                      help="test dataset")
  
  parser.add_argument('--num_train', type=int, default=10000,
                    help="number of cell_type")
  parser.add_argument('--num_val', type=int, default=100,
                      help="number of cell_type")
  parser.add_argument('--num_test', type=int, default=1,
                      help="number of cell_type")


  parser.add_argument('--sc_norm', action='store_true')
  parser.add_argument('--fine_tune', action='store_true')
  parser.add_argument('--zero_padding', action='store_true')
  parser.add_argument('--idx_tune', type=int, default=0)
  parser.add_argument('--alpha_0', type=float, default=1e-2)
  parser.add_argument('--alpha_1', type=float, default=1e-2)
  parser.add_argument('--alpha_2', type=float, default=1e-2)
  parser.add_argument('--alpha_3', type=float, default=1e-2)
  parser.add_argument('--decode', action='store_true')
  parser.add_argument('--alpha_4', type=float, default=1e-0)
  parser.add_argument('--alpha_5', type=float, default=1e-0)
  parser.add_argument('--alpha_6', type=float, default=1e-0)
  parser.add_argument('--alpha_7', type=float, default=1e-0)

  parser.add_argument('--gene_shuff_train', action='store_true')
  parser.add_argument('--gene_shuff_val', action='store_true')
  parser.add_argument('--gene_shuff_test', action='store_true')
  
  args = parser.parse_args()
  return args

# wandb.init(settings=wandb.Settings(init_timeout=120))
# os.environ["WANDB_API_KEY"] = 'b6f60853ff5d79ca4b91c14f0a8ed73abbe41000'
# os.environ["WANDB_MODE"] = "offline"

def main(args):
  pp.pprint(vars(args))
  # run = wandb.init(mode="dryrun")
  epochs = args.num_epochs
  project_name = args.project_name
  seed = args.seed

  def setup_seed(seed):
      torch.manual_seed(seed)
      # torch.cuda.manual_seed(seed)
      # torch.cuda.manual_seed_all(seed)
      np.random.seed(seed)
      random.seed(seed)
      torch.backends.cudnn.deterministic = True
  # 设置随机数种子
  n_gpu = torch.cuda.device_count() 

  # for i in range(n_gpu): 
  #   torch.cuda.set_device(torch.device(f'cuda:{i}'))
  #   torch.cuda.manual_seed(args.seed + i)
  #   setup_seed(args.seed + i)

  if args.task == 'de_conv':
    model_class = SC2STModel_light
    saving_mode = 'min'
  elif args.task == 'mis':
    model_class = MISModel
    saving_mode = 'max'
  else:
    raise NotImplementedError

  model = model_class(param_args=args)

  wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
  
  wandb_logger = WandbLogger(
      name=args.wandb_logger_name,
      project=project_name,
      entity=args.wandb_entity,
      save_dir=os.path.join(args.storage_path, f'models'),
      id=args.resume_id or wandb_id,
  )
  rank_zero_info(f"Logging to {wandb_logger.save_dir}/6{wandb_logger.name}/{wandb_logger.version}")

  checkpoint_callback = ModelCheckpoint(
      monitor='val/solved_cost', mode=saving_mode,
      save_top_k=3, save_last=True,
      dirpath=os.path.join(wandb_logger.save_dir,
                           args.wandb_logger_name,
                           wandb_logger._id,
                           'checkpoints'),
  )
  lr_callback = LearningRateMonitor(logging_interval='step')

  # strategy = FSDPStrategy(
  #   # Default: The CPU will schedule the transfer of weights between GPUs
  #   # at will, sometimes too aggressively
  #   # limit_all_gathers=False,
  #   # Enable this if you are close to the max. GPU memory usage
  #   limit_all_gathers=True,)


  trainer = Trainer(
      accelerator="auto",
      devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
      # devices=None,
      max_epochs=epochs,
      callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback, lr_callback],
      logger=wandb_logger,
      check_val_every_n_epoch=args.check_val,
      strategy=DDPStrategy(static_graph=True),
      # strategy="deepspeed_stage_3",
      # strategy=FSDPStrategy(cpu_offload=True),
      # strategy=FSDPStrategy(),
      # strategy="dp",
      precision=16 if args.fp16 else 32,
  )

  # rank_zero_info(
  #     f"{'-' * 100}\n"
  #     f"{str(model.model)}\n"
  #     f"{'-' * 100}\n"
  # )

  ckpt_path = args.ckpt_path

  if args.do_train:
    if args.resume_weight_only:
      model = model_class.load_from_checkpoint(ckpt_path, param_args=args)
      trainer.fit(model)
    else:
      trainer.fit(model, ckpt_path=ckpt_path) 

    if args.do_test:
      trainer.test(ckpt_path=checkpoint_callback.best_model_path)

    '''
    elif args.do_test:
      print('test on:', ckpt_path)
      trainer.validate(model, ckpt_path=ckpt_path)
      if not args.do_valid_only:
        trainer.test(model, ckpt_path=ckpt_path)
    trainer.logger.finalize("success")
    '''

  elif args.do_test:
    print('test on:', ckpt_path)
    start_time = time.time()
    trainer.test(model, ckpt_path=ckpt_path)
    print(time.time()-start_time)
      
  trainer.logger.finalize("success")

  # sys.stdout.flush()
  # input('[Press Any Key to start another run]')


if __name__ == '__main__':
  args = arg_parser()
  main(args)
