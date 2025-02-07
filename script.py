import os

# train
# script = "python difusco/train.py \
# --data_path '/data/jjingliu/stdd_data/csv/' \
# --storage_path '/data/jjingliu/stdd_result/models' \
# --task de_conv \
# --hidden_dim 128 --n_layers 6 \
# --parallel_sampling 1 \
# --project_name de_conv_ablation_study \
# --resume_id train_1100011_v21  \
# --wandb_logger_name train_110011_v21 \
# --do_train --do_test \
# --batch_size 1  --train_sampling 2 \
# --inference_diffusion_steps 25  \
# --alpha_0 1 --alpha_1 1 --alpha_2 0 --alpha_3 0  --alpha_4 0 --alpha_5 1 --alpha_6 1 --alpha_7 0 \
# --lr_scheduler 'cosine-decay'  --learning_rate 4.0e-5 \
# --aggregation 'sum' \
# --num_epochs 300  \
# --diffusion_schedule 'cosine' \
# --check_val 1 --validation_examples 900 \
# --num_workers 128 \
# --save_plt_loss \
# --gene_shuff_train --gene_shuff_val --gene_shuff_test  \
# --resume_weight_only --ckpt_path /data/jjingliu/stdd_result/models/models/train_110011_v2/train_1100011_v2/checkpoints/epoch=260-step=587250.ckpt "

# fine_tune
# script = "python difusco/fine_tune.py \
# --data_path '/data/jjingliu/real_data/seqfish+/pre_process/' \
# --storage_path '/data/jjingliu/stdd_result/models' \
# --task de_conv \
# --hidden_dim 128 --n_layers 6 \
# --parallel_sampling 8 \
# --project_name fine_tune \
# --resume_id tune_2_lr-5   \
# --wandb_logger_name tune_2_lr-5 \
# --do_train --do_test --fine_tune --idx_tune 2 --test_data 'seqfish+' \
# --batch_size 1  --train_sampling 8 \
# --inference_diffusion_steps 25  \
# --alpha_0 0 --alpha_1 0 --alpha_2 0 --alpha_3 0  --alpha_4 0 --alpha_5 1 --alpha_6 1 --alpha_7 1 \
# --lr_scheduler 'cosine-decay'  --learning_rate 1.0e-5 \
# --aggregation 'sum' \
# --num_epochs 30  \
# --diffusion_schedule 'cosine' \
# --check_val 1 --validation_examples 4 \
# --num_workers 128 \
# --save_plt_loss \
# --gene_shuff_train --gene_shuff_val \
# --resume_weight_only --ckpt_path /data/jjingliu/stdd_result/models/models/train_fuwuqi4_10_110011_v4/train_fuwuqi4_10_110011_v4/checkpoints/epoch=115-step=261000.ckpt "

# script = "python difusco/fine_tune.py \
# --data_path '/data/jjingliu/real_data/merfish/pre_process/' \
# --storage_path '/data/jjingliu/stdd_result/' \
# --resume_weight_only --ckpt_path /data/jjingliu/stdd_result/models/models/train_fuwuqi4_10_110011_v4/train_fuwuqi4_10_110011_v4/checkpoints/epoch=115-step=261000.ckpt \
# --task de_conv  \
# --hidden_dim 128 --n_layers 6 \
# --parallel_sampling 1 \
# --project_name de_conv_fine_tune \
# --resume_id tune_26_0_100 \
# --wandb_logger_name tune_26_0_100 \
# --do_train --do_test --fine_tune --test_data 'merfish'  --idx_tune 143 --seg 5 --gen_num 160 --resolution 100  \
# --batch_size 1 --train_sampling 1 \
# --inference_diffusion_steps 5  \
# --alpha_0 0 --alpha_1 0 --alpha_2 0 --alpha_3 0  --alpha_4 0 --alpha_5 1 --alpha_6 1 --alpha_7 1 \
# --lr_scheduler 'cosine-decay'  --learning_rate 1.0e-4 \
# --aggregation 'sum' \
# --num_epochs 30 \
# --diffusion_schedule 'cosine' \
# --check_val 1 --validation_examples 30 \
# --num_workers 128 \
# --gene_shuff_train --gene_shuff_val \
# --save_A0_only  --save_plt_loss \
# --path_A0 '/data/jjingliu/stdd_result/results_merfish/tune_26_0_100/100/T=5_1/' "

# --save_numpy_heatmap 
# loss_CE + self.args.alpha_1 * RMSE_global + self.args.alpha_2 * term(右随机) + self.args.alpha_3 * l1_norm + 
# self.args.alpha_4 * (Reconstruction_loss_sc + Reconstruction_loss_st) + self.args.alpha_5 * regular_01 + self.args.alpha_6 * KL

# # test simulation
# script = "python difusco/train.py \
# --data_path '/data/jjingliu/stdd_data/csv/' \
# --storage_path '/data/jjingliu/stdd_result/' \
# --ckpt_path /data/jjingliu/stdd_result/models/models/train_100011_v1/train_100011_v1/checkpoints/epoch=238-step=537750.ckpt \
# --task de_conv \
# --hidden_dim 128 --n_layers 6 \
# --parallel_sampling 1 \
# --project_name de_conv_ablation_study \
# --resume_id test_100011_v1_T=1000_1_num100    \
# --wandb_logger_name test_100011_v1_T=1000_1_num100 \
# --do_test --test_data 'simulation' \
# --inference_diffusion_steps 1000  \
# --aggregation 'sum' \
# --num_workers 128 \
# --diffusion_schedule 'cosine' \
# --save_A0_only --save_plt_loss --gene_shuff_test \
# --path_A0 '/data/jjingliu/stdd_result/results/test_100011_v1/T=1000_1/' \
#  "
# --gene_shuff_train --gene_shuff_val --gene_shuff_test

# --save_numpy_heatmap --save_plt_loss

# /data/jjingliu/stdd_result/models/models/tune_v15_3_lr-5_hard/tune_v15_3_lr-5_hard/checkpoints/epoch=16-step=17.ckpt
# # test on seqfish
# script = "python difusco/train.py \
# --data_path '/data/jjingliu/real_data/seqfish+/pre_process/' \
# --storage_path '/data/jjingliu/stdd_result/' \
# --ckpt_path  /data/jjingliu/stdd_result/models/models/tune_2_lr-4/tune_2_lr-4/checkpoints/epoch=29-step=30.ckpt \
# --task de_conv  \
# --hidden_dim 128 --n_layers 6 \
# --parallel_sampling 128 \
# --project_name de_conv_ablation_study \
# --resume_id test_tune_100_lr-4_2_128  \
# --wandb_logger_name test_tune_100_lr-4_2_238    \
# --do_test --test_data 'seqfish+' \
# --batch_size 1 \
# --inference_diffusion_steps 2  \
# --alpha_0 1 --alpha_1 0 --alpha_2 0 --alpha_3 0  --alpha_4 0 --alpha_5 1 --alpha_6 1 \
# --aggregation 'sum' \
# --num_workers 128 \
# --diffusion_schedule 'cosine' \
# --save_A0_only  --save_plt_loss \
# --path_A0 '/data/jjingliu/stdd_result/results_seqfish+/tune2_lr-4/T=2_128/' "

# # test on merfish
# script = "python difusco/train.py \
# --data_path '/data/jjingliu/real_data/merfish/pre_process/' \
# --storage_path '/data/jjingliu/stdd_result/' \
# --ckpt_path /data/jjingliu/stdd_result/models/tune_26_1_100/tune_26_1_100/checkpoints/last.ckpt \
# --task de_conv  \
# --hidden_dim 128 --n_layers 6 \
# --parallel_sampling 3 \
# --project_name de_conv_ablation_study \
# --resume_id tune_26_0_25_3_100 \
# --wandb_logger_name tune_26_0_25_3_100  \
# --do_test --test_data 'merfish'  --gen_num 160 --resolution 100 --idx_tune 143 \
# --batch_size 1 \
# --inference_diffusion_steps 25  \
# --alpha_0 1 --alpha_1 0 --alpha_2 0 --alpha_3 0  --alpha_4 0 --alpha_5 1 --alpha_6 1 \
# --aggregation 'sum' \
# --num_workers 128 \
# --diffusion_schedule 'cosine' \
# --save_A0_only  --save_plt_loss \
# --path_A0 '/data/jjingliu/stdd_result/results_merfish/tune_26_0_100/100/T=25_3/' "

script = "python difusco/fine_tune.py \
--data_path '/data/jjingliu/real_data/PDAC/pre_process/' \
--storage_path '/data/jjingliu/stdd_result/' \
--resume_weight_only --ckpt_path  '/data/jjingliu/stdd_result/models/models/train_fuwuqi4_10_110011_v4/train_fuwuqi4_10_110011_v4/checkpoints/epoch=115-step=261000.ckpt' \
--task de_conv  \
--hidden_dim 128 --n_layers 6 \
--parallel_sampling 4 \
--project_name de_conv_fine_tune \
--resume_id tune_out_PDAC_cos_lr-4_v8 \
--wandb_logger_name tune_out_PDAC_cos_lr-4_v8 \
--do_train --do_test --fine_tune --test_data 'PDAC' --seg 2 \
--batch_size 1 --train_sampling 1 \
--inference_diffusion_steps 2  \
--alpha_0 0 --alpha_1 0 --alpha_2 0 --alpha_3 0  --alpha_4 0 --alpha_5 0 --alpha_6 0 --alpha_7 0 --alpha_8 1  \
--lr_scheduler 'cosine-decay'  --learning_rate 1.0e-4 \
--aggregation 'sum' \
--num_workers 64 --num_epochs 30 \
--check_val 1 --validation_examples 30 \
--gene_shuff_train --gene_shuff_val \
--diffusion_schedule 'cosine' \
--save_A0_only  --save_plt_loss \
--path_A0 '/data/jjingliu/stdd_result/results_PDAC/tune_cos_lr-4_v8/T=2_4/' "

# list_T = [2, 25, 100, 200, 300, 600, 700, 800, 900]

# for ind_t in range(len(list_T)):
#     print(ind_t)
#     T = list_T[ind_t]

#     script = f"python difusco/train.py \
#     --data_path '/data/jjingliu/real_data/PDAC/pre_process/' \
#     --storage_path '/data/jjingliu/stdd_result/' \
#     --ckpt_path  '/data/jjingliu/stdd_result/models/tune_out_PDAC_cos_lr-3_v5/tune_out_PDAC_cos_lr-3_v5/checkpoints/epoch=23-step=600.ckpt' \
#     --task de_conv  \
#     --hidden_dim 128 --n_layers 6 \
#     --parallel_sampling 4 \
#     --project_name de_conv_ablation_study \
#     --resume_id tune_lr-3_{T}_v5  \
#     --wandb_logger_name tune_lr-3_{T}_v5  \
#     --do_test --test_data 'PDAC' \
#     --batch_size 1 \
#     --inference_diffusion_steps {T}  \
#     --alpha_0 1 --alpha_1 0 --alpha_2 0 --alpha_3 0  --alpha_4 0 --alpha_5 1 --alpha_6 1 \
#     --aggregation 'sum' \
#     --num_workers 128 \
#     --diffusion_schedule 'cosine' \
#     --save_A0_only  --save_plt_loss \
#     --path_A0 '/data/jjingliu/stdd_result/results_PDAC/tune_lr-3_v5/T={T}_4/' "

print(script)

os.system(script)