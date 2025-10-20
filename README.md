<script src="https://mozilla.github.io/pdf.js/build/pdf.js"></script>

# **DiSCO**

This is the PyTorch Implementation for our paper:

DiSCO: Deconvoluting Spatial Transcriptomics via Combinatorial Optimization with a Foundational Diffusion Model

Jing Liu, Yahao Wu, Limin Li

<div style="width: 100%; height: 600px;">
<canvas id="pdf-canvas" style="border: 1px solid;"></canvas>
</div>

<script>
var url = 'https://github.com/your_username/your_repository/raw/branch/path/to/your_pdf.pdf';

// 使用pdf.js渲染和显示PDF
pdfjsLib.getDocument(url).promise.then(function(pdfDoc) {
 var canvas = document.getElementById('pdf-canvas');
 var context = canvas.getContext('2d');

 // 获取PDF的第一页
 pdfDoc.getPage(1).then(function(page) {
   var viewport = page.getViewport({scale: 1});
   canvas.height = viewport.height;
   canvas.width = viewport.width;

   // 渲染PDF页面到canvas
   page.render({canvasContext: context, viewport: viewport});
 });
});
</script>

## `Requirements`

    
    pytorch-lightning==1.7.7
    scikit-learn==1.0.2
    scipy==1.7.3
    six==1.16.0
    torch-cluster==1.6.0+pt112cu116
    torch-geometric==2.2.0
    torch-scatter==2.1.0+pt112cu116
    torch-sparse==0.6.16+pt112cu116
    torch-spline-conv==1.2.1+pt112cu116
    torch==1.12.0+cu116
    torchaudio==0.12.0+cu116
    torchmetrics==0.11.4
    torchvision==0.13.0+cu116
    wandb==0.13.9


To train the model, you can simply run
```angular2html
python main.py --storage_path ./models --task de_conv --hidden_dim 128 --n_layers 6 --project_name de_conv --resume_id 1 --wandb_logger_name 1 --do_train --do_test --batch_size 1 --train_sampling 2 --inference_diffusion_steps 25 --alpha_0 1 --alpha_1 0 --alpha_2 0 --alpha_3 0 --alpha_4 0 --alpha_5 1 --alpha_6 1 --alpha_7 0 --alpha_8 0 --lr_scheduler cosine-decay --learning_rate 1.0e-3 --aggregation sum --num_epochs 240 --diffusion_schedule cosine --check_val 1 --validation_examples 900 --num_workers 128 --save_plt_loss --gene_shuff_train --gene_shuff_val --gene_shuff_test 
```
For testing, 
```angular2html
python main.py --data_path ... --storage_path ... --ckpt_path./models/.../checkpoints/epoch=115-step=261000.ckpt --task de_conv --hidden_dim 128 --n_layers 6 --parallel_sampling 128 --project_name de_conv --resume_id 2 --wandb_logger_name 2 --do_test --test_data seqfish+ --inference_diffusion_steps 800 --aggregation sum --num_workers 128 --diffusion_schedule cosine --save_A0_only 
```
