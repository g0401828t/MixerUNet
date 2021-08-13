# TransUnet_depthEstimation
Using TransUnet for depth estimation with kitti dataset  

> Code is quite messy, will update ASAP

## Reference
depth estimation model "BTS" => Refrence for dataloading and training, testing code.

https://github.com/g0401828t/TransUnet_depthEstimation/new/main?readme=1

semantic segmentation model "TransUnet" => Refrence for network code.

https://github.com/Beckschen/TransUNet

<img width="1242" alt="image" src="https://user-images.githubusercontent.com/55650445/128837226-80bc8950-da15-4ac2-945e-50009cf0da49.png">

## Implementation
<img width="1101" alt="image" src="https://user-images.githubusercontent.com/55650445/128836980-5f419cd3-d213-406b-9a2c-6dd7efe52732.png">

### Image Loader
Kitti dataset consists of images with size of [375, 1242] or [376, 1241]  
When loading the data, dataloader.py crops the images to [352, 1216] which is dividable by 16.  
TransUNet code only takes inputs with same width and height. However, Kitti dataset has different sizes of width and height.  
So I modified the TransUNet code to take inputs with different sized width and height.  
Only modifying the Encoder part was necessary.  
### Training (without position embeddings)
When training, images are random croped to [352, 704].  
The TransUnet's reshape part between Encoder and Decoder relys on the input image size so I modified the code for this issue.  
Modified the calss DecoderCup() in model.py
### Online Eval
When evaluation, images are not random croped. So the input size is [352, 1216].  
Due to this, the dimenstion for reshaping in between encoder and decoder was an issue.  
I modified the code by taking the input image size as input for DecoderCup() evertime to address the issue.  
I think this is a little bit awkward. Need to think about this method again because it can cause an ill-training of the network.
### Testing and saving the output image(depth_estimated).
Because trained without position embeddings,
when loading state_dict => model.load_state_dict(checkpoint['model'], strict=False)
to not load position embeddings weight or any other missing weights.





## 1st Trial (without pretraining and position enmbeddings)
after 16 epochs
|best|d1|d2|d3|silog|rms|abs_rel|log_rms|log10|sq_rel|
|------|---|---|---|---|---|---|---|---|---|
|  |0.90746|0.98142|0.99575|12.19566|3.19173|0.08877|0.13404|0.03874|0.39217|

<2011_09_26_drive_0009_sync_0000000128.png>
![image](https://user-images.githubusercontent.com/55650445/128855607-de5267a2-7b96-463f-b494-c435362a9b1b.png)
<2011_09_26_drive_0013_sync_0000000085.png>
![image](https://user-images.githubusercontent.com/55650445/128864079-a48d94bc-10e6-4738-8f3b-2be025d8cb4e.png)

## 2nd Trial (pretrained weights without position embeddings)
after 11 epochs
|best|d1|d2|d3|silog|rms|abs_rel|log_rms|log10|sq_rel|
|------|---|---|---|---|---|---|---|---|---|
|  |0.91733|0.98775|0.99784|9.64381|2.75907|0.09390|0.12225|0.03917|0.32307|

![image](https://user-images.githubusercontent.com/55650445/129292197-34562b75-4a9a-4ccd-a351-36c0da905476.png)

![image](https://user-images.githubusercontent.com/55650445/129292248-910476de-118d-4a93-844d-286c046da6a9.png)

Interim check  
![image](https://user-images.githubusercontent.com/55650445/129292919-56a41562-20a3-4296-a97c-05b5fb0495aa.png)  
Not predicting well on bright & far distance (e.g. sky)
