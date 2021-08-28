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
|TransUNet  |0.90746|0.98142|0.99575|12.19566|3.19173|0.08877|0.13404|0.03874|0.39217|

<2011_09_26_drive_0009_sync_0000000128.png>
![image](https://user-images.githubusercontent.com/55650445/128855607-de5267a2-7b96-463f-b494-c435362a9b1b.png)
<2011_09_26_drive_0013_sync_0000000085.png>
![image](https://user-images.githubusercontent.com/55650445/128864079-a48d94bc-10e6-4738-8f3b-2be025d8cb4e.png)

## 2nd Trial (pretrained weights without position embeddings)
after 11 epochs
|best|d1|d2|d3|silog|rms|abs_rel|log_rms|log10|sq_rel|
|------|---|---|---|---|---|---|---|---|---|
|TransUNet  |0.90746|0.98142|0.99575|12.19566|3.19173|0.08877|0.13404|0.03874|0.39217|
|TransUNet_Pre  |0.91733|0.98775|0.99784|9.64381|2.75907|0.09390|0.12225|0.03917|0.32307|


<2011_09_26_drive_0009_sync_0000000128.png>  
![image](https://user-images.githubusercontent.com/55650445/129292197-34562b75-4a9a-4ccd-a351-36c0da905476.png)
<2011_09_26_drive_0013_sync_0000000085.png>  
![image](https://user-images.githubusercontent.com/55650445/129292248-910476de-118d-4a93-844d-286c046da6a9.png)
<2011_09_26_drive_0017_sync_0000000572.png>  
<img width="1633" alt="image" src="https://user-images.githubusercontent.com/55650445/131207894-de89f20a-03fe-4a07-a92f-8214ccead00e.png">

### Interim check  
![image](https://user-images.githubusercontent.com/55650445/129292919-56a41562-20a3-4296-a97c-05b5fb0495aa.png)  
Not predicting well on bright & far distance (e.g. sky)

## 3rd Trial (ViT -> MLP-Mixer)  
*Limitations*
1. Due to mlp, the encoding input is fixed and in training and testing, the input size must be the same.  
    Can not random crop the input (352, 1216) to (352, 704) when training.  
    a. Not random cropping the input and train the whole image. => Used method.
    b. Random Resize Crop could be considered but the changing the ratio of the image might affect the training and prediction.  
2. Cannot load pretrained weights of MLP-Mixer.  
    MLP-Mixer pretrained image size is (224, 224) so the input of MLP Block is fixed to (196, 768) which is not the same for the image size (352, 704) or (352, 1216).  
     a. Train from scratch. 
     b. Weight initialization.
     c. Pretrained Channel Mixing weights and initialized Token Mixing weights. => Used method.

after 18 epochs
|best|d1|d2|d3|silog|rms|abs_rel|log_rms|log10|sq_rel|
|------|---|---|---|---|---|---|---|---|---|
|TransUNet  |0.90746|0.98142|0.99575|12.19566|3.19173|0.08877|0.13404|0.03874|0.39217|
|TransUNet_Pre  |0.91733|0.98775|0.99784|9.64381|2.75907|0.09390|0.12225|0.03917|0.32307|
|MixerUNet  |0.90336|0.97935|0.99501|12.21110|3.33323|0.09647|0.13867|0.04194|0.42090| 

<2011_09_26_drive_0009_sync_0000000128.png>  
<img width="1641" alt="image" src="https://user-images.githubusercontent.com/55650445/131207478-1192603e-f6ab-43eb-90fd-437ca08f9742.png">  
<2011_09_26_drive_0013_sync_0000000085.png>  
<img width="1638" alt="image" src="https://user-images.githubusercontent.com/55650445/131207492-d27028bd-e9b4-42df-8c69-a1b7bd2ae7d2.png">
<2011_09_26_drive_0052_sync_0000000030.png>  
<img width="1633" alt="image" src="https://user-images.githubusercontent.com/55650445/131207528-153ee6c5-e040-4f1c-a87e-200566580897.png">
<2011_09_26_drive_0017_sync_0000000572.png>  
<img width="1633" alt="image" src="https://user-images.githubusercontent.com/55650445/131207601-099756d9-8b34-4221-a1a3-434506f4ae96.png">



     
## 4th Trial (token mixing mlp dim: 384 -> 384*8)
params: 200107121
Did not considered the input size.  
For standard MLP-Mixer, the input size was 224 so the input token size was (224/16)^2 = 196
However, kitti dataset input size is 352x1216 so the input token size is (352*1216/16^2) = 1672 which is about 8.5 times larger than 384.  
So the token mixing layer's mlp dimension for Kitti dataset should be 8 times larger (384*8) than the standard token mixing layer's mlp dimension (384).  

after 17 epochs, lr 1e-4 => 1e-3
|best|d1|d2|d3|silog|rms|abs_rel|log_rms|log10|sq_rel|
|------|---|---|---|---|---|---|---|---|---|
|TransUNet  |0.90746|0.98142|0.99575|12.19566|3.19173|0.08877|0.13404|0.03874|0.39217|
|TransUNet_Pre  |0.91733|0.98775|0.99784|9.64381|2.75907|0.09390|0.12225|0.03917|0.32307|
|MixerUNet  |0.90336|0.97935|0.99501|12.21110|3.33323|0.09647|0.13867|0.04194|0.42090|  
|MixerUNet_Pre  |0.92374|0.9858|0.99702|10.81143|3.01994|0.08174|0.12245|0.03585|0.34027|  
<2011_09_26_drive_0009_sync_0000000128.png>  
<img width="1634" alt="image" src="https://user-images.githubusercontent.com/55650445/131207661-98793f44-1866-425d-aac2-1c053004274c.png">
<2011_09_26_drive_0013_sync_0000000085.png>  
<img width="1635" alt="image" src="https://user-images.githubusercontent.com/55650445/131207678-248b7dc7-b4e4-4960-9f6f-10b5604d8e93.png">
<2011_09_26_drive_0052_sync_0000000030.png>  
<img width="1630" alt="image" src="https://user-images.githubusercontent.com/55650445/131207792-dff7744a-4825-437a-b0ac-560b822c537d.png">
<2011_09_26_drive_0017_sync_0000000572.png>  
<img width="1634" alt="image" src="https://user-images.githubusercontent.com/55650445/131207838-f4220015-64aa-47cf-8020-81b3b01c80f8.png">
