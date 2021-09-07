# TransUnet_depthEstimation
Using TransUnet for depth estimation with kitti dataset  

> Code is quite messy, will update ASAP

## Reference
depth estimation model "[BTS](https://github.com/cogaplex-bts/bts)" => Refrence for dataloading and training, testing code.

semantic segmentation model "[TransUnet](https://github.com/Beckschen/TransUNet)" => Refrence for network code.


<img width="1242" alt="image" src="https://user-images.githubusercontent.com/55650445/128837226-80bc8950-da15-4ac2-945e-50009cf0da49.png">




## Preparation
### 1. Available pre-trained ViT models and MLP-Mixer models
- [Get ViT models](https://console.cloud.google.com/storage/browser/vit_models;tab=objects?prefix=&forceOnObjectsSortingFiltering=false)
- [Get MLP-Mixer models](https://console.cloud.google.com/storage/browser/mixer_models;tab=objects?prefix=&forceOnObjectsSortingFiltering=false)
### 2. Prepare KITTI dataset and Project folder
~~~
.  
├── dataset  
│     └── kitti_dataset  
│            ├── 2011_09_26  
│            ├── ...  
│            ├── 2011_10_03  
│            └── data_depth_annotated  
│                   ├── 2011_09_26_drive_0001_sync  
│                   └── ...  
└── TransUNet_depth  
     ├── checkpoints  
     ├── models 
     ├── outputs  
     ├── results  
     └── train_test_inputs 
      
~~~

### 3. Training
- For training TransUNet  
~~~
python main.py arguments_train_TransUNet.py
~~~
- For trainig MixerUNet  
~~~
python main.py arguments_train_MixerUNet.py
~~~
### 4. Testing and saving results
- For Testing TransUNet  
~~~
python main.py arguments_test_TransUNet.py
~~~
- For Testing MixerUNet  
~~~
python main.py arguments_test_MixerUNet.py
~~~


## Implementation Details
<img width="1101" alt="image" src="https://user-images.githubusercontent.com/55650445/128836980-5f419cd3-d213-406b-9a2c-6dd7efe52732.png">

### Image Loader
Kitti dataset consists of images with size of [375, 1242] or [376, 1241]  
When loading the data, dataloader.py crops the images to [352, 1216] which is dividable by 16.  
TransUNet code only takes inputs with same width and height[224, 224]. However, Kitti dataset has different sizes of width and height.  
So I modified the TransUNet code to take inputs with different sized width and height.  
Done by modifying the Encoder part.  
### Training (without position embeddings)
When training, images are random cropped to [352, 704].  
The TransUnet's reshape part between Encoder and Decoder relys on the input image size so I modified the code for this issue.  
Modified the class DecoderCup() in model.py
### Online Eval
When evaluation, images are not random croped. So the input size is [352, 1216].  
Due to this, the dimenstion for reshaping in between encoder and decoder was an issue.  
I modified the class DecoderCup() def forward() in model.py by adding reshape_size parameter to reshape the input of decoder with respect to the input image shape.   
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
<2011_09_26_drive_0052_sync_0000000030.png>  
<img width="1634" alt="image" src="https://user-images.githubusercontent.com/55650445/131207934-3035a3e1-0bff-470a-8f23-d57534ba1977.png">
<2011_09_26_drive_0117_sync_0000000572.png>  
<img width="1633" alt="image" src="https://user-images.githubusercontent.com/55650445/131207894-de89f20a-03fe-4a07-a92f-8214ccead00e.png">

### Interim check  
![image](https://user-images.githubusercontent.com/55650445/129292919-56a41562-20a3-4296-a97c-05b5fb0495aa.png)  
Not predicting well on bright & far distance (e.g. sky, high contrast pixels)

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
<2011_09_26_drive_0117_sync_0000000572.png>  
<img width="1633" alt="image" src="https://user-images.githubusercontent.com/55650445/131207601-099756d9-8b34-4221-a1a3-434506f4ae96.png">



     
## 4th Trial (token mixing mlp dim: 384 -> 384*8)
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
<2011_09_26_drive_0117_sync_0000000572.png>  
<img width="1634" alt="image" src="https://user-images.githubusercontent.com/55650445/131207838-f4220015-64aa-47cf-8020-81b3b01c80f8.png">

# Results

|RGB|TransUNet|TransUNet_pretrained|MixerUNet|MixerUNet_pretrained|
|---|---|---|---|---|
|<img width="1071" alt="image" src="https://user-images.githubusercontent.com/55650445/131208458-6f9b5fe9-4b5b-4031-b7fb-f91a12624cd5.png">|<img width="1065" alt="image" src="https://user-images.githubusercontent.com/55650445/131208451-68e7d1a2-ce8d-4043-abcc-b4886d3ea738.png">|<img width="1064" alt="image" src="https://user-images.githubusercontent.com/55650445/131208661-86de449b-34a9-44e9-bc01-64d4f029c9d1.png">|<img width="1066" alt="image" src="https://user-images.githubusercontent.com/55650445/131208666-feaba234-ce8d-4b1b-b65f-a0200e2011aa.png">|<img width="1063" alt="image" src="https://user-images.githubusercontent.com/55650445/131208680-7505ad47-1cd5-409c-a3a5-86a4b15b8de8.png">|
|<img width="1069" alt="image" src="https://user-images.githubusercontent.com/55650445/131208527-8e63564d-1b54-49c4-892d-a9595b9d3cec.png">|<img width="1068" alt="image" src="https://user-images.githubusercontent.com/55650445/131208538-8b9dafe4-c1d1-4fc3-b6bf-a6a6a42d40ca.png">|<img width="1063" alt="image" src="https://user-images.githubusercontent.com/55650445/131208611-19b5896e-fa93-4579-a8bb-24383ba39e0c.png">|<img width="1067" alt="image" src="https://user-images.githubusercontent.com/55650445/131208623-9ec04ec0-bf44-4c35-b72e-e82c00af2216.png">|<img width="1068" alt="image" src="https://user-images.githubusercontent.com/55650445/131208635-7f4dfc09-dee0-4063-b00a-9b582322a8e2.png">|
|<img width="1068" alt="image" src="https://user-images.githubusercontent.com/55650445/131208729-6202496c-a638-4d2e-ae89-124484de7f53.png">|<img width="1069" alt="image" src="https://user-images.githubusercontent.com/55650445/131208783-5634716d-7c6b-4995-8b63-612c3150af44.png">|<img width="1060" alt="image" src="https://user-images.githubusercontent.com/55650445/131208792-d87aa1c9-fd6f-48eb-915b-bd6a7239ca86.png">|<img width="1059" alt="image" src="https://user-images.githubusercontent.com/55650445/131208804-0c8db7f6-f4a1-4b02-b609-9825e2fdd474.png">|<img width="1065" alt="image" src="https://user-images.githubusercontent.com/55650445/131208820-79f02dd0-33fd-483d-81ae-f8912caeb0d1.png">|
|<img width="1073" alt="image" src="https://user-images.githubusercontent.com/55650445/131208897-bb3f7ba0-a0b1-4144-a33e-7c65938e4dc2.png">|<img width="1065" alt="image" src="https://user-images.githubusercontent.com/55650445/131208901-ef13b61a-c875-4299-812a-3372268f5106.png">|<img width="1063" alt="image" src="https://user-images.githubusercontent.com/55650445/131208914-a8c51316-bddb-493b-af77-5c5120b690b4.png">|<img width="1064" alt="image" src="https://user-images.githubusercontent.com/55650445/131208926-4cf9b480-9e12-4e47-b9ed-882b9d5f8502.png">|<img width="1065" alt="image" src="https://user-images.githubusercontent.com/55650445/131208942-a53a3814-df2e-4fd7-849c-a3a764a77079.png">|


|best|testing_time(sec)|parameters|d1|d2|d3|silog|rms|abs_rel|log_rms|log10|sq_rel|
|------|----|----|---|---|---|---|---|---|---|---|---|
|TransUNet_Pre  |126.24|  105M|0.91733|0.98775|0.99784|9.64381|2.75907|0.09390|0.12225|0.03917|0.32307|
|MixerUNet_Pre  | 93.67|  200M|0.92374|0.9858|0.99702|10.81143|3.01994|0.08174|0.12245|0.03585|0.34027| 

- MLP-Mixer's biggest limitation is fixed input dim which cause training and testing image size  to be the same. This is crucial because being not available to train with random cropped image will limit the model's performance.
    - Curious whether Random Crop and Resizing it back to the same size of the original input will affect the model's performance and how will it affect it.
- Despite the limitation of MLP-Mixer, we can see that the performance mlp_mixer gives us is quite good compared to ViT.
- Also, although MixerUNet(MLP-Mixer) has more parameters, the testing time is less than TransUNet(ViT). More computations are needed for TransUNet(ViT)
