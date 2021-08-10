# TransUnet_depthEstimation
Using TransUnet for depth estimation with kitti dataset

# reference
depth estimation model "BTS" => Refrence for dataloading and training, testing code.

https://github.com/g0401828t/TransUnet_depthEstimation/new/main?readme=1

semantic segmentation model "TransUnet" => Refrence for network code.

https://github.com/Beckschen/TransUNet

<img width="1101" alt="image" src="https://user-images.githubusercontent.com/55650445/128836980-5f419cd3-d213-406b-9a2c-6dd7efe52732.png">


# Image Loader
Kitti dataset consists of images with size of [375, 1242] or [376, 1241]  
When loading the data, dataloader.py crops the images to [352, 1216] which is dividable by 16.  
TransUNet code only takes inputs with same width and height. However, Kitti dataset has different sizes of width and height.  
So I modified the TransUNet code to take inputs with different sized width and height.  
Only modifying the Encoder part was necessary.  
## Training (without position embeddings)
When training, images are random croped to [352, 704].  
The TransUnet's reshape part between Encoder and Decoder relys on the input image size so I modified the code for this issue.  
Modified the calss DecoderCup() in model.py
## Online Eval
When evaluation, images are not random croped. So the input size is [352, 1216].  
Due to this, the dimenstion for reshaping in between encoder and decoder was an issue.  
I modified the code by taking the input image size as input for DecoderCup() evertime to address the issue.  
I think this is a little bit awkward. Need to think about this method again because it can cause an ill-training of the network.
## Testing and saving the output image(depth_estimated).
Because trained without position embeddings,
when loading state_dict => model.load_state_dict(checkpoint['model'], strict=False)
to not load position embeddings weight or any other missing weights.

Code is quite messy, will update ASAP
