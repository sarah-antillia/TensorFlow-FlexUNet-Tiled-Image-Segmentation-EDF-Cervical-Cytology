<h2>TensorFlow-FlexUNet-Tiled-Image-Segmentation-EDF-Cervical-Cytology (2025/08/26)</h2>

This is the first experiment of Image Segmentation for EDF Real Cervical Cytology Images, 
 based on our 
 <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
<b>TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass)</b></a>
, and a 512x512 pixels 
<a href="https://drive.google.com/file/d/1b8JEmhoRE_A8yxRyyO97EhrkGb_sbU9A/view?usp=sharing">
<b>Augmented-Tiled-EDF-Cervical-Cytology-ImageMask-Dataset.zip</b></a>.
which was derived by us from 
<br><br>
<a href="https://cs.adelaide.edu.au/~carneiro/isbi14_challenge/dataset.html">
<b>ISBI 2014 Challeng Dataset</b>
</a>
in <a href="https://cs.adelaide.edu.au/~carneiro/isbi14_challenge/index.html">
<b>Overlapping Cervical Cytology Image Segmentation Challenge - ISBI 2014</b>
</a>
<br><br>
On an example of the tileddly-splitted image and mask dataset, please refer to our repository
<a href="https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-Follicular-Cell">
Tiled-ImageMask-Dataset-Follicular-Cell</a>
<br>
<br>
As demonstrated in <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel">
TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel</a> ,
 our Multiclass TensorFlowFlexUNet, which uses categorized masks, can also be applied to 
single-class image segmentation models. 
This is because it inherently treats the background as one category and your single-class mask data as 
a second category. In essence, your single-class segmentation model will operate with two categorized classes within our Multiclass UNet framework.
<br>
<br>
<b>Acutual Image Segmentation for 512x512 Tiled-EDF-Cervical-Cytology images</b><br>

As shown below, the inferred masks predicted by our segmentation model trained on the 
PNG dataset appear similar to the ground truth masks.
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test/images/10003_0x1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test/masks/10003_0x1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test_output/10003_0x1.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test/images/barrdistorted_1003_0.3_0.3_10011_0x1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test/masks/barrdistorted_1003_0.3_0.3_10011_0x1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test_output/barrdistorted_1003_0.3_0.3_10011_0x1.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test/images/deformed_alpha_1300_sigmoid_8_10006_1x1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test/masks/deformed_alpha_1300_sigmoid_8_10006_1x1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test_output/deformed_alpha_1300_sigmoid_8_10006_1x1.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The image and mask dataset used here has been taken from the following dataset.<br>
<a href="https://cs.adelaide.edu.au/~carneiro/isbi14_challenge/dataset.html">
<b>ISBI 2014 Challeng Dataset</b>
</a> 
in <a href="https://cs.adelaide.edu.au/~carneiro/isbi14_challenge/index.html">
<b>Overlapping Cervical Cytology Image Segmentation Challenge - ISBI 2014</b>
</a>
<br>
<br>
The automated detection and segmentation of overlapping cells using microscopic images obtained from 
Pap smear can be considered to be one of the major hurdles for a robust automatic analysis of cervical cells. 
The Pap smear is a screening test used to detect pre-cancerous and cancerous processes, which consists of a sample of cells 
collected from the cervix that are smeared onto a glass slide and further examined under a microscope. 
The main factors affecting the sensitivity of the Pap smear test are the number of cells sampled, 
the overlap among these cells, the poor contrast of the cell cytoplasm, and the presence of mucus, blood and 
inflammatory cells. Automated slide analysis techniques attempt to 
improve both sensitivity and specificity by automatically detecting, segmenting and classifying the cells present on a slide.
<br><br>
<b>
In this challenge, the targets are to extract the boundaries of individual cytoplasm and nucleus 
from overlapping cervical cytology images.
</b>
<br>
The First Segmentation of Overlapping Cervical Cells from Extended Depth of Field Cytology Image Challenge is 
held under the auspices of the IEEE International Symposium on Biomedical Imaging (ISBI 2014) held in Beijing,
 China on April 28th - May 2nd, 2013.
<br>
<br>
<b>Please cite the dataset by the following papers:</b>
<br>
<li>“Zhi Lu, Gustavo Carneiro, Andrew P. Bradley, Daniela Ushizima, Masoud S. Nosrati, Andrea G. C. Bianchi,
 Claudia M. Carneiro, and Ghassan Hamarneh. Evaluation of Three Algorithms for the Segmentation of 
 Overlapping Cervical Cells. IEEE Journal of Biomedical and Health Informatics (J-BHI). Jan 2015 (Accepted).”
</li>
<li>
“Zhi Lu, Gustavo Carneiro, and Andrew P. Bradley. An Improved Joint Optimization of Multiple Level Set Functions for the Segmentation of Overlapping Cervical Cells. IEEE Transactions on Image Processing. Vol.24, No.4, pp.1261-1272, April 2015.”
</li>
<br>
<br>
<h3>
<a id="2">
2 Tiled-EDF-Cervical-Cytology ImageMask Dataset
</a>
</h3>
 If you would like to train this Tiled-EDF-Cervical-Cytology Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/1b8JEmhoRE_A8yxRyyO97EhrkGb_sbU9A/view?usp=sharing">
Augmented-Tiled-EDF-Cervical-Cytology-ImageMask-Dataset.zip</a>.
<br>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Tiled-EDF-Cervical-Cytology
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Tiled-EDF-Cervical-Cytology Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/Tiled-EDF-Cervical-Cytology_Statistics.png" width="512" height="auto"><br>
<br>

On the derivation of the augmented dataset, please refer to the following Python scripts:<br>
<li><a href="./generator/TiledImageMaskDatasetGenerator.py">TiledImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
<br>

As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained Tiled-EDF-Cervical-Cytology TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 2

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.02
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Tiled-EDF-Cervical-Cytology 1+1 classes.<br>
<pre>
[mask]
; 1+1 classes
; RGB colors     Cervical-Nucleus:white     
rgb_map = {(0,0,0):0,(255,255,255):1,}

</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInfereuncer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 25,26,27)</b><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 52,53,54)</b><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was terminated at epoch 54.<br><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/asset/train_console_output_at_epoch54.png" width="920" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology</b> folder,<br>
and run the following bat file to evaluate TensorFlowFlexUNet model for Tiled-EDF-Cervical-Cytology.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/asset/evaluate_console_output_at_epoch54.png" width="920" height="auto">
<br><br>

<a href="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Tiled-EDF-Cervical-Cytology/test was very low and dice_coef_multiclass 
very high as shown below.
<br>
<pre>
categorical_crossentropy,0.0095
dice_coef_multiclass,0.9957
</pre>
<br>

<h3>
5 Inference
</h3>
Please move <b>./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Tiled-EDF-Cervical-Cytology.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 512x512 pixels</b><br>
<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test/images/10007_1x0.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test/masks/10007_1x0.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test_output/10007_1x0.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test/images/barrdistorted_1002_0.3_0.3_10011_1x1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test/masks/barrdistorted_1002_0.3_0.3_10011_1x1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test_output/barrdistorted_1002_0.3_0.3_10011_1x1.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test/images/barrdistorted_1003_0.3_0.3_10011_0x1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test/masks/barrdistorted_1003_0.3_0.3_10011_0x1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test_output/barrdistorted_1003_0.3_0.3_10011_0x1.png" width="320" height="auto"></td>
</tr>



<tr>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test/images/barrdistorted_1004_0.3_0.3_10002_0x1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test/masks/barrdistorted_1004_0.3_0.3_10002_0x1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test_output/barrdistorted_1004_0.3_0.3_10002_0x1.png" width="320" height="auto"></td>
</tr>



<tr>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test/images/deformed_alpha_1300_sigmoid_8_10006_1x1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test/masks/deformed_alpha_1300_sigmoid_8_10006_1x1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test_output/deformed_alpha_1300_sigmoid_8_10006_1x1.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test/images/deformed_alpha_1300_sigmoid_10_10011_0x1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test/masks/deformed_alpha_1300_sigmoid_10_10011_0x1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-EDF-Cervical-Cytology/mini_test_output/deformed_alpha_1300_sigmoid_10_10011_0x1.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. CNSeg:: A dataset for cervical nuclear segmentation</b><br>
<a href="https://dl.acm.org/doi/abs/10.1016/j.cmpb.2023.107732">
https://dl.acm.org/doi/abs/10.1016/j.cmpb.2023.107732
</a>
<br>
@article{ZHAO2023107732, title = {CNSeg: A dataset for cervical nuclear segmentation}, <br>
journal = {Computer Methods and Programs in Biomedicine}, volume = {241}, pages = {107732}, year = {2023}, issn = {0169-2607}, <br>
doi = {https://doi.org/10.1016/j.cmpb.2023.107732}, <br>
url = {https://www.sciencedirect.com/science/article/pii/S016926072300398X}, 
author = {Jing Zhao and Yong-jun He and Shu-Hang Zhou and Jian Qin and Yi-ning Xie} }
<br>
<br>
<b>2. ImageMask-Dataset-Cervical-Nucleus</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/ImageMask-Dataset-Cervical-Nucleus">
https://github.com/sarah-antillia/ImageMask-Dataset-Cervical-Nucleus</a>

<br><br>
<b>3. TensorFlow-FlexUNet-Image-Segmentation-Cervical-Cancer </b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Cervical-Cancer">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Cervical-Cancer
</a>
<br>
<br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-Cervical-Nucleus </b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Cervical-Nucleus">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Cervical-Nucleus
</a>
<br>
<br>
<b>5. Tensorflow-Image-Segmentation-Clustered-Cervical-Cell </b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Clustered-Cervical-Cell">
https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Clustered-Cervical-Cell
</a>



