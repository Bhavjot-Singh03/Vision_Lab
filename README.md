<h1 style="color: Orange;">Vision Lab</h1>

Vision Lab aims to advance the state of the art in medical image segmentation by introducing attention-driven methods tailored for the Kvasir Instrument and Kvasir-SEG datasets. The primary goal is to enhance segmentation accuracy. Vision Lab extends DeepLabv3Plus by integrating multi-head attention into the Atrous Spatial Pyramid Pooling (ASPP) module, significantly improving detail capture. Vision Lab achieves notable mean Intersection over Union (mIoU) and Dice scores of 0.8864, 0.9396, and 0.8410, 0.9134 for Kvasir Instrument and Kvasir-SEG datasets, respectively, showcasing adept region delineation. 

<h2 style="color: Green;">Architecture</h2>

![MHA_Arch](https://github.com/Bhavjot-Singh03/Vision_Lab/assets/131793243/5feb4af2-bf3a-4f0d-bd4c-f51ea01ad3ca)
MHSAR Block
![MHA_Block](https://github.com/Bhavjot-Singh03/Vision_Lab/assets/131793243/489e1795-9913-42df-bd5a-951ada0339eb)

<h2>Performance comparison</h2>

<h3>1. Kvasir Instrument</h3>

| Models               | Dice Score | mIoU   | Accuracy | Recall | Specificity |
|----------------------|------------|--------|----------|--------|-------------|
| NanoNet              | 0.9284     | 0.8790 | 0.9875   | 0.9205 | -           |
| M. Double U-Net      | 0.9138     | 0.8413 | 0.9830   | 0.8787 | 0.9949      |
| BCDU-Net             | 0.8262     | 0.7039 | 0.9696   | 0.7632 | 0.9912      |
| SegNet               | 0.8762     | 0.7796 | 0.9896   | 0.830  | 0.9975      |
| Min-Max Similarity   | 0.925     | 0.873  | -        | -      | -           |
| Double U-Net         | 0.9038     | 0.8430 | -        | -      | -           |
| U-Net                | 0.9158     | 0.8578 | -        | -      | -           |
| Vision Lab (ours)    | 0.9396     | 0.8864 | 0.9862   | 0.9270 | 0.9948      |

<h3>2. Kvasir Seg</h3>

| Models              | mIoU   | Dice Score | Accuracy | Precision | Recall   |
|---------------------|--------|------------|----------|-----------|----------|
| Li-SegPNet          | 0.8800 | 0.9058     | -        | 0.9424    | 0.8509   |
| FCN                 | 0.8022 | 0.8902     | 0.9638   | -         | -        |
| Scaled Dilation     | -      | -          | 0.957    | 0.868     | 0.922    |
| ColonSegNet         | 0.7239 | 0.8206     | 0.9493   | 0.8435    | 0.8496   |
| ResUNet++           | 0.7927 | 0.8133     | -        | 0.8774    | 0.7064   |
| PraNet              | 0.840  | 0.898      | -        | -         | -        |
| Polyp-SAM++         | 0.86   | 0.90       | -        | -         | -        |
| AG-CUResNeSt        | 0.845  | 0.902      | -        | -         | -        |
| TGANet              | 0.8330 | 0.8982     | -        | 0.9123    | 0.9132   |
| TransResU-Net       | 0.8214 | 0.8884     | -        | 0.9022    | 0.9106   |
| ResUNet++           | 0.8329 | 0.8508     | -        | 0.8228    | 0.8756   |
| ConvSegNet          | 0.7936 | 0.8618     | 0.9617   | 0.8692    | 0.9124   |
| Vision Lab (ours)   | 0.8410 | 0.9134     | 0.9681   | 0.9362    | 0.8937   |

