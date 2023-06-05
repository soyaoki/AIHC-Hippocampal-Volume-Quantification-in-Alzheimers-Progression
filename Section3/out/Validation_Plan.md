# Validation Plan

## What is the intended use of the product?
 The intended use is to build an end-to-end AI system which features a machine learning algorithm that integrates into a clinical-grade viewer and automatically measures hippocampal volumes of new patients.

 ![](https://github.com/soyaoki/ND320-Hippocampal-Volume-Quantification-in-Alzheimers-Progression/blob/master/Section3/out/Study1.png)

## How was the training data collected?
 We are using the "Hippocampus" dataset from the [Medical Decathlon competition](http://medicaldecathlon.com/). This dataset is stored as a collection of NIFTI files, with one file per volume, and one file per corresponding segmentation mask. The original images here are T2 MRI scans of the full brain. As noted, in this dataset we are using cropped volumes where only the region around the hippocampus has been cut out. This makes the size of our dataset quite a bit smaller, our machine learning problem a bit simpler and allows us to have reasonable training times. 

## How did you label your training data?
 "All data has been labeled and verified by an expert human rater, and with the best effort to mimic the accuracy required for clinical use." [Link](http://medicaldecathlon.com/)
 
## How was the training performance of the algorithm measured and how is the real-world performance going to be estimated?
 The training performance of the algorithm is measured by using the following evaluation metrics:

 * CrossEntropyLoss

 ![](https://github.com/soyaoki/ND320-Hippocampal-Volume-Quantification-in-Alzheimers-Progression/blob/master/Section2/out/tensorboard.png)

 [Code is here.](https://github.com/soyaoki/ND320-Hippocampal-Volume-Quantification-in-Alzheimers-Progression/blob/master/Section2/src/experiments/UNetExperiment.py#LL73C57-L73C57)

 The real-world performance is estimated by using the following evaluation metrics:

 * Dice Similarity Coefficient
 * Jaccard Distance

 [Code is here.](https://github.com/soyaoki/ND320-Hippocampal-Volume-Quantification-in-Alzheimers-Progression/blob/master/Section2/src/utils/volume_stats.py)

## What data will the algorithm perform well in the real world and what data it might not perform well on?
 Below we can see the performance of our model on real-world data. It shows that our model performs well on most of the volumes. 

 * Data the algorithm performed well(jaccard>0.85) : ID212, ID242, ID204

 * Data the algorithm did not perform well(jaccard<0.75) : ID042, ID282, ID114, ID314, ID235
 
 ```
 {
    "volume_stats": [
    {
      "filename": "hippocampus_225.nii.gz",
      "dice": 0.8943188759926696,
      "jaccard": 0.8088397790055248
    },
    {
      "filename": "hippocampus_248.nii.gz",
      "dice": 0.8755174452986398,
      "jaccard": 0.7785958453852222
    },
    {
      "filename": "hippocampus_212.nii.gz",
      "dice": 0.9259920362487986,
      "jaccard": 0.8621835847609307
    },
    {
      "filename": "hippocampus_036.nii.gz",
      "dice": 0.9067883964070093,
      "jaccard": 0.8294719827586207
    },
    {
      "filename": "hippocampus_242.nii.gz",
      "dice": 0.9262646139599491,
      "jaccard": 0.8626563173781803
    },
    {
      "filename": "hippocampus_042.nii.gz",
      "dice": 0.8292747837658018,
      "jaccard": 0.7083428051829961
    },
    {
      "filename": "hippocampus_077.nii.gz",
      "dice": 0.8912080057183702,
      "jaccard": 0.8037648272305312
    },
    {
      "filename": "hippocampus_037.nii.gz",
      "dice": 0.8598249801113763,
      "jaccard": 0.7541166620150712
    },
    {
      "filename": "hippocampus_282.nii.gz",
      "dice": 0.8398604110120201,
      "jaccard": 0.7239304812834224
    },
    {
      "filename": "hippocampus_205.nii.gz",
      "dice": 0.889864477953808,
      "jaccard": 0.8015818431911967
    },
    {
      "filename": "hippocampus_316.nii.gz",
      "dice": 0.8743185501694416,
      "jaccard": 0.7767015706806283
    },
    {
      "filename": "hippocampus_252.nii.gz",
      "dice": 0.8970149253731343,
      "jaccard": 0.8132611637347767
    },
    {
      "filename": "hippocampus_114.nii.gz",
      "dice": 0.8132511556240369,
      "jaccard": 0.6852765515450532
    },
    {
      "filename": "hippocampus_130.nii.gz",
      "dice": 0.9174423927971921,
      "jaccard": 0.8474767409078094
    },
    {
      "filename": "hippocampus_257.nii.gz",
      "dice": 0.8699948883966604,
      "jaccard": 0.7699034981905911
    },
    {
      "filename": "hippocampus_243.nii.gz",
      "dice": 0.8618143459915611,
      "jaccard": 0.7571825764596849
    },
    {
      "filename": "hippocampus_026.nii.gz",
      "dice": 0.8971830985915493,
      "jaccard": 0.8135376756066411
    },
    {
      "filename": "hippocampus_314.nii.gz",
      "dice": 0.8410553410553411,
      "jaccard": 0.7257079400333148
    },
    {
      "filename": "hippocampus_268.nii.gz",
      "dice": 0.8759726854057488,
      "jaccard": 0.7793161910144109
    },
    {
      "filename": "hippocampus_084.nii.gz",
      "dice": 0.913100724160632,
      "jaccard": 0.8400969109630527
    },
    {
      "filename": "hippocampus_215.nii.gz",
      "dice": 0.918885827976737,
      "jaccard": 0.8499433748584372
    },
    {
      "filename": "hippocampus_235.nii.gz",
      "dice": 0.8366344477895737,
      "jaccard": 0.7191500953418687
    },
    {
      "filename": "hippocampus_204.nii.gz",
      "dice": 0.9220667012268878,
      "jaccard": 0.8554023725553062
    },
    {
      "filename": "hippocampus_044.nii.gz",
      "dice": 0.9131498470948012,
      "jaccard": 0.8401800787844682
    },
    {
      "filename": "hippocampus_104.nii.gz",
      "dice": 0.9076196898179366,
      "jaccard": 0.8308641975308642
    },
    {
      "filename": "hippocampus_039.nii.gz",
      "dice": 0.9147417051658967,
      "jaccard": 0.8428792569659442
    }
  ],
  "overall": {
    "mean_dice": 0.8851215520425222,
    "mean_jaccard": 0.7953986278217134
  },
  "config": {
    "name": "Basic_unet",
    "root_dir": "/home/workspace/data",
    "n_epochs": 2,
    "learning_rate": 0.0002,
    "batch_size": 8,
    "patch_size": 64,
    "test_results_dir": "/home/workspace/out"
  }
 }
 ```
 [Result](https://github.com/soyaoki/ND320-Hippocampal-Volume-Quantification-in-Alzheimers-Progression/blob/master/Section2/out/2022-09-21_1411_Basic_unet/results.json)