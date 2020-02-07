# High accuracy interpolation of DEM using general adversarial network

This GitHub repository implements and evaluates a general adversarial method for Digital Elevation Model(DEM) interpolation with high resolution, which is an adaptation to the context of Digital Elevation Models (DEMs) from the method DeepFill v2 described in [1]. Pre-trained models are provided, as well as the DEMs used for the evaluation of the method.
[1] J. Yu, Z. Lin, J. Yang, X. Shen, X. Lu, and T. S. Huang, “Free-Form Image Inpainting with Gated Convolution,” 2018.

## Run

0. Requirements:
    * Install python3,PIL, opencv-python.
    * Install [tensorflow](https://www.tensorflow.org/install/) (tested on Release 1.3.0, 1.4.0, 1.5.0, 1.6.0, 1.7.0).
    * Install tensorflow toolkit [neuralgym](https://github.com/JiahuiYu/neuralgym) (run `pip install git+https://github.com/JiahuiYu/neuralgym`), then substitute data_from_fnames.py for neuralgym/neuralgym/data/data_from_fnames.py
1. Training:
    * Prepare training images filelist and shuffle it ([example](https://github.com/JiahuiYu/generative_inpainting/issues/15)).
    * Modify [inpain_dem.yml](/inpaint_dem.yml) to set DATA_FLIST, LOG_DIR, IMG_SHAPES and other parameters.
    * Run `python train.py`.
2. Resume training:
    * Modify MODEL_RESTORE flag in [inpaint_dem.yml](/inpaint_dem.yml). E.g., MODEL_RESTORE: 20180115220926508503_places2_model.
    * Run `python train.py`.
3. Testing:
    * Run `python batch_test.py --flist your_flist --checkpoint_dir your_model_dir  --outlist your_output`.
4. Still have questions?
    * If you still have questions (e.g.: How filelist looks like? How to use multi-gpus? How to do batch testing?), please first search over closed issues at https://github.com/JiahuiYu/generative_inpainting

## Pretrained models
run:
```bash
python batch_test.py --flist your_flist --checkpoint_dir ./pretrained_model/vallina_2-4-8-8-4-2dilated --outlist your_output
```

## TensorBoard

Visualization on [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) for training and validation is supported. Run `tensorboard --logdir model_logs` to view training progress.

## License

CC 4.0 Attribution-NonCommercial International

The software is for educaitonal and academic research purpose only.

## Citing
```
```
