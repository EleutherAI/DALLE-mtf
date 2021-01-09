# DALL-E in Mesh-Tensorflow [WIP]

Open-AI's [DALL-E](https://openai.com/blog/dall-e/) in Mesh-Tensorflow.

If our this is similarly efficient to [GPT-Neo](https://github.com/EleutherAI/gpt-neo/), this repo should be able to train models up to, and larger than, the size of Open-AI's DALL-E (12B params).

No pretrained models... Yet.

vanilla tensorflow vae (`train_vae_tf.py`) also available.

Thanks to [Ben Wang](https://github.com/kingoflolz) for the tf vae implementation as well as getting the mtf version working, and [Aran Komatsuzaki](https://github.com/AranKomat) for help building the mtf VAE and input pipeline.

# Setup

```bash
git clone https://github.com/EleutherAI/GPTNeo
cd GPTNeo
pip3 install -r requirements.txt
```
## Training Setup

Runs on TPUs, untested on GPUs but should work *in theory*. 
The example configs are designed to run on a TPU v3-32 pod.

To set up TPUs, sign up for [Google Cloud Platform](https://cloud.google.com/), and create a [storage bucket](https://cloud.google.com/storage). 

Create your VM through a google shell (`https://ssh.cloud.google.com/`) with `ctpu up --vm-only` so that it can connect to your Google bucket and TPUs and setup the repo as above.

## VAE pretraining

DALLE needs a pretrained VAE to compress images to tokens. To run the VAE pretraining, adjust the params in `configs/vae_example.json` to a glob path pointing to a dataset of jpgs, and adjust image size to the appropriate size.

```
  "dataset": {
    "train_path": "gs://neo-datasets/CIFAR-10-images/train/**/*.jpg",
    "eval_path": "gs://neo-datasets/CIFAR-10-images/test/**/*.jpg",
    "image_size": 32
  }
```

Once this is all set up, create your TPU, then run:

```bash
python train_vae.py --tpu your_tpu_name --model vae_example
```

The training logs image tensors and loss values, to check progress, you can run:

```bash
tensorboard --logdir your_model_dir
```

## Dataset Creation [DALL-E]

Once the VAE is pretrained, you can move on to DALL-E.

Currently we are training on a dummy dataset. A public, large-scale dataset for DALL-E is in the works. In the meantime, to generate some dummy data, run:

```bash
python data/create_tfrecords.py
```

This should download CIFAR-10, and generate some random captions to act as text inputs.

Custom datasets should be formatted in a folder, with a jsonl file in the root folder containing caption data and paths to the respective images, as follows:

```
Folder structure:

        data_folder
            jsonl_file
            folder_1
                img1
                img2
                ...
            folder_2
                img1
                img2
                ...
            ...

jsonl structure:
    {"image_path": folder_1/img1, "caption": "some words"}
    {"image_path": folder_2/img2, "caption": "more words"}
    ...
```

you can then use the `create_paired_dataset` function in `data/create_tfrecords.py` to encode the dataset into tfrecords for use in training.

Once the dataset is created, copy it over to your bucket with gsutil:

```bash
gsutil cp -r DALLE-tfrecords gs://neo-datasets/
```

And finally, run training with

```bash
python train_dalle.py --tpu your_tpu_name --model dalle_example
```

## Config Guide

VAE:

```
{
  "model_type": "vae",
  "dataset": {
    "train_path": "gs://neo-datasets/CIFAR-10-images/train/**/*.jpg", # glob path to training images
    "eval_path": "gs://neo-datasets/CIFAR-10-images/test/**/*.jpg", # glob path to eval images
    "image_size": 32 # size of images (all images will be cropped / padded to this size)
  },
  "train_batch_size": 32, 
  "eval_batch_size": 32,
  "predict_batch_size": 32,
  "steps_per_checkpoint": 1000, # how often to save a checkpoint
  "iterations": 500, # number of batches to infeed to the tpu at a time. Must be < steps_per_checkpoint
  "train_steps": 100000, # total training steps
  "eval_steps": 0, # run evaluation for this many steps every steps_per_checkpoint
  "model_path": "gs://neo-models/vae_test2/", # directory in which to save the model
  "mesh_shape": "data:16,model:2", # mapping of processors to named dimensions - see mesh-tensorflow repo for more info
  "layout": "batch_dim:data", # which named dimensions of the model to split across the mesh - see mesh-tensorflow repo for more info
  "num_tokens": 512, # vocab size
  "dim": 512, 
  "hidden_dim": 64, # size of hidden dim
  "n_channels": 3, # number of input channels
  "bf_16": false, # if true, the model is trained with bfloat16 precision
  "lr": 0.001, # learning rate [by default learning rate starts at this value, then decays to 10% of this value over the course of the training]
  "num_layers": 3, # number of blocks in the encoder / decoder
  "train_gumbel_hard": true, # whether to use hard or soft gumbel_softmax
  "eval_gumbel_hard": true
}
```

DALL-E:

```
{
  "model_type": "dalle",
  "dataset": {
    "train_path": "gs://neo-datasets/DALLE-tfrecords/*.tfrecords", # glob path to tfrecords data
    "eval_path": "gs://neo-datasets/DALLE-tfrecords/*.tfrecords",
    "image_size": 32 # size of images (all images will be cropped / padded to this size)
  },
  "train_batch_size": 32, # see above
  "eval_batch_size": 32,
  "predict_batch_size": 32,
  "steps_per_checkpoint": 1000,
  "iterations": 500,
  "train_steps": 100000,
  "predict_steps": 0,
  "eval_steps": 0,
  "n_channels": 3,
  "bf_16": false,
  "lr": 0.001,
  "model_path": "gs://neo-models/dalle_test/",
  "mesh_shape": "data:16,model:2",
  "layout": "batch_dim:data",
  "n_embd": 512, # size of embedding dim
  "text_vocab_size": 50258, # vocabulary size of the text tokenizer
  "image_vocab_size": 512, # vocabulary size of the vae - should equal num_tokens above
  "text_seq_len": 256, # length of text inputs (all inputs longer / shorter will be truncated / padded)
  "n_layers": 6, 
  "n_heads": 4, # number of attention heads. For best performance, n_embd / n_heads should equal 128
  "vae_model": "vae_example" # path to or name of vae model config
}
```
