import tensorflow.compat.v1 as tf
try:
    from .tokenizer_utils import get_tokenizer
except ImportError:
    from tokenizer_utils import get_tokenizer
import json
from pathlib import PurePath, Path
import cv2
from tqdm import tqdm
import glob
import random
import os
import shutil


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')


def load_jsonl(input_path):
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    return data


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(image, caption):
    feature = {
        'image': _bytes_feature(image),
        'caption': _int64_feature(caption),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def create_random_dataset(path_to_images, out_dir, max_images_per_folder=1000, words_per_caption=50):
    """
    creates a paired image / text folder with random captions in the correct format to feed to
    create_paired_tfrecord_dataset (for testing)

    Args:
        out_dir: str
        path_to_images: str
            glob path to images
        max_images_per_folder: int
        words_per_caption: int

    """
    import requests
    word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
    response = requests.get(word_site)
    WORDS = response.content.splitlines()

    out_dir = Path(out_dir)
    jsonl_path = out_dir / "captions_data.jsonl"
    os.makedirs(out_dir, exist_ok=True)

    images = glob.glob(path_to_images)
    print(f"{len(images)} images found")
    pbar = tqdm()
    folder_count = 0
    for i, image in enumerate(images):
        if i % 100 == 0 or i == 0:
            pbar.update(100)
        if i % max_images_per_folder == 0 or i == 0:
            sub_folder = Path(out_dir) / str(folder_count)
            os.makedirs(Path(out_dir) / str(folder_count), exist_ok=True)
            folder_count += 1
        data = {}
        image = Path(image)
        data["caption"] = " ".join([random.choice(WORDS).decode() for i in range(words_per_caption)])
        data["image_path"] = str(sub_folder.relative_to(out_dir) / image.name)
        shutil.copy(image, sub_folder)
        dump_jsonl([data], jsonl_path, append=True)


def create_paired_dataset(path_to_jsonl, name, out_dir, examples_per_tfrecord=1000, tokenizer=None, reencode=False):
    """
    takes in a jsonl with relative paths to images & captions, and saves tfrecords files with num_examples
    examples to out_dir.

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

    Jsonl structure:
        {"image_path": relative_image_path, "caption": caption}
        {"image_path": relative_image_path, "caption": caption}
        ...

    TODO: multiprocessing

    Args:
        path_to_jsonl: str / path  / list of str / path
            path to jsonl file

        examples_per_tfrecord: int
            number of examples to write to each tfrecords file

        name: str
            name of tfrecords files

        out_dir: str / path
            path to folder in which to save tfrecords

        tokenizer: custom HF tokenizer
            if None, defaults to GPT2TokenizerFast

    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)
        os.makedirs(out_dir, exist_ok=True)
    if isinstance(path_to_jsonl, PurePath) or isinstance(path_to_jsonl, str):
        path_to_jsonl = [path_to_jsonl]
    if not isinstance(path_to_jsonl, list):
        raise TypeError(f"path_to_jsonl type not recognized, should be str, path, or list")
    tfrecord_count = 0
    example_count = 0
    writer = tf.io.TFRecordWriter(str(out_dir / f"{name}_{tfrecord_count}.tfrecords"))
    pbar = tqdm()
    for path in path_to_jsonl:
        path = Path(path)
        data = load_jsonl(path)
        for item in data:
            if example_count % examples_per_tfrecord == 0 and example_count != 0:
                writer.close()
                writer = tf.io.TFRecordWriter(str(out_dir / f"{name}_{tfrecord_count}.tfrecords"))
                tfrecord_count += 1
            image_path = path.parent / item["image_path"]
            if reencode:
                img = cv2.imread(str(image_path))
                img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()  # encodes image to string
            else:
                img = open(image_path, "rb").read()

            caption = tokenizer.encode(item["caption"][0])
            example = serialize_example(img, caption)
            writer.write(example)
            example_count += 1
            if example_count % 100 == 0:
                pbar.set_description(f"{example_count} examples written to {tfrecord_count + 1} files")
                pbar.update(100)
    writer.close()


if __name__ == "__main__":
    # creates random test dataset with CIFAR 10
    create_paired_dataset("/home/data/coco/coco_captions.jsonl", "COCO", "DALLE-tfrecords", examples_per_tfrecord=1000,
                          tokenizer=None)
