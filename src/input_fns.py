import tensorflow.compat.v1 as tf


def crop_center_and_resize(img, size):
    s = tf.shape(img)
    w, h = s[0], s[1]
    c = tf.maximum(w, h)
    wn, hn = h / c, w / c
    result = tf.image.crop_and_resize(tf.expand_dims(img, 0),
                                      [[(1 - wn) / 2, (1 - hn) / 2, wn, hn]],
                                      [0], [size, size])
    return tf.squeeze(result, 0)


def decode_img(img, size, channels=3):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=channels)
    # resize the image to the desired size
    img = crop_center_and_resize(img, size)
    img = (tf.cast(img, tf.float32) - 127.5) / 127.5
    return img


def configure_for_performance(ds, params, eval=False):
    if not eval:
        ds = ds.shuffle(buffer_size=params["batch_size"] * 5)
    ds = ds.batch(params["batch_size"], drop_remainder=True)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def truncate_or_pad_label(label, params):
    # pad by sequence length and gather the first sequence length items
    # definitely a more efficient way to do this
    label = tf.pad(label, [[0, params["text_seq_len"]]], constant_values=params["padding_id"])
    label = tf.gather(label, tf.range(params["text_seq_len"]))
    label = tf.reshape(label, [params["text_seq_len"]])
    return label


def read_labeled_tfrecord(params):
    def read_fn(example):
        features = {
            "image": tf.FixedLenFeature([], tf.string),
            "caption": tf.VarLenFeature(tf.int64),
        }
        example = tf.parse_single_example(example, features)
        label = tf.sparse.to_dense(example["caption"], example["caption"].dense_shape[0])
        image = decode_img(example["image"], params["dataset"]["image_size"], params["n_channels"])
        label = truncate_or_pad_label(label, params)
        label = tf.cast(label, tf.int32)
        return image, label  # returns a dataset of (image, label) pairs

    return read_fn


def read_tfrecord(params):
    def read_fn(example):
        features = {
            "image": tf.FixedLenFeature([], tf.string),
        }
        example = tf.parse_single_example(example, features)
        image = decode_img(example["image"], params["dataset"]["image_size"], params["n_channels"])
        return image, image  # returns image twice because they expect 2 returns

    return read_fn


def vae_input_fn(params, eval=False):
    path = params["dataset"]["train_path"] if not eval else params["dataset"]["eval_path"]

    if "tfrecords" in params["dataset"] and params["dataset"]["tfrecords"]:
        files = tf.io.gfile.glob(path)
        file_count = len(files)
        tf.logging.info(path)
        tf.logging.info(f'FILE COUNT: {file_count}')
        dataset = tf.data.Dataset.from_tensor_slices(files)
        if not eval:
            dataset = dataset.shuffle(file_count, reshuffle_each_iteration=False)
        dataset = dataset.apply(
            tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4, sloppy=False))
        parse_fn = read_tfrecord(params)
        dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = configure_for_performance(dataset, params, eval)
        return dataset.repeat()
    else:
        files = tf.data.Dataset.list_files(path, shuffle=False)
        image_count = len(tf.io.gfile.glob(path))
        tf.logging.info(path)
        tf.logging.info(f'IMAGE COUNT: {image_count}')

        if not eval:
            files = files.shuffle(image_count, reshuffle_each_iteration=False)
        img_size = params["dataset"]["image_size"]

        def _process_path(file_path):
            img = tf.io.read_file(file_path)
            img = decode_img(img, img_size)
            # TODO: figure out if we can do away with the fake labels
            return img, img

        dataset = files.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = configure_for_performance(dataset, params, eval)
        return dataset.repeat()

def dalle_input_fn(params, eval=False):
    path = params["dataset"]["train_path"] if not eval else params["dataset"]["eval_path"]
    files = tf.io.gfile.glob(path)
    file_count = len(files)
    tf.logging.info(path)
    tf.logging.info(f'FILE COUNT: {file_count}')
    dataset = tf.data.Dataset.from_tensor_slices(files)

    if not eval:
        dataset = dataset.shuffle(file_count, reshuffle_each_iteration=False)
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4, sloppy=False))
    parse_fn = read_labeled_tfrecord(params)
    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = configure_for_performance(dataset, params, eval)
    return dataset.repeat()
