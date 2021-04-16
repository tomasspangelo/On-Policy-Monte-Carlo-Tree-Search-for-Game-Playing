import tensorflow as tf
import os
import glob
import sys


def custom_cross_entropy(targets, outs):
    """
    Custom cross entropy function that can be used as the loss
    function during the training of a tensorflow/keras model.
    :param targets: Numpy array containing targets.
    :param outs: Numpy array containing output of model.
    :return: Cross Entropy Loss
    """
    return tf.reduce_mean(tf.reduce_sum(-1 * targets * safelog(outs), axis=-1))


def safelog(tensor, base=0.0001):
    """
    Safe logarithm, takes borderline case where
    tensor=0.
    :param tensor: Input tensor
    :param base: Small float to be added.
    :return: Safe logarithm of tensor.
    """
    return tf.math.log(tf.math.maximum(tensor, base))


def upload_local_directory_to_gcs(local_path, bucket, gcs_path):
    """
    Upload an entire hierarchy to Google Cloud Storage.
    :param local_path: Local path (everything down in the hierarchy will be saved to GCS).
    :param bucket: The bucket to save to (should be bucket object from google API)
    :param gcs_path: Path to save to in GCS.
    :return:
    """
    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
            upload_local_directory_to_gcs(local_file, bucket, gcs_path + "/" + os.path.basename(local_file))
        else:
            remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)


def start_progress(title):
    """
    Start to display progress bar.
    :param title: Title of the process going on.
    :return: None
    """
    global progress_x
    sys.stdout.write(title + ": [" + "-" * 50 + "]" + chr(8) * 51)
    sys.stdout.flush()
    progress_x = 0


def progress(x, total):
    """
    Notify system that process has progressed.
    :param x: Progress
    :param total: Total "length" of progress.
    :return: None
    """
    global progress_x
    x = int((x / total) * 50)
    sys.stdout.write("#" * (x - progress_x))
    sys.stdout.flush()
    progress_x = x


def end_progress(timeout=False):
    """
    End progress (i.e. "close off" progress bar)
    :param timeout: True if progress didn't finish, False otherwise.
    :return: None
    """
    if timeout:
        sys.stdout.write("-" * (50 - progress_x) + "]\n")
    else:
        sys.stdout.write("#" * (50 - progress_x) + "]\n")
    sys.stdout.flush()
