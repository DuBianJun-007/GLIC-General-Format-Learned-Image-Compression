import os
import shutil
import torch
from torch import nn


def get_run_count(count=0, new_train=True):
    data_set_path = './checkpoint'
    existing_runs = [name for name in os.listdir(data_set_path) if name.startswith('run_')]
    if new_train:  # new training
        run_count = 'run_00'
        if existing_runs:
            latest_run = max(existing_runs, key=lambda x: int(x.split('_')[1]))
            num = int(latest_run.split('_')[1]) + 1
            num = str(num).zfill(2) if num < 10 else str(num)
            run_count = 'run_' + num
        run_path = new_run = os.path.join(data_set_path, run_count)
        os.makedirs(new_run)
        os.makedirs(os.path.join(new_run, 'checkpoint'))
        os.makedirs(os.path.join(new_run, 'best_checkpoint'))
        os.makedirs(os.path.join(new_run, 'updataModel'))
        os.makedirs(os.path.join(new_run, 'log'))
    else:  # continue for training
        count = str(count).zfill(2) if int(count) < 10 else str(count)
        if 'run_' + count in existing_runs:
            run_path = os.path.join(data_set_path, 'run_' + count)
        else:
            raise Exception("The run file does not exist")
    return run_path


def get_checkpoint_from_runpath(run_path):
    checkpoint_path = os.path.join(run_path, 'checkpoint')
    checkpoint_path_list = os.listdir(checkpoint_path)  # checkpoint-01.pth.tar
    return os.path.join(checkpoint_path, checkpoint_path_list[-1])


def save_checkpoint(state, is_best, run_path):
    init_checkpoint = 'checkpoint-000.pth.tar'
    check_dir = os.path.join(run_path, 'checkpoint')
    existing_checkpoints = [name for name in os.listdir(check_dir) if name.startswith('checkpoint')]
    checkpoint_num = [int(name.split('.')[0][-3:]) for name in existing_checkpoints]
    all_checkpointFile = os.listdir(check_dir)
    num = int(0)
    if checkpoint_num:
        num_max = max(checkpoint_num)
        num = num_max + 1
        num = str(num).zfill(3) if num < 100 else str(num)
        new_checkpoint_name = 'checkpoint-{}.pth.tar'.format(num)
        # Store the new checkpoint file
        save_path = os.path.join(check_dir, new_checkpoint_name)
    else:
        save_path = os.path.join(check_dir, init_checkpoint)
    torch.save(state, save_path)
    # Delete previous files
    delete_files(all_checkpointFile, check_dir)
    if is_best:
        best_path = os.path.join(run_path, 'best_checkpoint')
        best_checkpoint_filename = "checkpoint_best_loss.pth.tar"
        best_checkpoint_path = os.path.join(best_path, best_checkpoint_filename)
        shutil.copyfile(save_path, best_checkpoint_path)


def delete_files(file_list, directory):
    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error while deleting file: {file_path} - {e}")


def log_training_stats(writer, out_criterion, epoch):
    for key, value in out_criterion.items():
        writer.add_scalar(key, value, epoch)


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)
