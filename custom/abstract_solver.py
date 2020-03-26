import numpy as np
import torch

from abc import ABC, abstractmethod
from collections import defaultdict
from configparser import ConfigParser, SectionProxy
from datetime import datetime
from inspect import getfile
from os import mkdir, listdir
from os.path import dirname, join
from sys import stderr, exc_info
from shutil import copyfile
from time import time
from torch import device, optim, zeros, mean, tensor, save, load
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from tqdm import tqdm
from typing import DefaultDict, List, Tuple, Optional

from .utils.drawer import Drawer
from .utils.logger import Logger
from .factory import build_optimizer, build_scheduler


def measure_speed(model, *inputs, iter_cnt=10):
    model = model.eval()
    with torch.cuda.profiler.profile():
        with torch.no_grad():
            t1 = time()
            for i in range(iter_cnt):
                out = model(*inputs)[-1].cpu().numpy()
            t2 = time()

            with torch.autograd.profiler.emit_nvtx():
                out = model(*inputs)[-1].cpu().numpy()

    model = model.train()
    return (t2 - t1) / iter_cnt


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t1 = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t2 = time()
        diff = self.t2 - self.t1
        print(f"{self.label}: {diff:.5f}")


class AbstractSolver(ABC):
    def __init__(self, config: ConfigParser):

        self._cfg = config
        nn_cfg: SectionProxy = config["CNN"]

        self.net: Optional[Module] = None
        self.drawer = None
        self.logger = None

        self.iteration = 0
        self.device = device(nn_cfg['Device'])
        self.batch_size = nn_cfg.getint('BatchSize')
        self.learning_rate = nn_cfg.getfloat('LearningRate')

        self.iter_limit = nn_cfg.getint('IterationLimit')
        self.iter_per_snapshot = nn_cfg.getint('IterationsPerSnapshot')
        self.iter_per_image = nn_cfg.getint('IterationsPerImage')
        self.iter_to_eval = nn_cfg.getint('IterationsToEvaluation')
        # self.test_iter = nn_cfg.getint('EvaluationIterations')

        timestamp = str(datetime.fromtimestamp(time()).strftime('%Y.%m.%d-%H:%M:%S'))
        self.output_folder = None if "OutputFolder" not in nn_cfg else f"{nn_cfg['OutputFolder']}/{self.name}-{timestamp}"

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def get_net_instance(self):
        pass

    @abstractmethod
    def compute_loss(self, output, target):
        pass

    def save_model(self):
        checkpoint_path = f"{self.output_folder}/snapshot-{self.iteration}.mdl"
        dic = {
            'iteration': self.iteration,
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

        save(dic, checkpoint_path)
        self.net.to(self.device)

    def load_model(self, checkpoint_path):
        checkpoint = load(checkpoint_path)

        self.net.load_state_dict(checkpoint["model"])

        # add level after trained network
        if "AddLevels" in self._cfg['CNN']:
            new_levels = self._cfg['CNN'].getint("AddLevels")

            for _ in range(new_levels):
                self.net.add_level()

        self.optimizer = build_optimizer(self._cfg['Optimizer'], self.net.parameters())
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if 'Scheduler' in self._cfg:
            self.scheduler = build_scheduler(self._cfg['Scheduler'], self.optimizer)
            self.scheduler.load_state_dict(checkpoint["scheduler"])


    def train_setup(self):
        self.optimizer = build_optimizer(self._cfg['Optimizer'], self.net.parameters())
        self.scheduler = build_scheduler(self._cfg['Scheduler'], self.optimizer)
        self.net.to(self.device)

        if self.output_folder is not None:
            try:
                mkdir(self.output_folder)
            except Exception:
                print("Can't create output folder!", file=stderr)
                ex_type, ex_inst, ex_tb = exc_info()
                raise ex_type.with_traceback(ex_inst, ex_tb)

            self.drawer = Drawer(self.output_folder)
            self.logger = Logger(self.output_folder)

            # save config
            with open(self.output_folder + '/config.ini', 'w') as f:
                self._cfg.write(f)

            # save code
            # copy all python modules found in directory of trained model
            module_folder = dirname(getfile(self.net.__class__))
            files = [file for file in listdir(module_folder) if file.endswith(".py")]

            for file in files:
                copyfile(join(module_folder, file), join(self.output_folder, file))

    def train(self, train_set: DataLoader):
        self.train_setup()

        train_values = []
        # loss_component_collector: DefaultDict[str, List] = defaultdict(list)

        progress_bar = tqdm(total=self.iter_limit)

        while self.iteration < self.iter_limit:
            for _, (images1, images2, ground_truth) in enumerate(train_set):
                if len(images1) < self.batch_size:
                    continue

                images1, images2 = images1.to(self.device), images2.to(self.device)
                ground_truth = [flow.float().to(self.device) for flow in ground_truth]
                self.optimizer.zero_grad()

                # print(measure_speed(self.net, images1, images2, iter_cnt=20))
                # break
                flow = self.net(images1, images2)
                loss = self.compute_loss(flow, ground_truth)

                train_values.append(loss.item())

                loss.backward()
                self.optimizer.step()

                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    old_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                    self.scheduler.step(loss)
                    new_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                    if old_lr != new_lr and self.logger:
                        self.logger.log("LearningRateAdapted")
                else:
                    self.scheduler.step()

                self.iteration += 1
                # ######## Statistics #########
                if self.iteration > 0 and self.iteration % self.iter_to_eval == 0:
                    # (test_loss, _), (psnr, psnr_diff, ssim), distances = self.evaluate(test_set)

                    pixel_per_sec = (images1.shape[0] * images1.shape[2] * images1.shape[3] / 1_000_000) / measure_speed(self.net, images1, images2)

                    if self.logger:
                        # components_line = [f"{k}: {round(np.mean(v), 5):.5f}" for k, v
                        #                    in loss_component_collector.items()]
                        line = " ".join([f"Iter:{self.iteration}",
                                         f"Train_loss:{np.round(np.mean(train_values), 5)}",
                                         f"MPx_per_sec:{pixel_per_sec:.4f}"
                                         ]) #+ components_line)
                        self.logger.log(line)

                    train_values.clear()
                    # loss_component_collector.clear()

                # store training collage
                if self.drawer and self.iteration % self.iter_per_image == 0:
                    images1 = images1.cpu().numpy().transpose((0, 2, 3, 1)) * 256
                    images2 = images2.cpu().numpy().transpose((0, 2, 3, 1)) * 256
                    result = flow[-1].detach().cpu().numpy().transpose((0, 2, 3, 1))
                    target = ground_truth[-1].cpu().numpy().transpose((0, 2, 3, 1))

                    self.drawer.save_images(images1, images2, result, target, "Train-" + str(self.iteration))

                # snapshot
                if self.iter_per_snapshot is not None \
                        and self.iteration % self.iter_per_snapshot == 0 \
                        and self.iteration > 0 \
                        and self.output_folder is not None:
                    self.save_model()

                if self.iteration > self.iter_limit:
                    break

                progress_bar.update()
        progress_bar.close()







