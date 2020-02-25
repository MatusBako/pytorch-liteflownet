import numpy as np

from collections import defaultdict
from configparser import ConfigParser, SectionProxy
from datetime import datetime
from os import mkdir
from sys import stderr, exc_info
from time import time
from torch import device, optim, zeros, mean, tensor, save
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from tqdm import tqdm
from typing import DefaultDict, List, Tuple

from .utils.drawer import Drawer
from .utils.logger import Logger
from .model.liteflownet import LiteFlowNet


class Solver:
    def __init__(self, config: ConfigParser):

        self._cfg = config
        nn_cfg: SectionProxy = config["CNN"]
        self.name = "LiteFlowCustom"

        self.iteration = 0
        self.device = device(nn_cfg['Device'])
        self.batch_size = nn_cfg.getint('BatchSize')
        self.learning_rate = nn_cfg.getfloat('LearningRate')

        self.iter_limit = nn_cfg.getint('IterationLimit')
        self.iter_per_snapshot = nn_cfg.getint('IterationsPerSnapshot')
        # self.iter_per_image = nn_cfg.getint('IterationsPerImage')
        self.iter_to_eval = nn_cfg.getint('IterationsToEvaluation')
        # self.test_iter = nn_cfg.getint('EvaluationIterations')

        timestamp = str(datetime.fromtimestamp(time()).strftime('%Y.%m.%d-%H:%M:%S'))
        self.output_folder = f"{nn_cfg['OutputFolder']}/{self.name}-{timestamp}"

        self.loss = MSELoss().to(self.device)

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

    def train_setup(self):
        self.net: LiteFlowNet = LiteFlowNet(6)
        self.net.train().to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.1)

        try:
            mkdir(self.output_folder)
        except Exception:
            print("Can't create output folder!", file=stderr)
            ex_type, ex_inst, ex_tb = exc_info()
            raise ex_type.with_traceback(ex_inst, ex_tb)

        self.drawer = Drawer(self.output_folder)
        self.logger = Logger(self.output_folder)

    def train(self, train_set: DataLoader):
        self.train_setup()

        train_values = []
        # loss_component_collector: DefaultDict[str, List] = defaultdict(list)

        progress_bar = tqdm(total=self.iter_limit)

        while self.iteration <= self.iter_limit:
            for _, (images1, images2, ground_truth) in enumerate(train_set):
                if len(images1) < self.batch_size:
                    continue

                images1, images2 = images1.to(self.device), images2.to(self.device)
                ground_truth = [flow.float().to(self.device) for flow in ground_truth]
                self.optimizer.zero_grad()

                flow = self.net(images1, images2)
                loss = self.compute_loss(flow, ground_truth)

                train_values.append(loss.item())

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.iteration += 1

                # ######## Statistics #########
                if self.iteration > 0 and self.iteration % self.iter_to_eval == 0:

                    # (test_loss, _), (psnr, psnr_diff, ssim), distances = self.evaluate(test_set)

                    if self.logger:
                        # components_line = [f"{k}: {round(np.mean(v), 5):.5f}" for k, v
                        #                    in loss_component_collector.items()]
                        line = " ".join([f"Iter:{self.iteration}",
                                         f"Train_loss:{np.round(np.mean(train_values), 5)}",
                                         ]) #+ components_line)
                        self.logger.log(line)

                    train_values.clear()
                    # loss_component_collector.clear()

                # snapshot
                if self.iter_per_snapshot is not None and self.iteration % self.iter_per_snapshot == 0 and self.iteration > 0:
                    self.save_model()

                if self.iteration > self.iter_limit:
                    break

                progress_bar.update()
        progress_bar.close()

    def compute_loss(self, computed_flow, ground_truth):
        loss = zeros(1)

        from itertools import repeat
        weights = repeat(1)
        # weights = [24, 16, 8, 4, 2, 1]
        losses = []

        for l1, l2, weight in zip(computed_flow, ground_truth, weights):
            losses.append(weight * self.loss(l1, l2))

        return mean(tensor(losses, requires_grad=True))






