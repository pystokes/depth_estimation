#!/usr/bin/python3
# -*- coding: utf-8 -*-

from logging import getLogger
import math
import torch
from torch import optim
from model.modules.loss_func import LossFunction
from utils.common import CommonUtils
from utils.optimizers import Optimizers

logger = getLogger('DepthEstimation')

class Trainer(object):

    def __init__(self, model, device, config, save_dir):
        
        self.model = model
        self.device = device
        self.config = config
        self.save_dir = save_dir


    def run(self, train_loader, valid_loader):

        loss_fn = LossFunction(self.config, self.device)
        optimizer = Optimizers.get_optimizer(self.config.train.optimizer, self.model.parameters())

        if self.config.train.optimizer.type == 'sgd':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.train.optimizer.sgd.T_max)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        logger.info('Begin training')
        for epoch in range(1, self.config.train.epoch+1):

            # For SGDR
            if self.config.train.optimizer.type == 'sgd':

                enable_scheduler = (epoch > self.config.train.optimizer.sgd.wait_decay_epoch)
                if epoch == self.config.train.optimizer.sgd.wait_decay_epoch + 1:
                    logger.info(f'Enable learning rate scheduler at Epoch: {epoch:05}')

                # Warm restart
                if enable_scheduler and (epoch % self.config.train.optimizer.sgd.T_max == 1):
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.config.train.optimizer.sgd.lr

            # Run train and validation
            train_depth_loss, train_grad_loss, train_loss = self._train(loss_fn, optimizer, train_loader)
            valid_depth_loss, valid_grad_loss, valid_loss = self._validate(loss_fn, valid_loader)

            # Scheduler
            if self.config.train.optimizer.type == 'sgd':
                if enable_scheduler:
                    scheduler.step(valid_loss)
            else:
                scheduler.step(valid_loss)

            # Show information
            train_loss_info = f'[Train Loss] Total: {train_loss:.5f}, Dep: {train_depth_loss:.5f}, Grad: {train_grad_loss:.5f}'
            valid_loss_info = f'[Valid Loss] Total: {valid_loss:.5f}, Dep: {valid_depth_loss:.5f}, Grad: {valid_grad_loss:.5f}'
            logger.info(f'Epoch [{epoch:05}/{self.config.train.epoch:05}], {train_loss_info}, {valid_loss_info}')

            # Save weights
            if epoch % self.config.train.weight_save_period == 0:
                save_path = self.save_dir.joinpath('weights', f'weight-{str(epoch).zfill(5)}_{train_loss:.5f}_{valid_loss:.5f}.pth')
                CommonUtils.save_weight(self.model, save_path)
                logger.info(f'Saved weight at Epoch : {epoch:05}')


    def _train(self, loss_fn, optimizer, train_data_loader):

        # Keep track of training loss
        depth_loss = 0.
        grad_loss = 0.
        train_loss = 0.

        # Train the model in each mini-batch
        self.model.train()
        for mini_batch in train_data_loader:

            # Send data to GPU dvice
            if self.device.type == 'cuda':
                images = mini_batch[0].to(self.device)
                depths = mini_batch[1].to(self.device)
            else:
                images = mini_batch[0]
                depths = mini_batch[1]
            
            # Forward
            optimizer.zero_grad()
            out_depths = self.model(images)
            dep_loss, grd_loss = loss_fn.calc_loss(out_depths, depths)
            loss = dep_loss + grd_loss

            # Backward and update weights
            loss.backward()
            optimizer.step()

            # Update training loss
            depth_loss += dep_loss.item()
            grad_loss += grd_loss.item()
            train_loss += loss.item()

        depth_loss /= len(train_data_loader.dataset)
        grad_loss /= len(train_data_loader.dataset)
        train_loss /= len(train_data_loader.dataset)

        return depth_loss, grad_loss, train_loss


    def _validate(self, loss_fn, valid_data_loader):

        # Keep track of validation loss
        depth_loss = 0.
        grad_loss = 0.
        valid_loss = 0.

        # Not use gradient for inference
        self.model.eval()
        with torch.no_grad():

            # Validate in each mini-batch
            for mini_batch in valid_data_loader:

                # Send data to GPU dvice
                if self.device.type == 'cuda':
                    images = mini_batch[0].to(self.device)
                    depths = mini_batch[1].to(self.device)
                else:
                    images = mini_batch[0]
                    depths = mini_batch[1]

                # Forward
                out_depths = self.model(images)
                dep_loss, grd_loss = loss_fn.calc_loss(out_depths, depths)
                loss = dep_loss + grd_loss

                # Update validation loss
                depth_loss += dep_loss.item()
                grad_loss += grd_loss.item()
                valid_loss += loss.item()

        depth_loss /= len(valid_data_loader.dataset)
        grad_loss /= len(valid_data_loader.dataset)
        valid_loss /= len(valid_data_loader.dataset)

        return depth_loss, grad_loss, valid_loss
