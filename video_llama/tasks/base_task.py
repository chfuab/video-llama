"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from video_llama.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from video_llama.common.logger import MetricLogger, SmoothedValue
from video_llama.common.registry import registry
from video_llama.datasets.data_utils import prepare_sample
from video_llama.common.eval_metrics import metrics_mapping
from torch.nn import Softmax

class BaseTask:
    """ def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id" """
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"
        # self.caption_scorer = Caption_scorer()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, eps=1e-7)

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            dataset['train'].name = name
            if 'sample_ratio' in dataset_config:
                dataset['train'].sample_ratio = dataset_config.sample_ratio

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        if type(model).__name__ == "q_former_aligned":
            images = samples["image"]
            loss = model(images)["loss"]
        else:
            loss = model(samples)["loss"]
        return loss


    """def train_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss
    """
    """ def valid_step(self, model, samples):
        raise NotImplementedError """
    def valid_step(self, model, samples, metrics_name, model_name):
        if model_name == 'video_llama':
            if metrics_name == "accuracy":
                """ logits = model(samples)["logits"]
                # assume the logits is of shape (batch_size, seq_length, vocab_size)
                output_softmax = Softmax(dim=2)(logits)
                output_ids = torch.argmax(output_softmax, dim=2)
                output_ids_shape = tuple(output_ids.size())
                
                output_token_flatten = [model.llama_tokenizer._convert_id_to_token(id) for id in torch.flatten(output_ids).tolist()]
                
                batch_of_tokens = []
                tokens = []
                idx = 0
                for token in output_token_flatten:
                    tokens.append(token)
                    idx += 1
                    if idx % output_ids_shape[1] == 0 :
                        batch_of_tokens.append(tokens)
                        tokens = []
                all_string = []
                for token_batch in batch_of_tokens:
                    output_string = model.llama_tokenizer.convert_tokens_to_string(token_batch)
                    all_string.append(output_string)

                assert len(samples["correct_ans"]) == len(all_string), "number of label answers must equal to number of predicted answers" """
                
                """ max_score = len(samples["correct_ans"])
                score = 0
                for i, correct_ans in enumerate(samples["correct_ans"]):
                    if correct_ans == all_string[i]:
                        score += 1
                accuracy = score / max_score """
                embs = model(samples)["inference_embeds"]
                output = model.llama_model.generate(
                    inputs_embeds=embs,
                    max_new_tokens=20,
                    do_sample=False,
                )
                return {"accuracy": str(output)}
                # return {"accuracy": str(all_string)}
            
            elif metrics_name == "loss":
                return {"loss": model(samples)["loss"]}


    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    """ def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples) # load the best checkpoint of the model?
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results """

    
    def evaluation(self, model, data_loader, iters_per_epoch, metrics, model_name, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 50

        """ if model_name == 'q_former_aligned':
            if not hasattr(data_loader, "__next__"):
                # convert to iterator if not already
                data_loader = iter(data_loader)

            metric_logger = MetricLogger(delimiter="  ")
            

            ###
            if verify_q_former_aligned:
                metric_logger.add_meter("loss_verify_q_former", SmoothedValue(window_size=1, fmt="{value:.4f}"))
                metric_logger.add_meter("loss_general", SmoothedValue(window_size=1, fmt="{value:.4f}"))
            ###
            else:
                metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))


            for i in metric_logger.log_every(range(iters_per_epoch), print_freq, header):
                if i >= iters_per_epoch:
                    break
                samples = next(data_loader)
                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

                eval_output = {}
                for name in metrics:    # metrics for Q-former-aligned is loss only
                    with torch.cuda.amp.autocast(enabled=True):
                        eval_output_temp = self.valid_step(model=model, samples=samples, metrics_name=name, 
                                                           model_name=model_name, verify_q_former_aligned=verify_q_former_aligned)
                        eval_output.update(eval_output_temp)

                ###
                if verify_q_former_aligned:
                    metric_logger.update(loss_verify_q_former=eval_output['loss_verify_q_former'].item())
                    metric_logger.update(loss_general=eval_output['loss_general'].item())
                ###
                else:
                    metric_logger.update(loss=eval_output['loss'].item())
            # after train_epoch()
            # gather the stats from all processes

            metric_logger.synchronize_between_processes()
            logging.info("Averaged stats: " + str(metric_logger.global_avg()))

            return {
                k: "{:.3f}".format(meter.global_avg())
                for k, meter in metric_logger.meters.items()
            }, {
                k: meter.value_record
                for k, meter in metric_logger.meters.items()
            } """


        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('accuracy', SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt="{value:.4f}"))

        for i in metric_logger.log_every(range(iters_per_epoch), print_freq, header):
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            
            eval_output = {}
            # metrics are loss and accuracy
            for name in metrics:
                eval_output_temp = self.valid_step(model=model, samples=samples, metrics_name=name, model_name=model_name) # load the best checkpoint of the model?
                eval_output.update(eval_output_temp)

            metric_logger.update(accuracy=eval_output['accuracy'], loss=eval_output['loss'].item())

        logging_str = "Averaged stats: \n" + str(metric_logger.global_avg())    # getting avg over all batch size of samples in one epoch
        logging.info(logging_str)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return {
            k: meter.global_avg_1()
            for k, meter in metric_logger.meters.items()
        }, {
            k: meter.value_record
            for k, meter in metric_logger.meters.items()
        }


    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    """ def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        ) """


    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)
        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg())
            for k, meter in metric_logger.meters.items()
        }

    """ def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):

        # An inner training loop compatible with both epoch-based and iter-based training.

        # When using epoch-based, training stops after one epoch; when using iter-based,
        # training stops after #iters_per_epoch iterations.

        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        } """

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file



