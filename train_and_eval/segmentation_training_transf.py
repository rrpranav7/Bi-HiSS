import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import copy
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import wandb
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_loss_data_input, get_model_data_input


# Metrics aggregated (mean ± std) across seeds at the end of the run.
_AGG_METRIC_KEYS = [
    ("macro", "IOU"),
    ("macro", "Accuracy"),
    ("macro", "Precision"),
    ("macro", "Recall"),
    ("macro", "F1"),
    ("micro", "IOU"),
    ("micro", "Accuracy"),
    ("micro", "Precision"),
    ("micro", "Recall"),
    ("micro", "F1"),
]


def set_seed(seed):
    """Seed python/numpy/torch (and CUDA) so each seed produces a different run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_and_evaluate(net, dataloaders, config, device, lin_cls=False,
                       wandb_run_name=None, run_tag=None, resume_epoch=None):
    """Train `net` on `dataloaders['train']` and evaluate on `dataloaders['eval']`.

    Returns a dict of the best eval metrics (macro + micro variants) observed
    during training. The multi-seed driver collects these across seeds and
    prints mean/std at the end.
    """

    def train_step(net, sample, loss_fn, optimizer, device, loss_input_fn, model_input_fn):
        optimizer.zero_grad()
        outputs = net(model_input_fn(sample, device))
        outputs = outputs.permute(0, 2, 3, 1)
        ground_truth = loss_input_fn(sample, device)
        loss = loss_fn['mean'](outputs, ground_truth)
        loss.backward()
        optimizer.step()
        return outputs, ground_truth, loss

    def evaluate(net, evalloader, loss_fn, config):
        num_classes = config['MODEL']['num_classes']
        predicted_all = []
        labels_all = []
        losses_all = []
        net.eval()
        with torch.no_grad():
            for step, sample in enumerate(evalloader):
                logits = net(model_input_fn(sample, device))
                logits = logits.permute(0, 2, 3, 1)
                _, predicted = torch.max(logits.data, -1)
                ground_truth = loss_input_fn(sample, device)
                loss = loss_fn['all'](logits, ground_truth)
                target, mask = ground_truth
                if mask is not None:
                    predicted_all.append(predicted.view(-1)[mask.view(-1)].cpu().numpy())
                    labels_all.append(target.view(-1)[mask.view(-1)].cpu().numpy())
                else:
                    predicted_all.append(predicted.view(-1).cpu().numpy())
                    labels_all.append(target.view(-1).cpu().numpy())
                losses_all.append(loss.view(-1).cpu().detach().numpy())

        print("finished iterating over dataset after step %d" % step)
        print("calculating metrics...")
        predicted_classes = np.concatenate(predicted_all)
        target_classes = np.concatenate(labels_all)
        losses = np.concatenate(losses_all)

        eval_metrics = get_classification_metrics(predicted=predicted_classes, labels=target_classes,
                                                  n_classes=num_classes, unk_masks=None)

        micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics['micro']
        macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics['macro']
        class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics['class']

        un_labels, class_loss = get_per_class_loss(losses, target_classes, unk_masks=None)

        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("Mean (micro) Evaluation metrics (micro/macro), loss: %.7f, iou: %.4f/%.4f, accuracy: %.4f/%.4f, "
              "precision: %.4f/%.4f, recall: %.4f/%.4f, F1: %.4f/%.4f, unique pred labels: %s" %
              (losses.mean(), micro_IOU, macro_IOU, micro_acc, macro_acc, micro_precision, macro_precision,
               micro_recall, macro_recall, micro_F1, macro_F1, np.unique(predicted_classes)))
        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------------------------------")

        return (un_labels,
                {"macro": {"Loss": losses.mean(), "Accuracy": macro_acc, "Precision": macro_precision,
                           "Recall": macro_recall, "F1": macro_F1, "IOU": macro_IOU},
                 "micro": {"Loss": losses.mean(), "Accuracy": micro_acc, "Precision": micro_precision,
                           "Recall": micro_recall, "F1": micro_F1, "IOU": micro_IOU},
                 "class": {"Loss": class_loss, "Accuracy": class_acc, "Precision": class_precision,
                           "Recall": class_recall,
                           "F1": class_F1, "IOU": class_IOU}}
                )

    #------------------------------------------------------------------------------------------------------------------#
    num_classes = config['MODEL']['num_classes']
    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    save_steps = config['CHECKPOINT']["save_steps"]
    save_path = config['CHECKPOINT']["save_path"]
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    num_steps_train = len(dataloaders['train'])
    local_device_ids = config['local_device_ids']
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)

    dataset_name = config['DATASETS']['train']['dataset']
    model_arch = config['MODEL']['architecture']
    if wandb_run_name is None:
        wandb_run_name = f"{model_arch}_{dataset_name}"

    wandb.init(
        project="Segmentation",
        name=wandb_run_name,
        reinit=True,
        config={
            "model": config['MODEL'],
            "solver": config['SOLVER'],
            "dataset": dataset_name,
            "num_classes": num_classes,
            "num_epochs": num_epochs,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "run_tag": run_tag,
        },
    )

    start_global = 1
    start_epoch = 1
    end_epoch = start_epoch + num_epochs
    if resume_epoch is not None:
        # Continue a killed run: epochs resume_epoch..num_epochs, keeping the original step numbering.
        start_epoch = resume_epoch
        start_global = (resume_epoch - 1) * num_steps_train + 1
        end_epoch = num_epochs + 1
    if checkpoint:
        load_from_checkpoint(net, checkpoint, partial_restore=False)

    print("current learn rate: ", lr)

    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)

    if save_path and (not os.path.exists(save_path)):
        os.makedirs(save_path)

    copy_yaml(config)

    loss_input_fn = get_loss_data_input(config)
    model_input_fn = get_model_data_input(config)

    loss_fn = {'all': get_loss(config, device, reduction=None),
               'mean': get_loss(config, device, reduction="mean")}

    trainable_params = get_net_trainable_params(net)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    optimizer.zero_grad()

    scheduler = build_scheduler(config, optimizer, num_steps_train)

    writer = SummaryWriter(save_path)

    BEST_IOU = 0.0
    best_eval = None  # Full eval metrics dict at the best mIoU checkpoint.
    if resume_epoch is not None:
        scheduler.step_update(start_global - 1)
        print("resuming at epoch %d, abs_step %d, lr %.3e" % (start_epoch, start_global, optimizer.param_groups[0]["lr"]))
        resumed_eval = evaluate(net, dataloaders['eval'], loss_fn, config)
        BEST_IOU = resumed_eval[1]['macro']['IOU']
        best_eval = resumed_eval[1]
        print("mIoU of loaded checkpoint: %.4f" % BEST_IOU)
    net.train()
    for epoch in range(start_epoch, end_epoch):  # loop over the dataset multiple times
        for step, sample in enumerate(dataloaders['train']):
            abs_step = start_global + (epoch - start_epoch) * num_steps_train + step
            logits, ground_truth, loss = train_step(
                net, sample, loss_fn, optimizer, device,
                loss_input_fn=loss_input_fn, model_input_fn=model_input_fn)
            if len(ground_truth) == 2:
                labels, unk_masks = ground_truth
            else:
                labels = ground_truth
                unk_masks = None
            if abs_step % train_metrics_steps == 0:
                logits = logits.permute(0, 3, 1, 2)
                batch_metrics = get_mean_metrics(
                    logits=logits, labels=labels, unk_masks=unk_masks, n_classes=num_classes, loss=loss, epoch=epoch,
                    step=step)
                write_mean_summaries(writer, batch_metrics, abs_step, mode="train", optimizer=optimizer)
                wandb.log({
                    "train/loss": float(loss),
                    "train/OA": batch_metrics['Accuracy'],
                    "train/mIoU": batch_metrics['IOU'],
                    "train/precision": batch_metrics['Precision'],
                    "train/recall": batch_metrics['Recall'],
                    "train/F1": batch_metrics['F1'],
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "step": abs_step,
                })
                print("abs_step: %d, epoch: %d, step: %5d, loss: %.7f, batch_iou: %.4f, batch accuracy: %.4f, batch precision: %.4f, "
                      "batch recall: %.4f, batch F1: %.4f" %
                      (abs_step, epoch, step + 1, loss, batch_metrics['IOU'], batch_metrics['Accuracy'], batch_metrics['Precision'],
                       batch_metrics['Recall'], batch_metrics['F1']))

            if abs_step % save_steps == 0:
                if len(local_device_ids) > 1:
                    torch.save(net.module.state_dict(), "%s/%depoch_%dstep.pth" % (save_path, epoch, abs_step))
                else:
                    torch.save(net.state_dict(), "%s/%depoch_%dstep.pth" % (save_path, epoch, abs_step))

            if abs_step % eval_steps == 0:
                eval_metrics = evaluate(net, dataloaders['eval'], loss_fn, config)
                if eval_metrics[1]['macro']['IOU'] > BEST_IOU:
                    if len(local_device_ids) > 1:
                        torch.save(net.module.state_dict(), "%s/best.pth" % (save_path))
                    else:
                        torch.save(net.state_dict(), "%s/best.pth" % (save_path))
                    BEST_IOU = eval_metrics[1]['macro']['IOU']
                    best_eval = eval_metrics[1]

                micro = eval_metrics[1]['micro']
                macro = eval_metrics[1]['macro']
                class_metrics = eval_metrics[1]['class']
                un_labels = eval_metrics[0]

                wandb_eval_log = {
                    "eval/loss": micro['Loss'],
                    "eval_micro/OA": micro['Accuracy'],
                    "eval_micro/mIoU": micro['IOU'],
                    "eval_micro/precision": micro['Precision'],
                    "eval_micro/recall": micro['Recall'],
                    "eval_micro/F1": micro['F1'],
                    "eval_macro/OA": macro['Accuracy'],
                    "eval_macro/mIoU": macro['IOU'],
                    "eval_macro/precision": macro['Precision'],
                    "eval_macro/recall": macro['Recall'],
                    "eval_macro/F1": macro['F1'],
                    "eval/best_mIoU": BEST_IOU,
                    "eval/OA_at_best_mIoU": (best_eval['micro']['Accuracy']
                                             if best_eval is not None else 0.0),
                    "step": abs_step,
                }

                for i, label in enumerate(un_labels):
                    label_str = str(label)
                    wandb_eval_log[f"eval_class/accuracy_{label_str}"] = class_metrics['Accuracy'][i]
                    wandb_eval_log[f"eval_class/precision_{label_str}"] = class_metrics['Precision'][i]
                    wandb_eval_log[f"eval_class/recall_{label_str}"] = class_metrics['Recall'][i]
                    wandb_eval_log[f"eval_class/F1_{label_str}"] = class_metrics['F1'][i]
                    wandb_eval_log[f"eval_class/IoU_{label_str}"] = class_metrics['IOU'][i]
                    wandb_eval_log[f"eval_class/loss_{label_str}"] = class_metrics['Loss'][i]

                wandb.log(wandb_eval_log)

                write_mean_summaries(writer, eval_metrics[1]['micro'], abs_step, mode="eval_micro", optimizer=None)
                write_mean_summaries(writer, eval_metrics[1]['macro'], abs_step, mode="eval_macro", optimizer=None)
                write_class_summaries(writer, [eval_metrics[0], eval_metrics[1]['class']], abs_step, mode="eval",
                                      optimizer=None)
                net.train()

        scheduler.step_update(abs_step)

    wandb.finish()

    best_metrics = {}
    if best_eval is not None:
        for group, key in _AGG_METRIC_KEYS:
            best_metrics[f"{group}/{key}"] = float(best_eval[group][key])
    else:
        for group, key in _AGG_METRIC_KEYS:
            best_metrics[f"{group}/{key}"] = float("nan")
    return best_metrics


def _mean_std(values):
    arr = np.array([v for v in values if not (isinstance(v, float) and np.isnan(v))], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), (float(arr.std(ddof=0)) if arr.size > 1 else 0.0)


def _print_multi_seed_table(config_file, seeds, runs):
    """Print a mean ± std table over the seeds for a single config."""
    print("\n" + "=" * 100)
    print(f"Multi-seed summary   config={config_file}   seeds={list(seeds)}")
    print("=" * 100)
    print(f"{'Metric':<24}{'N':>4}   {'Mean':>10}   {'Std':>10}   {'Per-seed values':<}")
    print("-" * 100)
    for group, key in _AGG_METRIC_KEYS:
        name = f"{group}/{key}"
        vals = [r[name] for r in runs]
        mean, std = _mean_std(vals)
        vals_str = ", ".join(f"{v:.4f}" if not np.isnan(v) else "nan" for v in vals)
        print(f"{name:<24}{len(vals):>4}   {mean:>10.4f}   {std:>10.4f}   {vals_str}")
    print("=" * 100)


def run_multi_seed(config_file, seeds, device_ids, lin_cls=False):
    """Train the same config across multiple seeds and report mean/std of the
    best-eval metrics. PASTIS 5-fold aggregation is not done here — run this
    script once per fold and combine the printed means yourself."""
    device = get_device(device_ids, allow_cpu=False)

    runs = []
    for seed in seeds:
        print("\n" + "#" * 100)
        print(f"# Starting run: config={config_file}  seed={seed}")
        print("#" * 100)

        set_seed(seed)

        cfg = copy.deepcopy(read_yaml(config_file))
        cfg['local_device_ids'] = device_ids
        cfg['seed'] = seed

        base_save = cfg['CHECKPOINT']['save_path']
        cfg['CHECKPOINT']['save_path'] = f"{base_save}/seed{seed}"

        dataloaders = get_dataloaders(cfg)
        net = get_model(cfg, device)

        run_tag = f"{os.path.splitext(os.path.basename(config_file))[0]}_seed{seed}"
        wandb_run_name = f"{cfg['MODEL']['architecture']}_{cfg['DATASETS']['train']['dataset']}_seed{seed}"

        best_metrics = train_and_evaluate(
            net, dataloaders, cfg, device, lin_cls=lin_cls,
            wandb_run_name=wandb_run_name, run_tag=run_tag,
        )
        best_metrics['_seed'] = seed
        runs.append(best_metrics)

        del net, dataloaders
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _print_multi_seed_table(config_file, seeds, runs)

    summary = {
        "config": config_file,
        "seeds": list(seeds),
        "runs": runs,
    }
    base_save = read_yaml(config_file)['CHECKPOINT']['save_path']
    summary_path = os.path.join(base_save, "multi_seed_summary.json")
    try:
        os.makedirs(base_save, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nWrote aggregate summary -> {summary_path}")
    except OSError as e:
        print(f"Could not write summary JSON ({summary_path}): {e}")

    return runs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', required=True, help='configuration (.yaml) file to use')
    parser.add_argument('--device', default='0,1', type=str,
                        help='gpu ids to use')
    parser.add_argument('--seeds', default='0', type=str,
                        help='comma-separated list of random seeds (e.g. "0,1,2"). '
                             'One full training run is executed per seed on this single '
                             'config, and mean/std of the best eval metrics is reported '
                             'at the end. Combine across folds manually if needed.')
    parser.add_argument('--lin', action='store_true',
                        help='train linear classifier only')
    parser.add_argument('--resume', default=None, type=str,
                        help='checkpoint (.pth) of a killed run to continue from (single seed only)')
    parser.add_argument('--start_epoch', default=None, type=int,
                        help='epoch to resume at; trains from here up to SOLVER.num_epochs')

    args = parser.parse_args()
    print(args.device)
    device_ids = [int(d) for d in args.device.split(',')]
    lin_cls = args.lin

    seeds = [int(s) for s in args.seeds.split(',') if s.strip() != '']

    if args.resume is not None:
        assert len(seeds) == 1 and args.start_epoch is not None, \
            "--resume needs exactly one seed and --start_epoch"
        seed = seeds[0]
        device = get_device(device_ids, allow_cpu=False)
        set_seed(seed)
        config = read_yaml(args.config)
        config['local_device_ids'] = device_ids
        config['seed'] = seed
        config['CHECKPOINT']['save_path'] = f"{config['CHECKPOINT']['save_path']}/seed{seed}"
        config['CHECKPOINT']['load_from_checkpoint'] = args.resume
        dataloaders = get_dataloaders(config)
        net = get_model(config, device)
        wandb_run_name = f"{config['MODEL']['architecture']}_{config['DATASETS']['train']['dataset']}_seed{seed}_resume{args.start_epoch}"
        best_metrics = train_and_evaluate(net, dataloaders, config, device,
                                          wandb_run_name=wandb_run_name, resume_epoch=args.start_epoch)
        print("\nBest eval metrics for seed %d:" % seed)
        for name, value in best_metrics.items():
            print(f"{name:<24}{value:.4f}")
    elif len(seeds) == 1:
        # Preserve historical single-run behaviour so existing run*.sh keep working.
        device = get_device(device_ids, allow_cpu=False)
        set_seed(seeds[0])
        config = read_yaml(args.config)
        config['local_device_ids'] = device_ids
        config['seed'] = seeds[0]
        dataloaders = get_dataloaders(config)
        net = get_model(config, device)
        train_and_evaluate(net, dataloaders, config, device)
    else:
        run_multi_seed(args.config, seeds, device_ids, lin_cls=lin_cls)
