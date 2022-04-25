import argparse
import os

import torch
import yaml
from ignite.contrib import metrics
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from skimage.segmentation import mark_boundaries
from skimage import morphology

import constants as const
import dataset
import fastflow
import utils


def build_train_data_loader(args, config):
    train_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=True,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )


def build_test_data_loader(args, config):
    test_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


def build_model(config):
    model = fastflow.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model


def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )


def train_one_epoch(dataloader, model, optimizer, epoch):
    model.train()
    loss_meter = utils.AverageMeter()
    for step, data in enumerate(dataloader):
        # forward
        data = data.cuda()
        ret = model(data)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )
            
def plot_anomaly(img, heat, segment, idx):
    
    fig, (ax0, ax2,ax3) = plt.subplots(ncols=3, figsize=(10, 5), facecolor='white')
    ax0.set_axis_off()
#     ax1.set_axis_off()
    ax2.set_axis_off()
    
    ax0.set_title('input image')
#     ax1.set_title('reconstructed image')
    ax2.set_title('heatmap ')
    ax3.set_title('anomalies')
    
    ax0.imshow(img, cmap=plt.cm.gray, interpolation='nearest') 
#     ax1.imshow(output, cmap=plt.cm.gray, interpolation='nearest')   
    ax2.imshow(heat, cmap=plt.cm.gray, interpolation='nearest')  
    ax3.imshow(segment, cmap=plt.cm.gray, interpolation='nearest')
    
#     x,y = np.where(H > threshold)
#     ax3.scatter(y,x,color='red',s=0.1) 

    plt.axis('off')
    
    fig.savefig(f'comp/{idx}.png')

    
def compute_mask(anomaly_map: np.ndarray, threshold: float, kernel_size: int = 4) -> np.ndarray:
    """Compute anomaly mask via thresholding the predicted anomaly map.

    Args:
        anomaly_map (np.ndarray): Anomaly map predicted via the model
        threshold (float): Value to threshold anomaly scores into 0-1 range.
        kernel_size (int): Value to apply morphological operations to the predicted mask. Defaults to 4.

    Returns:
        Predicted anomaly mask
    """

    anomaly_map = anomaly_map.squeeze()
    mask: np.ndarray = np.zeros_like(anomaly_map).astype(np.uint8)
    mask[anomaly_map > threshold] = 1

    kernel = morphology.disk(kernel_size)
    mask = morphology.opening(mask, kernel)

    mask *= 255

    return mask

def eval_once(dataloader, model):
   
    model.eval()
    auroc_metric = metrics.ROC_AUC()
    
    for data, targets in dataloader:
        data, targets = data.cuda(), targets.cuda()
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach() * 255
#         outputs = ret["anomaly_map"].cpu().detach()
        i = 0
        for output in outputs:
        
            output = outputs[i].numpy().squeeze().astype(np.uint8)
            
            pred_mask = compute_mask(output, 100)
            
            
#             output = outputs[5].numpy().squeeze()
#             output = (output - output.min()) / np.ptp(output)
#             output = output * 255
#             output = output.astype(np.uint8)

            colormap = cv2.applyColorMap(output, cv2.COLORMAP_JET)
    #         colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2GRAY)
            print("colormap", colormap.shape, colormap.dtype)


            original_img = data[i].cpu().detach().numpy().transpose(1, 2, 0) * 255
            original_img = original_img.astype(np.uint8)
    #         original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            print("original", original_img.shape, original_img.dtype)
        
            vis_img = mark_boundaries(original_img, pred_mask, color=(1, 0, 0), mode="thick")
            
            alpha = 0.3
            gamma = 0
            superimposed_map = cv2.addWeighted(colormap, alpha, original_img, (1 - alpha), gamma)

            heatmap_output = Image.fromarray(superimposed_map)
            heatmap_output.save(f'heatmap{i}.png')  
            
            plot_anomaly(original_img, superimposed_map, vis_img, i)



            dst_im = Image.fromarray(colormap)
    #         dst_im = Image.fromarray(outputs[5].numpy().squeeze().astype(np.uint8)).convert('RGB')

            print(type(dst_im))  # タイプ
            dst_im.save(f'output{i}.png')  # 画像を保存
            
            i += 1
            
        outputs = outputs.flatten()
        targets = targets.flatten()
        auroc_metric.update((outputs, targets))
#     auroc = auroc_metric.compute()
#     print("AUROC: {}".format(auroc))


def train(args):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    optimizer = build_optimizer(model)

    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()

    for epoch in range(const.NUM_EPOCHS):
        train_one_epoch(train_dataloader, model, optimizer, epoch)
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            eval_once(test_dataloader, model)
        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, "%d.pt" % epoch),
            )


def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    eval_once(test_dataloader, model)


def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument("--data", type=str, required=True, help="path to mvtec folder")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES,
        required=True,
        help="category name in mvtec",
    )
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)
