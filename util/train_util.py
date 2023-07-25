# CACIOUS CODING
# Data     : 7/24/23  9:45 PM
# File name: train_util
# Desc     :
import torch
import time
from util.weight_util import save_model


def train(model, dataloader, loss_func, optimizer, args):
    if args.device.type == "cuda":
        print("Running on GPU.")
    model = model.to(args.device)
    model.train()
    print("Start training...")
    for epoch in range(args.epochs):
        correct = 0
        for data_batch, label_batch in dataloader:
            data_batch = data_batch.to(args.device)
            label_batch = label_batch.to(args.device)
            output = model(data_batch)
            loss = loss_func(output, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(output, 1)
            correct += torch.sum(preds == label_batch)
        # scheduler.step()
        print("epoch: " + str(epoch) + "---loss: " + str(loss.item()) +
              "---acc: " + str((correct / len(dataloader.dataset)).item()))

        if epoch % 10 == 0 and epoch > 0:
            # res_md_h_epoch.pth
            now_time = time.strftime('%m%d_%H', time.localtime(time.time()))
            save_model(f"./weight/res_{now_time}_{epoch}.pth",
                       epoch, model, optimizer)
            print(f"Weight saved successfully.epoch:{epoch}")
        if args.epochs - epoch < 5:
            save_model(f"./weight/res_{now_time}_{epoch}.pth",
                       epoch, model, optimizer)
            print(f"Weight saved successfully.epoch:{epoch}")
