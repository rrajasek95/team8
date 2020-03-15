import logging
import torch
import torch.nn as nn
import torch.utils.tensorboard as tensorboard
import os
import math

def checkpoint_model(model, args, suffix):
    file_name = "{}_{}.pth".format(args.model, suffix)
    model_path = os.path.join(args.checkpoint_dir, file_name)
    torch.save(model.state_dict(), model_path)

def train_hred_model(model, optimizer, loss_func, loaders, args):
    highest_validation_loss = 1000.
    checkpoint_model(model, args, "best")
    writer = tensorboard.SummaryWriter(comment="_{}".format(args.model))

    step_idx = 0
    for epoch_idx in range(args.n_epochs):
        logging.info("Epoch {}:".format(epoch_idx))

        train_loss = 0.

        model.train()
        for batch_idx, batch in enumerate(loaders.train):
            optimizer.zero_grad()
            y_pred = model(batch['x_data'], batch['y_target'])

            out_size = y_pred.shape[-1]
            reshaped_target = batch['y_target'][1:].flatten()

            loss = loss_func(y_pred[1:].view(-1, out_size), reshaped_target)
            loss.backward()
            
            minibatch_loss = loss.item()
            logging.info("minibatch_loss Loss: {:.3f}".format(minibatch_loss))
            writer.add_scalar('MinibatchLoss/train', train_loss, step_idx)
            step_idx += 1

            train_loss += (minibatch_loss - train_loss) / (batch_idx + 1)
            nn.utils.clip_grad_value_(model.parameters(), 1)
            optimizer.step()

        logging.info("Training Loss: {:.3f}".format(train_loss))

        writer.add_scalar('Loss/train', train_loss, epoch_idx)
        writer.add_scalar('Perplexity/train', math.exp(train_loss), epoch_idx)

        checkpoint_model(model, args, epoch_idx)

        validation_loss = 0.
        model.eval()
        for batch_idx, batch in enumerate(loaders.val):
            y_pred = model(batch['x_data'], batch['y_target'])
            out_size = y_pred.shape[-1]
            reshaped_target = batch['y_target'][1:].flatten()
            loss = loss_func(y_pred[1:].view(-1, out_size), reshaped_target)

            minibatch_loss = loss.item()
            validation_loss += (minibatch_loss - validation_loss) / (batch_idx + 1)
        logging.info("Validation Loss: {:.3f}".format(validation_loss))

        writer.add_scalar('Loss/val', validation_loss, epoch_idx)
        writer.add_scalar('Perplexity/val', math.exp(validation_loss), epoch_idx)

        if validation_loss < highest_validation_loss:
            logging.info("Saving best model reported on epoch {}".format(epoch_idx))
            checkpoint_model(model, args, "best")

    test_loss = 0.
    model.eval()
    for batch_idx, batch in enumerate(loaders.test):
        y_pred = model(batch['x_data'], batch['y_target'])
        out_size = y_pred.shape[-1]
        reshaped_target = batch['y_target'][1:].flatten()
        loss = loss_func(y_pred[1:].view(-1, out_size), reshaped_target)
        minibatch_loss = loss.item()
        test_loss += (minibatch_loss - test_loss) / (batch_idx + 1)
    logging.info("Test Loss: {:.3f}".format(test_loss))


def train_memnet_model(model, optimizer, loss_func, loaders, args):
    lowest_validation_loss = 1000.
    checkpoint_model(model, args, "best")
    writer = tensorboard.SummaryWriter(comment="_{}".format(args.model))

    step_idx = 0
    for epoch_idx in range(args.n_epochs):
        logging.info("Epoch {}:".format(epoch_idx))

        train_loss = 0.

        model.train()
        for batch_idx, batch in enumerate(loaders.train):
            optimizer.zero_grad()
            y_pred, loss = model(batch['x_data'], batch['x_facts'], batch['y_target'], loss_func, 0.)
            # logging.debug("y_pred: {}".format(y_pred[:, 1:].shape))
            # [batch_size, seq_len, n_classes]
            # reshaped_target = batch['y_target'][:, 1:]
            # logging.debug("y_target: {}".format(reshaped_target.shape))
            # [batch_size, seq_len]

            # loss = loss_func(y_pred[:, 1:], reshaped_target)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), args.max_norm)
            minibatch_loss = loss.item()
            logging.info("minibatch_loss Loss: {:.3f}".format(minibatch_loss))
            writer.add_scalar('MinibatchLoss/train', train_loss, step_idx)
            step_idx += 1

            train_loss += (minibatch_loss - train_loss) / (batch_idx + 1)

            optimizer.step()

        logging.info("Training Loss: {:.3f}".format(train_loss))
        writer.add_scalar('Loss/train', train_loss, epoch_idx)
        writer.add_scalar('Perplexity/train', math.exp(train_loss), epoch_idx)
        checkpoint_model(model, args, epoch_idx)

        validation_loss = 0.
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(loaders.dev):
                y_pred, loss = model(batch['x_data'], batch['x_facts'], batch['y_target'], loss_func,  0.)
                # out_size = y_pred.shape[-1]
                # reshaped_target = batch['y_target'][1:].flatten()
                # loss = loss_func(y_pred[1:].view(-1, out_size), reshaped_target)

                minibatch_loss = loss.item()
                validation_loss += (minibatch_loss - validation_loss) / (batch_idx + 1)
            logging.info("Validation Loss: {:.3f}".format(validation_loss))
            writer.add_scalar('Loss/val', validation_loss, epoch_idx)
            writer.add_scalar('Perplexity/val', math.exp(validation_loss), epoch_idx)
            if validation_loss < lowest_validation_loss:
                checkpoint_model(model, args, "best")

    test_loss = 0.
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loaders.test):
            y_pred, loss = model(batch['x_data'], batch['x_facts'], batch['y_target'], loss_func,  0.)
            # out_size = y_pred.shape[-1]
            # reshaped_target = batch['y_target'][1:].flatten()
            # loss = loss_func(y_pred[1:].view(-1, out_size), reshaped_target)
            minibatch_loss = loss.item()
            test_loss += (minibatch_loss - test_loss) / (batch_idx + 1)
        logging.info("Test Loss: {:.3f}".format(test_loss))
