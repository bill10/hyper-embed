import glob
import os
import random
from collections import Counter
from datetime import datetime
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange


class HypergraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, sep: str = ",", aggregate: bool = True):
        """Dataset for a single hypergraph

        Args:
            data_dir (str): Path to the directory of data files. Each line in 
                the file corresponds to a hyperedge. Fields in each line are 
                separated by `sep`. The first field is edge ID, and remaining
                fields are node IDs in the hyperedge. Different lines can have 
                different number of fields. Node IDs have to be integers between
                0 and number_of_nodes-1. Edge IDs can be anything.
            sep: Separator/delimiter of fields in a data file. Defaults to ','.
            aggregate (bool, optional): If True, duplicated hyperedges will be
                merged into one hyperedge, which is weighted by its number of 
                occurances. Defaults to True.
        """
        edge_id = []
        hyperedges = []
        num_nodes = -1
        for filename in glob.glob(os.path.join(data_dir, "*")):
            with open(filename) as infile:
                for line in infile:
                    items = line.split(sep)
                    edge_id.append(items[0])
                    edge = tuple(sorted(int(i.strip('\x00\n')) for i in items[1:]))
                    hyperedges.append(edge)
                    num_nodes = max(num_nodes, max(edge))
        self.num_nodes = num_nodes + 1
        if aggregate:
            self.hyperedges = list(Counter(hyperedges).items())
        else:
            self.hyperedges = list(zip(hyperedges, edge_id))
        self.num_edges = len(self.hyperedges)

    def __len__(self):
        return len(self.hyperedges)

    def __getitem__(self, idx):
        return self.hyperedges[idx]

    @staticmethod
    def collate_fn_pad(batch, padding_value: int, label_tensor: bool = True):
        """
        Pad batch of variable length
        """
        ## pad
        batch = [(torch.tensor(t[0]), t[1], len(t[0])) for t in batch]
        data, label, lengths = zip(*batch)
        data = nn.utils.rnn.pad_sequence(
            data, batch_first=True, padding_value=padding_value
        )
        if label_tensor:
            label = torch.tensor(label)
        return data, label, torch.tensor(lengths)

    def get_collate_fn(self, padding: int = None, label_tensor: bool = True):
        padding_value = padding if padding else self.num_nodes
        return lambda batch: self.collate_fn_pad(batch, padding_value, label_tensor)           


class HyperEmbed(nn.Module):
    def __init__(
        self, num_nodes: int, embedding_dim: int = None, embedding: torch.Tensor = None
    ):
        """HyperEmbedding model for a single hypergraph.

        Args:
            num_nodes (int): Number of nodes.
            embedding_dim (int, optional): Dimension of the embedding vector. 
                Either `embedding_dim` or `embedding` needs to be provided.
                Defaults to None.
            embedding (torch.Tensor, optional): Embedding matrix of all nodes to start
                with. It will be updated during training. If given, `embedding_dim` 
                will be ignored. Defaults to None.
        """
        super().__init__()
        if embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding, freeze=False, padding_idx=num_nodes
            )
        elif embedding_dim:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(
                    np.random.dirichlet([0.5] * embedding_dim, size=num_nodes + 1)
                ),
                freeze=False,
                padding_idx=num_nodes,
            )
        else:
            raise ValueError(
                "Please provide either embedding dimension or the embedding matrix"
            )
        with torch.no_grad():
            self.embedding.weight[num_nodes] = torch.ones(
                self.embedding.weight.shape[1]
            )

    def forward(self, combinations: torch.Tensor):
        """Forward pass.

        Args:
            combinations (torch.Tensor): A 2D array of combinations. Each row is a
                a list of node IDs, which corresponds to a combination. 
                The rows might be padded to form the matrix. 

        Returns:
            torch.Tensor: Propensities of the input combinations.
        """
        out = self.embedding(combinations)
        out = out.prod(dim=1).sum(dim=-1)
        return out

    def get_novelty(self, combinations: torch.Tensor, comb_sizes: torch.Tensor):
        """Calculate novelty of the given combinations.

        Args:
            combinations (torch.Tensor): A 2D array of combinations. Each row is a
                a list of node IDs, which corresponds to a combination. 
                The rows might be padded to form the matrix. 
            comb_sizes (torch.Tensor): A 1D array of the size of each given
                combination.

        Returns:
            torch.Tensor: Novelty scores of the input combinations.
        """
        out = self.embedding(combinations)
        propensities = out.prod(dim=1).sum(dim=-1)
        popularities = out.sum(dim=-1).cumprod(dim=-1)
        popularities = torch.gather(popularities, 1, comb_sizes.unsqueeze(1)-1).squeeze()
        return propensities / popularities


class DynamicHypergraphDataset:
    def __init__(self, data_dir: str, sep: str = ",",  aggregate: bool = True):
        """Dataset for a (temporal) sequence of hypergraphs.

        Args:
            data_dir (str): Path to the directory of data. Each sub directory under it
                will be read in as one hypergraph (HypergraphDataset) and the 
                sub directory name is used as the timestamp for the snapshot. The
                timestamps (sub directory names) have to be integers but don't
                have to be consecutive.
            sep: Separator/delimiter of fields in a data file. Defaults to ','.
            aggregate (bool, optional): If True, duplicated hyperedges will be
                merged into one hyperedge, which is weighted by its number of 
                occurances. Defaults to True.
        """
        self.hypergraphs = {}
        self.time_keys = []
        self.num_nodes = 0
        pbar = tqdm(sorted(glob.glob(os.path.join(data_dir, "*"))))
        for folder in pbar:
            foldername = os.path.basename(folder)
            self.time_keys.append(int(foldername))
            self.hypergraphs[int(foldername)] = HypergraphDataset(folder, sep, aggregate)
            self.num_nodes = max(
                self.num_nodes, self.hypergraphs[int(foldername)].num_nodes
            )
            pbar.set_description("Loaded {}. Overall".format(foldername))
        self.time_keys = sorted(self.time_keys)


class DynamicHyperEmbed:
    def __init__(
        self, num_nodes: int, embedding_dim: int, time_keys: list, time_variance: float, device: str="cpu"
    ):
        """HyperEmbed model for a (temporal) sequence of hypergraphs.

        Args:
            num_nodes (int): Number of nodes.
            embedding_dim (int): Dimension of the embedding vector for each node.
            time_keys (list): A list of time keys (integer timestamps) of the hypergraph snapshots.
            time_variance (float): Variance of embeddings between time snapshots.
            device (str, optional): Device used for computing. Defaults to cpu.
        """
        super().__init__()
        self.models = {}
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.time_keys = time_keys
        self.time_variance = time_variance
        self.optimizers = {}
        self.device = device

    def get_propensity(self, t: int, combinations: torch.tensor):
        """Calculate propensity of the given combinations with embeddings from time t.

        Args:
            combinations (torch.Tensor): A 2D array of combinations. Each row is a
                a list of node IDs, which corresponds to a combination. 
                The rows might be padded to form the matrix. 

        Returns:
            torch.Tensor: Propensity scores of the input combinations.
        """
        return self.models[t](combinations)

    def get_novelty(self, t: int, combinations: torch.Tensor, comb_sizes: torch.Tensor):
        """Calculate novelty of the given combinations with embeddings from time t.

        Args:
            combinations (torch.Tensor): A 2D array of combinations. Each row is a
                a list of node IDs, which corresponds to a combination. 
                The rows might be padded to form the matrix. 
            comb_sizes (torch.Tensor): A 1D array of the size of each given
                combination.

        Returns:
            torch.Tensor: Novelty scores of the input combinations.
        """
        return self.models[t].get_novelty(combinations, comb_sizes)

    def train_one_graph(
        self,
        model: HyperEmbed,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        hypergraph: HypergraphDataset,
        prev_weight: torch.Tensor = None,
        global_steps: int = 0,
        log_interval: int = 10,
        log_callback: Callable = None,
    ):
        model.train()
        collate_fn = hypergraph.get_collate_fn(self.num_nodes)
        for bid, batch in enumerate(
            tqdm(dataloader, desc="Batch", position=2, leave=False)
        ):
            # Loss on hyperedges
            inputs = batch[0].to(self.device)
            preds = model(inputs)
            labels = batch[1].to(self.device)
            pos_loss = loss_fn(preds, labels)
            # Loss on random combinations
            negative_samples = [
                (tuple(sorted(random.sample(range(hypergraph.num_nodes), len(i)))), 0)
                for i in batch[0]
            ]
            neg_batch = collate_fn(negative_samples)
            neg_inputs = neg_batch[0].to(self.device)
            preds = model(neg_inputs)
            labels = neg_batch[1].to(self.device)
            neg_loss = loss_fn(preds, labels)
            # Total loss
            loss = pos_loss + neg_loss
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Regulate by previous time point
            with torch.no_grad():
                if prev_weight is not None:
                    model.embedding.weight -= (
                        model.embedding.weight - prev_weight
                    ) / self.time_variance
                torch.clamp_(model.embedding.weight, 0.001)
            global_steps += 1
            # Logging
            if bid % log_interval == 0 and log_callback is not None:
                log_callback(loss=loss.item(), steps=global_steps)
        return global_steps, loss.item()

    def test(self, model, dataloader, hypergraph):
        successes = 0
        trials = 0
        collate_fn = hypergraph.get_collate_fn(self.num_nodes)
        with torch.no_grad():
            model.eval()
            for bid, batch in enumerate(
                tqdm(dataloader, desc="Evaluate", position=2, leave=False)
            ):
                inputs = batch[0].to(self.device)
                pos_preds = model(inputs)
                negative_samples = [
                    (
                        tuple(
                            sorted(random.sample(range(hypergraph.num_nodes), len(i)))
                        ),
                        0,
                    )
                    for i in batch[0]
                ]
                neg_batch = collate_fn(negative_samples)
                neg_inputs = neg_batch[0].to(self.device)
                neg_preds = model(neg_inputs)
                successes += (pos_preds > neg_preds).sum().item()
                trials += len(pos_preds)
        return {"AUC": successes / trials}

    def train(
        self,
        dataset: DynamicHypergraphDataset,
        num_epochs: int = 10,
        batch_size: int = 2048,
        shuffle: bool = True,
        lr: float = 0.001,
        start_epoch: int = 0,
        loss_fn: Callable = None,
        log_dir: str = None,
        log_interval: int = 10,
        checkpoint_dir: str = None,
        global_steps: dict = {}
    ):
        # Set up logging
        if not log_dir:
            log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_logger = SummaryWriter(log_dir)

        def log_loss(loss, steps, epoch, time, tb_logger):
            tb_logger.add_scalars(
                "Loss/{}".format(time), {"Epoch_{}".format(epoch): loss}, steps
            )
        
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Set up loss function
        if loss_fn is None:
            loss_fn = nn.PoissonNLLLoss(log_input=False)
        # Start training
        for epoch in tqdm(range(start_epoch, num_epochs), desc="Epoch", position=0):
            for t in trange(len(self.time_keys), desc="Time", position=1, leave=False):
                time = self.time_keys[t]
                prev_time = self.time_keys[t - 1] if t > 0 else None
                hypergraph = dataset.hypergraphs[time]
                train_dataloader = DataLoader(
                    hypergraph,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    collate_fn=hypergraph.get_collate_fn(self.num_nodes),
                )
                # Create the models sequentially in first epoch
                if epoch == 0:
                    if t == 0:
                        self.models[time] = HyperEmbed(
                            self.num_nodes, self.embedding_dim
                        )
                    else:
                        self.models[time] = HyperEmbed(
                            self.num_nodes,
                            embedding=self.models[
                                prev_time
                            ].embedding.weight.data.detach().clone(),
                        )
                    self.models[time].to(self.device)
                    global_steps[time] = 0
                    # Create optimizer as well
                    self.optimizers[time] = torch.optim.SGD(
                        self.models[time].parameters(), lr=lr
                    )
                steps, loss = self.train_one_graph(
                    model=self.models[time],
                    dataloader=train_dataloader,
                    optimizer=self.optimizers[time],
                    loss_fn=loss_fn,
                    hypergraph=hypergraph,
                    prev_weight=self.models[prev_time].embedding.weight if prev_time else None,
                    global_steps=global_steps[time],
                    log_interval=log_interval,
                    log_callback=lambda loss, steps: log_loss(
                        loss, steps, epoch, time, tb_logger
                    ),
                )
                global_steps[time] = steps
                log_loss(loss, global_steps[time], epoch, time, tb_logger)
                # Testing
                if t != len(self.time_keys) - 1:
                    next_time = self.time_keys[t + 1]
                    test_graph = dataset.hypergraphs[next_time]
                    test_dataloader = DataLoader(
                        test_graph,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=test_graph.get_collate_fn(self.num_nodes),
                    )
                    metrics = self.test(self.models[time], test_dataloader, test_graph)
                    tb_logger.add_scalar("AUC/{}".format(time), metrics["AUC"], epoch)
                tb_logger.flush()
            if checkpoint_dir:
                os.makedirs(os.path.join(checkpoint_dir, "epoch_{}".format(epoch)))
                torch.save(
                    {
                        "num_epochs": num_epochs,
                        "batch_size": batch_size,
                        "shuffle": shuffle,
                        "lr": lr,
                        "start_epoch": epoch+1,
                        "loss_fn": loss_fn,
                        "log_dir": log_dir,
                        "log_interval": log_interval,
                        "global_steps": global_steps,
                        "checkpoint_dir": checkpoint_dir
                    },
                    os.path.join(checkpoint_dir, "epoch_{}".format(epoch), "train_env.pkl")
                )
                self.save(os.path.join(checkpoint_dir, "epoch_{}".format(epoch)))
    
    def resume_train_from_checkpoint(self, checkpoint_dir: str, dataset: DynamicHypergraphDataset):
        train_env = torch.load(os.path.join(checkpoint_dir, "train_env.pkl"))
        self.load(checkpoint_dir)
        for time in self.models:
            self.optimizers[time] = torch.optim.SGD(
                self.models[time].parameters(), lr=train_env["lr"]
            )
        self.train(
            dataset=dataset,
            num_epochs=train_env["num_epochs"],
            batch_size=train_env["batch_size"],
            shuffle=train_env["shuffle"],
            lr=train_env["lr"],
            start_epoch=train_env["start_epoch"],
            loss_fn=train_env["loss_fn"],
            log_dir=train_env["log_dir"],
            log_interval=train_env["log_interval"],
            checkpoint_dir=train_env["checkpoint_dir"],
            global_steps=train_env["global_steps"]
        )

    def save(self, file_path):
        os.makedirs(file_path, exist_ok=True)
        pbar = tqdm(self.models)
        for key in pbar:
            torch.save(
                self.models[key].state_dict(),
                os.path.join(file_path, "model_{}.pt".format(key)),
            )
            pbar.set_description("Saved {}. Overall".format(key))
        torch.save(
            {
                "num_nodes": self.num_nodes,
                "embedding_dim": self.embedding_dim,
                "time_keys": self.time_keys,
                "time_variance": self.time_variance
            },
            os.path.join(file_path, "model_config.pkl")
        )

    def load(self, file_path):
        pbar = tqdm(sorted(glob.glob(os.path.join(file_path, "model_*.pt"))))
        for filename in pbar:
            time_key = int(os.path.basename(filename)[6:-3])
            self.models[time_key] = HyperEmbed(self.num_nodes, self.embedding_dim)
            self.models[time_key].load_state_dict(torch.load(filename, map_location=self.device))
            self.models[time_key].to(self.device)
            pbar.set_description("Loaded {}. Overall".format(time_key))
