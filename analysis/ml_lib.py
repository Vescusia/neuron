import inspect
from pathlib import Path
from copy import deepcopy

import sklearn
import torch
import dill


def save_model(model: torch.nn.Module, model_kwargs: dict | None, state_dicts: list[dict], path):
    """
    save models to disk using dill.
    len(models) has to be >= 1.

    :param model: model to save.
    :param model_kwargs: keyword arguments with which the model got instantiated.
    :param state_dicts: list of state dictionaries to save.
    :param path: the path at which to save the model.
    """
    # get the source of the model class
    source = inspect.getsource(dill.source.getmodule(model))
    # and the name
    model_name = model.__class__.__name__

    # write them all
    with open(path, "wb") as f:
        dill.dump({
            "source": source,
            "model_name": model_name,
            "model_kwargs": model_kwargs if model_kwargs is not None else {},
            "state_dicts": state_dicts,
        }, f)


def load_model(path, state_dict_index: int = -1) -> torch.nn.Module:
    with open(path, "rb") as f:
        model_dict = dill.load(f)

    # load the source
    source = model_dict["source"]
    # execute the source code
    # DANGEROUS!!!
    input(f"[WARNING - Input Required] Loading Model from {path}. This may execute arbitrary code. Please only continue (<ENTER>) if you trust this File. (<CTRL-C> to abort)")
    namespace = {}
    exec(source, namespace)

    # instantiate the model class
    model_name = model_dict["model_name"]
    model = namespace[model_name]
    model: torch.nn.Module = model(**model_dict["model_kwargs"])

    # load desired state dict
    state_dicts = model_dict["state_dicts"]
    state_dict = state_dicts[state_dict_index]
    model.load_state_dict(state_dict)

    return model


class Embedder:
    """
    Embedder superclass. Subclasses only have to implement the `embed` method. Scaler has to be fitted using `.fit`.
    Calling convention is `__call__`.
    """
    def fit(self, games) -> None:
        games = self.embed(games)
        try:
            self._scaler.fit(games)
        except AttributeError:
            self._scaler = sklearn.preprocessing.StandardScaler()
            self._scaler.fit(games)

    @staticmethod
    def load(path: str | Path):
        # load Embedder dict from file
        with open(path, "rb") as f:
            embedder_dict = dill.load(f)

        # execute source code of saved Embedder
        # DANGEROUS!!
        namespace = {}
        input(f"[WARNING - Input Required] Loading Embedder from {path}. This may execute arbitrary code. Please only continue (<ENTER>) if you trust this File. (<CTRL-C> to abort)")
        exec(embedder_dict["source"], namespace)

        # instantiate the embedder from the namespace
        embedder_class = namespace[embedder_dict["name"]]
        embedder: Embedder = embedder_class(**embedder_dict["kwargs"])

        # set scaler state
        embedder._scaler = sklearn.preprocessing.StandardScaler()
        embedder._scaler.__setstate__(embedder_dict["state"])

        return embedder

    def save(self, path: str | Path, embedder_kwargs: dict | None) -> None:
        try:
            # get the state of the scaler
            state = self._scaler.__getstate__()

            # get the source code for the subclass
            source = dill.source.getsource(dill.source.getmodule(self))

            # save to file
            with open(path, "wb") as f:
                dill.dump(
                    {
                        "source": source,
                        "name": self.__class__.__name__,
                        "state": state,
                        "kwargs": embedder_kwargs if embedder_kwargs is not None else {},
                    }, f)

        except AttributeError:
            raise AttributeError("Scaler not fitted yet, please call self.fit first!")

    def _transform(self, embedded_games):
        """
        Transform embedded games.
        self.fit has to be called before this function!
        :return: The transformed games.
        """
        try:
            return self._scaler.transform(embedded_games)
        except AttributeError:
            raise AttributeError("Scaler not fitted yet, please call self.fit first!")

    def embed(self, games):
        """
        Returns a 2d array of embedded games.
        Do not use the scaler (self._transform) in this method.
        """
        raise NotImplementedError("Please implement this method in your subclass!")

    def __call__(self, games):
        return self._transform(self.embed(games))


class LROneCycleTillCyclic(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, one_cycle_epochs: int, steps_per_epoch: int, initial_lr: float, max_lr: float, min_lr: float):
        # do not initialize super. It will do an initial step, that is not wanted.

        self.one_cycle_epochs = one_cycle_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = 0
        self._last_lr = [initial_lr]

        self.one_cycle = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=one_cycle_epochs,
            div_factor=max_lr / initial_lr,
            final_div_factor=initial_lr / min_lr,
        )

        self.cyclic = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            max_lr=initial_lr,
            base_lr=min_lr,
            mode="triangular2",
        )

    def step(self, _=None):
        epoch = self.total_steps // self.steps_per_epoch
        self.total_steps += 1

        if epoch < self.one_cycle_epochs:
            self.one_cycle.step()
            self._last_lr = self.one_cycle.get_last_lr()
        else:
            self.cyclic.step()
            self._last_lr = self.cyclic.get_last_lr()

    def visualize(self):
        """
        Will try to visualize the learning rate development using matplotlib
        """
        scheduler = deepcopy(self)
        visualize_lr_scheduler(scheduler, self.one_cycle_epochs * 2, steps=True, steps_per_epoch=self.steps_per_epoch)


def visualize_lr_scheduler(scheduler: torch.optim.lr_scheduler.LRScheduler, epochs: int, steps: bool = False, steps_per_epoch: int | None = None) -> None:
    import matplotlib.pyplot as plt

    lrs = []

    for _ in range(epochs):
        if steps:
            for _ in range(steps_per_epoch):
                scheduler.step()
                lrs.append(scheduler.get_last_lr()[0])
        else:
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

    fig, ax = plt.subplots()
    ax.plot(lrs)
    ax.set_xlabel("Epoch" if not steps else "Step")
    ax.set_ylabel("LR")
    plt.show()
