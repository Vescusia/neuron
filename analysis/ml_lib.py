import inspect

import torch
import dill
from pandas.tseries.holiday import next_monday


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


def save_embedder(embedder: object, path):
    with open(path, "wb") as f:
        dill.dump({
            "source": dill.source.getsource(dill.source.getmodule(embedder)),
            "embedder": embedder,
        }, f)


def load_embedder(path) -> object:
    with open(path, "rb") as f:
        embedder_dict = dill.load(f)

    # load source code
    # DANGEROUS!!
    exec(embedder_dict["source"])

    # reload embedder now that the class is in the current frame
    with open(path, "rb") as f:
        embedder_dict = dill.load(f)
    return embedder_dict["embedder"]
