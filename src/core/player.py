"""Base classes for QEC game-loop players and configurations."""

import importlib
import torch
import inspect
from typing import Optional, Any, Dict
from ..base.dem import DetectorErrorModel


class Config:
    """Base class for configuration containers.

    Provides an automatic ``__repr__`` that prints only instance attributes
    that differ from their ``__init__`` default values.  Subclasses only need
    to set ``self.*`` attributes in ``__init__``; no manual ``__repr__`` required.
    Unknown kwargs passed to subclasses are forwarded to super(); Config ignores
    any that reach it.
    """

    def __init__(self, **kwargs):
        """Accept and ignore unknown kwargs from subclasses."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Export config to a serializable dict for save. Includes __class__ and __module__ for from_dict."""
        sig = inspect.signature(self.__class__.__init__)
        param_names = set(sig.parameters) - {'self'}
        d: Dict[str, Any] = {
            "__class__": self.__class__.__name__,
            "__module__": self.__class__.__module__,
        }
        for k, v in vars(self).items():
            if k in param_names:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Restore config from dict. Uses __class__ and __module__ if present for subclass resolution."""
        d = dict(d)
        cls_name = d.pop("__class__", None)
        module_name = d.pop("__module__", None)
        if cls_name and module_name:
            mod = importlib.import_module(module_name)
            config_cls = getattr(mod, cls_name)
            return config_cls(**d)
        return cls(**d)

    def __repr__(self) -> str:
        defaults = {}
        for cls in inspect.getmro(self.__class__):
            if cls is object:
                continue
            sig = inspect.signature(cls.__init__)
            for name, p in sig.parameters.items():
                if name == "self":
                    continue
                if name not in defaults and p.default is not inspect.Parameter.empty:
                    defaults[name] = p.default
        items = []
        for k, v in vars(self).items():
            if k in defaults and v == defaults[k]:
                continue
            items.append(f"    {k}={v!r}")
        if not items:
            return f"{self.__class__.__name__}()"
        body = ",\n".join(items)
        return f"{self.__class__.__name__}(\n{body}\n)"

    def absorb(self, **kwargs):
        """Return a new config with the given attributes updated.

        Only attributes that are ``__init__`` parameters are copied; internal
        state is omitted. Unknown keys are forwarded to the constructor and
        ignored by Config base.
        """
        sig = inspect.signature(self.__class__.__init__)
        param_names = set(sig.parameters) - {'self'}
        merged = {k: v for k, v in vars(self).items() if k in param_names}
        merged.update(kwargs)
        return self.__class__(**merged)


class Player:
    """Base class for players in the QEC game loop.

    A Player operates on a :class:`DetectorErrorModel` and exposes the
    unified ``update_corrections(syndromes)`` interface so that any
    subclass (Simulator, Decoder, …) can act as a player.

    Subclasses must implement ``update_corrections()``.
    """

    def __init__(self, dem: DetectorErrorModel, config: Optional[Config] = Config(), **kwargs):
        """Initialize Player.

        Args:
            dem: DetectorErrorModel instance the player operates on.
            config: Optional configuration object for the player.
        """
        if dem.validate():
            self.dem = dem
        else:
            raise ValueError("DetectorErrorModel is not valid")
        self.batch_shape = dem.batch_shape
        self.device = dem.device
        self.config = config.absorb(**kwargs)
        self.corrections = None

    def __repr__(self) -> str:
        dem_repr = repr(self.dem)
        if self.config is None:
            return f"{self.__class__.__name__}(\n    {dem_repr}\n)"
        config_repr = repr(self.config)
        config_lines = config_repr.split("\n")
        config_body = "    config=" + config_lines[0]
        if len(config_lines) > 1:
            config_body += "\n" + "\n".join(
                "    " + line for line in config_lines[1:]
            )
        if self.device != torch.device('cpu'):
            config_body += f",\n    device='{str(self.device)}'"
        return f"{self.__class__.__name__}(\n    {dem_repr},\n{config_body}\n)"

    def update_corrections(self, syndromes: torch.Tensor, **kwargs) -> None:
        """Predict or construct corrections from syndromes.

        Stores the result in ``self.corrections``.  Unified interface for
        all Player subclasses in the QEC game loop.  Subclasses must override.

        Args:
            syndromes: ``[*B, num_detectors]`` binary tensor ``{0, 1}``.
            **kwargs: Subclass-specific options (e.g. Simulator accepts
                ``range`` for teacher masking radius).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement update_corrections()"
        )

    def draw_on(self, draw_kwargs: Dict, **style) -> None:
        """Hook for ``DetectorErrorModel.draw()`` visualization.

        Called by ``DetectorErrorModel.draw()`` to annotate the visualization
        with player-specific state.  Override in subclasses.

        Args:
            draw_kwargs: Mutable dict of ``nx.draw_networkx`` kwargs.
            **style: Style keyword arguments from ``DetectorErrorModel.draw()``.
        """
        pass
