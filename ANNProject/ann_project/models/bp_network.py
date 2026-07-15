"""Compatibility shim for the BP network module."""

from src.models.bp_network import ANN, save_model, train_model

__all__ = ["ANN", "save_model", "train_model"]
