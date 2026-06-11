"""src/utils/logging.py — 統一的日誌設定。"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "stickerforge") -> logging.Logger:
    """取得設定好的 logger,確保格式一致且只初始化一次。"""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger("stickerforge")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(
        f"stickerforge.{name}" if name != "stickerforge" else "stickerforge"
    )
