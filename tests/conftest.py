"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np
import random
from pathlib import Path


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 2


@pytest.fixture
def seq_length():
    """Default sequence length for tests."""
    return 10


@pytest.fixture
def d_model():
    """Default model dimension for tests."""
    return 64


@pytest.fixture
def n_heads():
    """Default number of attention heads for tests."""
    return 4


@pytest.fixture
def vocab_size():
    """Default vocabulary size for tests."""
    return 100


@pytest.fixture
def sample_tokens(batch_size, seq_length, vocab_size):
    """Generate sample token IDs."""
    return torch.randint(0, vocab_size, (batch_size, seq_length))


@pytest.fixture
def sample_embeddings(batch_size, seq_length, d_model):
    """Generate sample embeddings."""
    return torch.randn(batch_size, seq_length, d_model)


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Create a temporary directory for model checkpoints."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


# Custom markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )