import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from core.data_loader.BinanceDataLoader import DataLoader

@pytest.fixture
def fake_parquet(tmp_path) -> str:
    fake_data = pd.DataFrame({
        "time": pd.date_range("2025-02-01", periods=5, freq="1D"),
        "symbol": ["BTC/TEST"] * 5,
        "open": np.random.rand(5),
        "high": np.random.rand(5),
        "low": np.random.rand(5),
        "close": np.random.rand(5),
        "volume": np.random.rand(5)
    })
    file_path = tmp_path / "test_data.parquet"
    fake_data.to_parquet(file_path, compression="snappy")
    return str(file_path)

def test_load_data_from_local_parquet(fake_parquet, mocker):
    loader = DataLoader(data_path=fake_parquet, start_date="2025-02-01", end_date="2025-02-28")
    mock_fetch = mocker.patch.object(loader, "_fetch_and_build_dataset", return_value=pd.DataFrame())
    df = loader.load_data()
    mock_fetch.assert_not_called()
    assert len(df) == 5

def test_load_data_when_no_local_file(mocker, tmp_path):
    non_existent_file = str(tmp_path / "nonexistent.parquet")
    loader = DataLoader(data_path=non_existent_file, start_date="2025-02-01", end_date="2025-02-28")

    fake_data = pd.DataFrame({
        "time": pd.date_range("2025-02-01", periods=2, freq="1D"),
        "symbol": ["BTC/TEST", "BTC/TEST"],
        "open": [1.0, 2.0],
        "high": [1.5, 2.5],
        "low": [0.8, 1.8],
        "close": [1.2, 2.2],
        "volume": [100, 200],
    })
    mock_fetch = mocker.patch.object(loader, "_fetch_and_build_dataset", return_value=fake_data)
    df = loader.load_data()
    mock_fetch.assert_called_once()
    assert len(df) == 2
    assert os.path.exists(non_existent_file)

def test_data_validation(mocker, tmp_path):
    loader = DataLoader(data_path=str(tmp_path / "somefile.parquet"),
                        start_date="2025-02-01", end_date="2025-02-28")
    broken_data = pd.DataFrame({
        "time": pd.date_range("2025-02-01", periods=5, freq="1D"),
        "symbol": ["BTC/TEST"] * 5,
        "random_col": [1, 2, 3, 4, 5]
    })
    mocker.patch.object(loader, "_fetch_and_build_dataset", return_value=broken_data)
    with pytest.raises(ValueError) as excinfo:
        loader.load_data()
    assert "Missing columns" in str(excinfo.value)

def test_get_top_liquid_symbols(fake_parquet):
    loader = DataLoader(data_path=fake_parquet, start_date="2025-02-01", end_date="2025-02-28")
    df = loader.load_data()
    top_symbols = loader.get_top_liquid_symbols(limit=1)
    assert len(top_symbols) == 1
