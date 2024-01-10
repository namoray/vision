
import pytest

from core import resource_management


@pytest.fixture()
def test_config_kandinsky():
    config = resource_management.ResourceConfig(KANDINSKY_DEVICE="cuda:0", SAFETY_CHECKERS_DEVICE="cuda:0")
    resource_singleton = resource_management.SingletonResourceManager()
    resource_singleton.load_config(config)
    resource_singleton.load_all_resources()
    yield
    resource_singleton.move_all_models_to_cpu()


@pytest.fixture()
def test_config_sdxl_turbo():
    config = resource_management.ResourceConfig(SDXL_TURBO_DEVICE="cuda:0", SAFETY_CHECKERS_DEVICE="cuda:0")
    resource_singleton = resource_management.SingletonResourceManager()
    resource_singleton.load_config(config)
    resource_singleton.load_all_resources()
    yield
    resource_singleton.move_all_models_to_cpu()

@pytest.fixture()
def test_config_realesrgan():
    config = resource_management.ResourceConfig(UPSCALE_DEVICE="cuda:0", SAFETY_CHECKERS_DEVICE="cuda:0")
    resource_singleton = resource_management.SingletonResourceManager()
    resource_singleton.load_config(config)
    resource_singleton.load_all_resources()
    yield

@pytest.fixture()
def test_config_scribble():
    config = resource_management.ResourceConfig(SCRIBBLE_DEVICE="cuda:0", SAFETY_CHECKERS_DEVICE="cuda:0")
    resource_singleton = resource_management.SingletonResourceManager()
    resource_singleton.load_config(config)
    resource_singleton.load_all_resources()
    yield
    resource_singleton.move_all_models_to_cpu()
