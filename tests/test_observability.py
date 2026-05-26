import logging
import warnings

from flexivtrainer.observability import (
    describe_exception,
    install_dependency_log_bridge,
)
from flexivtrainer.observability import console as console_module


def reset_dependency_bridge() -> None:
    root_logger = logging.getLogger()
    if console_module.DEPENDENCY_LOG_HANDLER is not None:
        root_logger.removeHandler(console_module.DEPENDENCY_LOG_HANDLER)
    console_module.DEPENDENCY_LOG_HANDLER = None
    logging.captureWarnings(False)


def test_describe_exception_includes_type_name() -> None:
    assert (
        describe_exception(RuntimeError("driver offline"))
        == "RuntimeError: driver offline"
    )


def test_dependency_log_bridge_forwards_warning_records(capsys) -> None:
    reset_dependency_bridge()
    install_dependency_log_bridge()

    logger = logging.getLogger("dependency.test")
    logger.warning("camera driver warm-up warning")

    captured = capsys.readouterr()
    assert "camera driver warm-up warning" in captured.err
    assert "logger=dependency.test" in captured.err


def test_dependency_log_bridge_captures_python_warnings(capsys) -> None:
    reset_dependency_bridge()
    install_dependency_log_bridge()

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn("sdk calibration drift", RuntimeWarning)

    captured = capsys.readouterr()
    assert "sdk calibration drift" in captured.err
    assert "logger=py.warnings" in captured.err
