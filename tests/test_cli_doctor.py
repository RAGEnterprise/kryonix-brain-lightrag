def test_cli_doctor():
    # Basic import check for doctor requirements
    try:
        from kryonix_brain_lightrag import cli
        assert hasattr(cli, "cmd_doctor")
    except ImportError:
        assert False
