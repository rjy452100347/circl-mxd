from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication


def create_application(argv=None):
    """Create the open-source desktop application and its main objects."""
    from src.ui.AutoBotController import AutoBotController
    from src.ui.ui import MainWindow

    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    controller = AutoBotController()
    window = MainWindow(controller)
    controller.update_signal(window)
    app._developer_objects = (controller, window)
    return app, controller, window


def main():
    app, _controller, window = create_application()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
