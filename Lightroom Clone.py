import os
import sys

from PySide6.QtWidgets import QApplication

from lightroom_clone.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    if len(sys.argv) > 1:
        arg_path = sys.argv[1]
        if arg_path.lower().endswith(".lrc") and os.path.isfile(arg_path):
            try:
                window.import_project_from_path(arg_path)
            except Exception as exc:
                print(f"Failed to auto-load project: {exc}")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
