'''
Utility Functions for UI
'''
# Standard Import
import logging

# PySide 6
from PySide6.QtWidgets import (
    QLabel, QWidget, QHBoxLayout, QSizePolicy, QKeySequenceEdit, QComboBox,
    QGroupBox, QCheckBox, QFormLayout, QLineEdit, QLabel, QWidget, QHBoxLayout
)
from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QKeySequence, QKeyEvent, QDoubleValidator, QIntValidator
from PySide6.QtGui import QPixmap

# Local import
from src.utils.logger import logger
from src.ui.zh_cn import section_label, field_label, enum_label, field_tooltip

def validate_numerical_input(input_str: str, error_label: QLabel,
                              val_lowest: float, val_highest: float):
    try:
        val = float(input_str)
        if not (val_lowest <= val <= val_highest):
            raise ValueError
        error_label.setVisible(False)
        return True
    except ValueError:
        error_label.setText(f"请输入 {val_lowest} 到 {val_highest} 之间的数字。")
        error_label.setVisible(True)
        logger.debug(f"Invalid numerical input: {input_str}")
        return False

def create_error_label():
    error_label = QLabel()
    error_label.setStyleSheet("color: red;")
    error_label.setVisible(False)
    return error_label

def create_field(label_text, field_widget):
    container = QWidget()
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)  # Slight spacing

    label = QLabel(label_text)
    layout.addWidget(label)
    layout.addWidget(field_widget)

    # Avoid expanding
    label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
    field_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

    layout.setAlignment(Qt.AlignLeft)
    container.setLayout(layout)
    container.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)

    return container

def clear_debug_canvas(canvas):
    '''
    Flush the debug window viz canvas
    '''
    if hasattr(canvas, "show_message"):
        canvas.show_message("点击“开始”或按 F1 启动自动助手")
    else:
        canvas.clear()
        canvas.setText("点击“开始”或按 F1 启动自动助手")
        canvas.setAlignment(Qt.AlignCenter)


class ResponsiveImageLabel(QLabel):
    """A black image canvas that rescales its source whenever the tab resizes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 180)
        self.setStyleSheet("background-color: #101010; color: #dddddd;")

    def set_source_pixmap(self, pixmap: QPixmap):
        self._source_pixmap = QPixmap(pixmap)
        self._render_source()

    def show_message(self, message: str):
        self._source_pixmap = None
        QLabel.clear(self)
        self.setText(message)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_source()

    def _render_source(self):
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        target = self.contentsRect().size()
        if target.width() <= 0 or target.height() <= 0:
            return
        scaled = self._source_pixmap.scaled(
            target, Qt.KeepAspectRatio, Qt.FastTransformation
        )
        QLabel.setPixmap(self, scaled)

class SingleKeyEdit(QKeySequenceEdit):
    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        modifiers = event.modifiers()

        # Allow modifier-only keys
        if key in (Qt.Key_Shift, Qt.Key_Control, Qt.Key_Alt, Qt.Key_Meta):
            self.setKeySequence(QKeySequence(key))
            return

        # Otherwise, record only one key (replace previous)
        if modifiers:
            # This prevents combos like Ctrl+A
            self.setKeySequence(QKeySequence(key))
        else:
            self.setKeySequence(QKeySequence(key))

    def set_key(self, key_str):
        """
        Set the key sequence from string like 'A', 'F1', etc.
        """
        self.setKeySequence(QKeySequence(key_str))

    def get_key(self):
        """
        Return the currently set key as a string, like 'A', 'F1', etc.
        """
        seq = self.keySequence()
        return seq.toString(QKeySequence.NativeText).strip().lower()

class QtLogHandler(logging.Handler, QObject):
    '''
    QtLogHandler
    '''
    log_signal = Signal(str, int)  # message, level

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg, record.levelno)

def create_advance_setting_gbox(
    title, cfg, comments=None, comments_section=None, hidden_keys=()
):
    gbox = QGroupBox(section_label(title))
    form_layout = QFormLayout()
    gbox._field_refs = {}

    gbox.setToolTip(f"高级设置：{section_label(title)}（{title}）")

    def get_comment(key):
        if comments and title in comments and key in comments[title]:
            return comments[title][key]
        return ""

    def add_field(key, value):
        original_tooltip = get_comment(key)
        tooltip = field_tooltip(title, key)

        if isinstance(value, bool):
            checkbox = QCheckBox()
            checkbox.setChecked(value)
            checkbox.setToolTip(tooltip)
            def update_checkbox(state):
                cfg[title][key] = Qt.CheckState(state) == Qt.Checked
            checkbox.stateChanged.connect(update_checkbox)
            form_layout.addRow(QLabel(field_label(key)), checkbox)
            gbox._field_refs[key] = checkbox

        elif isinstance(value, list) or isinstance(value, tuple):
            hbox = QHBoxLayout()
            edits = []
            for v in value:
                line = QLineEdit(str(v))
                validator = QDoubleValidator() if isinstance(v, float) else QIntValidator()
                line.setValidator(validator)
                line.setToolTip(tooltip)
                edits.append(line)
                hbox.addWidget(line)

            def update_list():
                cfg[title][key] = [
                    float(e.text()) if isinstance(v, float) else int(e.text())
                    for e, v in zip(edits, value) if e.text() != ''
                ]

            for edit in edits:
                edit.textChanged.connect(update_list)

            form_layout.addRow(QLabel(field_label(key)), hbox)
            gbox._field_refs[key] = edits

        elif isinstance(value, (int, float)):
            line = QLineEdit(str(value))
            validator = QDoubleValidator() if isinstance(value, float) else QIntValidator()
            line.setValidator(validator)
            line.setToolTip(tooltip)
            def update_value(val):
                if val != '':
                    cfg[title][key] = float(val) if isinstance(value, float) else int(val)
            line.textChanged.connect(update_value)
            form_layout.addRow(QLabel(field_label(key)), line)
            gbox._field_refs[key] = line

        elif isinstance(value, str):
            if original_tooltip and "Options:" in original_tooltip:
                # Parse options from tooltip
                options_part = original_tooltip.split("Options:")[1]
                options = [opt.strip('" ') for opt in options_part.strip().split('"') if opt.strip()]

                # Create drop list
                combo = QComboBox()
                for option in options:
                    combo.addItem(enum_label(option), option)
                if value in options:
                    combo.setCurrentIndex(combo.findData(value))
                combo.setToolTip(tooltip)

                combo.currentIndexChanged.connect(
                    lambda _index, c=combo: cfg[title].__setitem__(key, c.currentData())
                )
                form_layout.addRow(QLabel(field_label(key)), combo)
                gbox._field_refs[key] = combo
            else:
                line = QLineEdit(value)
                line.setToolTip(tooltip)
                line.textChanged.connect(lambda val: cfg[title].__setitem__(key, val))
                form_layout.addRow(QLabel(field_label(key)), line)
                gbox._field_refs[key] = line

    hidden_keys = frozenset(hidden_keys)
    for key, value in cfg[title].items():
        if key in hidden_keys:
            continue
        add_field(key, value)

    gbox.setLayout(form_layout)
    return gbox
