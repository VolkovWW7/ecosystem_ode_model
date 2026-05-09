from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QScrollArea, QGroupBox, QFormLayout, 
                             QDoubleSpinBox, QFrame, QFileDialog, QMessageBox,
                             QTabWidget) 
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import mathmodel

class BioOxWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BioOx Desktop - Анализ микробиоты")
        self.resize(1400, 900)
        self.params_widgets = {}

        # Основной виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)

        # --- ВЕРХНЯЯ ПАНЕЛЬ (Управление) ---
        self.top_panel = QHBoxLayout()
        self.btn_run = QPushButton("Запустить расчет")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold;")
        
        self.btn_import = QPushButton("Импорт JSON")
        self.btn_export = QPushButton("Экспорт JSON")
        self.btn_report = QPushButton("Создать отчет")

        self.top_panel.addWidget(self.btn_run)
        self.top_panel.addWidget(self.btn_import)
        self.top_panel.addWidget(self.btn_export)
        self.top_panel.addStretch()
        self.top_panel.addWidget(self.btn_report)
        self.main_layout.addLayout(self.top_panel)

        # --- ЦЕНТРАЛЬНАЯ ЧАСТЬ ---
        self.content_layout = QHBoxLayout()
        self.main_layout.addLayout(self.content_layout)

        # ЛЕВАЯ ПАНЕЛЬ (Параметры)
        self.scroll = QScrollArea()
        self.scroll.setFixedWidth(350)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        
        params_container = QWidget()
        self.params_layout = QVBoxLayout(params_container)
        
        # Группы параметров (без изменений)
        group_init = self._create_group("Начальные концентрации", [
            ('X0', "Автотрофы X", 0.0024),
            ('Y0', "Гетеротрофы Y", 0.00025),
            ('Dcl0', "Детрит хлореллы", 0.04),
            ('Dps0', "Детрит бактерий", 0.04),
            ('Ac0', "АТФ внутрикл.", 0.9),
            ('C0', "Минеральный C", 0.11),
            ('N0', "Минеральный N", 0.15),
        ])
        self.params_layout.addWidget(group_init)

        group_auto = self._create_group("Кинетика: Автотрофы", [
            ('Vx_c', "Max скорость роста (Vx_c)", 10.0),
            ('Kc', "Конст. полунас. C (Kc)", 0.3),
            ('Kn', "Конст. полунас. N (Kn)", 0.03),
            ('Ki', "Ингибирование (Ki)", 0.0),
            ('Mxx', "Коэф. смертности (Mxx)", 0.005),
            ('Ax', "Адаптация (Ax)", 1.0),
        ])
        self.params_layout.addWidget(group_auto)

        group_hetero = self._create_group("Кинетика: Гетеротрофы", [
            ('Vy_c', "Max скорость роста (Vy_c)", 25.0),
            ('Kpp', "Конст. полунас. P (Kpp)", 0.2),
            ('Kfc', "Конст. полунас. F/Ch", 0.1),
            ('Ka', "Конст. полунас. АТФ (Ka)", 0.2),
            ('Myy', "Коэф. смертности (Myy)", 0.03),
            ('Ay', "Адаптация (Ay)", 1.0),
        ])
        self.params_layout.addWidget(group_hetero)

        group_phys = self._create_group("Распад и Энергия", [
            ('Vm', "Скорость обмена (Vm)", 5.0),
            ('k_d1', "Распад детрита X (k_d1)", 0.0007),
            ('k_d2', "Распад детрита Y (k_d2)", 0.0007),
            ('k_d', "Трата АТФ (k_d)", 0.5),
            ('eps2', "Энерг. барьер (eps2)", 5.0),
        ])
        self.params_layout.addWidget(group_phys)
        
        group_composition = self._create_group("Состав биомассы", [
            ('bXP', "Белки в X (bXP)", 0.58),
            ('bXF', "Жиры в X (bXF)", 0.22),
            ('bYP', "Белки в Y (bYP)", 0.60),
            ('bYF', "Жиры в Y (bYF)", 0.10),
        ])
        self.params_layout.addWidget(group_composition)

        self.group_time = QGroupBox("Контроль времени")
        self.form_time = QFormLayout()
        self.params_widgets['total_time'] = self._create_spin(10000, 0, 1000000, 100)
        self.params_widgets['output_step'] = self._create_spin(10, 0.1, 1000, 1)
        self.form_time.addRow("Общее время:", self.params_widgets['total_time'])
        self.form_time.addRow("Шаг вывода:", self.params_widgets['output_step'])
        self.group_time.setLayout(self.form_time)
        self.params_layout.addWidget(self.group_time)

        self.params_layout.addStretch()
        self.scroll.setWidget(params_container)
        self.content_layout.addWidget(self.scroll)

        # ПРАВАЯ ПАНЕЛЬ (Графики)
        self.tabs = QTabWidget()
        self.tab_trofs = QWidget(); self.layout_trofs = QVBoxLayout(self.tab_trofs)
        self.tab_detrit = QWidget(); self.layout_detrit = QVBoxLayout(self.tab_detrit)
        self.tab_pfc = QWidget(); self.layout_pfc = QVBoxLayout(self.tab_pfc)
        self.tab_atp = QWidget(); self.layout_atp = QVBoxLayout(self.tab_atp)

        self.tabs.addTab(self.tab_trofs, "Популяции")
        self.tabs.addTab(self.tab_detrit, "Детрит")
        self.tabs.addTab(self.tab_pfc, "БЖУ")
        self.tabs.addTab(self.tab_atp, "АТФ")

        self.content_layout.addWidget(self.tabs, stretch=1)
        self.canvases = {"trofs": None, "detrit": None, "pfc": None, "atp": None}

    def _create_spin(self, val, min_v=0, max_v=100, step=0.01):
        sb = QDoubleSpinBox()
        sb.setRange(min_v, max_v)
        sb.setSingleStep(step)
        sb.setDecimals(4)
        sb.setValue(val)
        return sb

    def _create_group(self, title, params):
        group = QGroupBox(title)
        layout = QFormLayout()
        for key, label, default in params:
            spin = QDoubleSpinBox()
            spin.setRange(0, 1000000)
            spin.setDecimals(5)
            spin.setSingleStep(0.01)
            spin.setValue(default)
            layout.addRow(label, spin)
            self.params_widgets[key] = spin
        group.setLayout(layout)
        return group

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def display_figures(self, figs):
        layouts = {
            "trofs": self.layout_trofs,
            "detrit": self.layout_detrit,
            "pfc": self.layout_pfc,
            "atp": self.layout_atp
        }
        for key, fig in figs.items():
            self.clear_layout(layouts[key])
            canvas = FigureCanvas(fig)
            layouts[key].addWidget(canvas)
            self.canvases[key] = canvas

    def get_all_parameters(self):
        return {key: spin.value() for key, spin in self.params_widgets.items()}

    def set_parameters(self, params_dict):
        for key, spin in self.params_widgets.items():
            if key in params_dict:
                spin.setValue(params_dict[key])

    def export_params(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Сохранить параметры", "", "JSON (*.json)")
        if filepath:
            if not filepath.endswith('.json'):
                filepath += '.json'
            params = self.get_all_parameters()
            full_params = mathmodel.get_default_params()
            full_params.update(params)
            mathmodel.save_params_to_json(full_params, filepath)
            QMessageBox.information(self, "Успех", f"Параметры сохранены в {filepath}")

    def import_params(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Загрузить параметры", "", "JSON (*.json)")
        if filepath:
            new_params = mathmodel.load_params_from_json(filepath)
            gui_params = {k: new_params[k] for k in self.params_widgets.keys() if k in new_params}
            self.set_parameters(gui_params)
            QMessageBox.information(self, "Успех", f"Параметры загружены из {filepath}")
