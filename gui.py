from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QScrollArea, QGroupBox, QFormLayout, 
                             QDoubleSpinBox, QFrame, QLabel, QTabWidget)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

class BioOxWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BioOx Desktop - Анализ микробиоты")
        self.resize(1400, 900)

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

        # --- ЦЕНТРАЛЬНАЯ ЧАСТЬ (Параметры | Графики) ---
        self.content_layout = QHBoxLayout()
        self.main_layout.addLayout(self.content_layout)

        # ЛЕВАЯ ПАНЕЛЬ (Параметры)
        self.scroll = QScrollArea()
        self.scroll.setFixedWidth(350)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        
        params_container = QWidget()
        self.params_layout = QVBoxLayout(params_container)
        
        # 1. Группа: Начальные условия
        self.group_init = QGroupBox("Начальные условия")
        self.form_init = QFormLayout()
        self.sp_X0 = self._create_spin(0.0024); self.form_init.addRow("X0 (Автотрофы):", self.sp_X0)
        self.sp_Y0 = self._create_spin(0.00025); self.form_init.addRow("Y0 (Гетеротрофы):", self.sp_Y0)
        self.sp_C0 = self._create_spin(0.11); self.form_init.addRow("C0 (Углерод):", self.sp_C0)
        self.sp_N0 = self._create_spin(0.15); self.form_init.addRow("N0 (Азот):", self.sp_N0)
        self.group_init.setLayout(self.form_init)
        self.params_layout.addWidget(self.group_init)

        # 2. Группа: Параметры роста
        self.group_growth = QGroupBox("Параметры роста")
        self.form_growth = QFormLayout()
        self.sp_Vx_c = self._create_spin(10.0); self.form_growth.addRow("Vx_c (Авто):", self.sp_Vx_c)
        self.sp_Vy_c = self._create_spin(25.0); self.form_growth.addRow("Vy_c (Гетеро):", self.sp_Vy_c)
        self.sp_Vm = self._create_spin(5.0); self.form_growth.addRow("Vm (Потребление):", self.sp_Vm)
        self.group_growth.setLayout(self.form_growth)
        self.params_layout.addWidget(self.group_growth)

        # 3. Группа: Время
        self.group_time = QGroupBox("Контроль времени")
        self.form_time = QFormLayout()
        self.sp_total_time = self._create_spin(10000, 0, 1000000, 100); self.form_time.addRow("Общее время:", self.sp_total_time)
        self.sp_output_step = self._create_spin(10, 0.1, 1000, 1); self.form_time.addRow("Шаг вывода:", self.sp_output_step)
        self.group_time.setLayout(self.form_time)
        self.params_layout.addWidget(self.group_time)

        self.params_layout.addStretch()
        self.scroll.setWidget(params_container)
        self.content_layout.addWidget(self.scroll)

        # ПРАВАЯ ПАНЕЛЬ (Графики с вкладками)
        self.tabs = QTabWidget()
        
        # Создаем контейнеры для каждой вкладки
        self.tab_trofs = QWidget(); self.layout_trofs = QVBoxLayout(self.tab_trofs)
        self.tab_detrit = QWidget(); self.layout_detrit = QVBoxLayout(self.tab_detrit)
        self.tab_pfc = QWidget(); self.layout_pfc = QVBoxLayout(self.tab_pfc)
        self.tab_atp = QWidget(); self.layout_atp = QVBoxLayout(self.tab_atp)

        self.tabs.addTab(self.tab_trofs, "Популяции")
        self.tabs.addTab(self.tab_detrit, "Детрит")
        self.tabs.addTab(self.tab_pfc, "БЖУ")
        self.tabs.addTab(self.tab_atp, "АТФ")

        self.content_layout.addWidget(self.tabs, stretch=1)

        # Слоты для хранения холстов
        self.canvases = {"trofs": None, "detrit": None, "pfc": None, "atp": None}

    def _create_spin(self, val, min_v=0, max_v=100, step=0.01):
        sb = QDoubleSpinBox()
        sb.setRange(min_v, max_v)
        sb.setSingleStep(step)
        sb.setDecimals(4)
        sb.setValue(val)
        return sb

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def display_figures(self, figs):
        """
        Принимает словарь вида {'trofs': fig, 'detrit': fig, ...}
        """
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
