"""
VR Stereo Image Viewer - Графический лаунчер
Перетащите изображения на окно для просмотра в VR
"""

import os
import sys
import subprocess
import threading
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Optional

# Пытаемся импортировать tkinterdnd2 для drag & drop
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False


class VRLauncher:
    """Графический лаунчер для VR Stereo Viewer"""
    
    # Цветовая схема
    COLORS = {
        'bg': '#1a1a2e',
        'bg_light': '#16213e',
        'accent': '#0f3460',
        'highlight': '#e94560',
        'text': '#eaeaea',
        'text_dim': '#8a8a9a',
        'success': '#4ecca3',
        'border': '#0f3460',
    }
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    CONFIG_FILE = "vr_viewer_config.json"
    
    def __init__(self):
        # Загружаем конфигурацию
        self.config = self.load_config()
        # Создаём главное окно с поддержкой DnD если доступно
        if HAS_DND:
            self.root = TkinterDnD.Tk()
        else:
            self.root = tk.Tk()
        
        self.root.title("VR Stereo Image Viewer")
        self.root.geometry("600x500")
        self.root.minsize(500, 400)
        self.root.configure(bg=self.COLORS['bg'])
        
        # Центрируем окно
        self.center_window()
        
        # Иконка (если есть)
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass
        
        # Выбранный путь (файл или папка)
        self.selected_path: Optional[str] = None
        self.vr_process: Optional[subprocess.Popen] = None
        
        # Создаём интерфейс
        self.create_ui()
        
        # Привязываем drag & drop
        if HAS_DND:
            self.setup_dnd()
        
        # Восстанавливаем последний путь
        self.restore_last_path()
        
    def center_window(self):
        """Центрирование окна на экране"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')
        
    def create_ui(self):
        """Создание пользовательского интерфейса"""
        # Основной контейнер
        main_frame = tk.Frame(self.root, bg=self.COLORS['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Заголовок
        title_label = tk.Label(
            main_frame,
            text="🥽 VR Stereo Viewer",
            font=('Segoe UI', 24, 'bold'),
            fg=self.COLORS['text'],
            bg=self.COLORS['bg']
        )
        title_label.pack(pady=(0, 5))
        
        subtitle_label = tk.Label(
            main_frame,
            text="Просмотр стереоизображений в виртуальной реальности",
            font=('Segoe UI', 10),
            fg=self.COLORS['text_dim'],
            bg=self.COLORS['bg']
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Зона для перетаскивания
        self.drop_frame = tk.Frame(
            main_frame,
            bg=self.COLORS['bg_light'],
            highlightbackground=self.COLORS['border'],
            highlightthickness=2,
            cursor='hand2'
        )
        self.drop_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Внутренний контейнер для зоны drop
        drop_inner = tk.Frame(self.drop_frame, bg=self.COLORS['bg_light'])
        drop_inner.place(relx=0.5, rely=0.5, anchor='center')
        
        # Иконка
        self.drop_icon = tk.Label(
            drop_inner,
            text="📁",
            font=('Segoe UI Emoji', 48),
            bg=self.COLORS['bg_light']
        )
        self.drop_icon.pack()
        
        # Текст в зоне drop
        if HAS_DND:
            drop_text = "Перетащите изображения сюда"
        else:
            drop_text = "Нажмите для выбора файлов"
        
        self.drop_label = tk.Label(
            drop_inner,
            text=drop_text,
            font=('Segoe UI', 14),
            fg=self.COLORS['text'],
            bg=self.COLORS['bg_light']
        )
        self.drop_label.pack(pady=10)
        
        self.drop_hint = tk.Label(
            drop_inner,
            text="Поддерживаются: JPG, PNG, BMP, TIFF",
            font=('Segoe UI', 9),
            fg=self.COLORS['text_dim'],
            bg=self.COLORS['bg_light']
        )
        self.drop_hint.pack()
        
        # Счётчик файлов
        self.files_count_label = tk.Label(
            drop_inner,
            text="",
            font=('Segoe UI', 11, 'bold'),
            fg=self.COLORS['success'],
            bg=self.COLORS['bg_light']
        )
        self.files_count_label.pack(pady=(15, 0))
        
        # Путь к выбранному файлу/папке
        self.path_label = tk.Label(
            drop_inner,
            text="",
            font=('Segoe UI', 9),
            fg=self.COLORS['text_dim'],
            bg=self.COLORS['bg_light'],
            wraplength=400
        )
        self.path_label.pack(pady=(5, 0))
        
        # Привязываем клик к зоне drop
        for widget in [self.drop_frame, drop_inner, self.drop_icon, self.drop_label, self.drop_hint]:
            widget.bind('<Button-1>', lambda e: self.open_files())
        
        # Кнопки
        buttons_frame = tk.Frame(main_frame, bg=self.COLORS['bg'])
        buttons_frame.pack(fill=tk.X, pady=15)
        
        # Стиль кнопок
        button_style = {
            'font': ('Segoe UI', 11),
            'cursor': 'hand2',
            'relief': 'flat',
            'padx': 20,
            'pady': 10,
        }
        
        # Кнопка выбора файлов
        self.files_btn = tk.Button(
            buttons_frame,
            text="📄 Выбрать файлы",
            bg=self.COLORS['accent'],
            fg=self.COLORS['text'],
            activebackground=self.COLORS['border'],
            activeforeground=self.COLORS['text'],
            command=self.open_files,
            **button_style
        )
        self.files_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        
        # Кнопка выбора папки
        self.folder_btn = tk.Button(
            buttons_frame,
            text="📁 Выбрать папку",
            bg=self.COLORS['accent'],
            fg=self.COLORS['text'],
            activebackground=self.COLORS['border'],
            activeforeground=self.COLORS['text'],
            command=self.open_folder,
            **button_style
        )
        self.folder_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))
        
        # Кнопка запуска VR
        self.start_btn = tk.Button(
            main_frame,
            text="🚀 Запустить в VR",
            font=('Segoe UI', 14, 'bold'),
            bg=self.COLORS['highlight'],
            fg='white',
            activebackground='#c73e54',
            activeforeground='white',
            cursor='hand2',
            relief='flat',
            padx=30,
            pady=15,
            command=self.start_vr,
            state='disabled'
        )
        self.start_btn.pack(fill=tk.X, pady=(5, 0))
        
        # Статус
        self.status_label = tk.Label(
            main_frame,
            text="",
            font=('Segoe UI', 9),
            fg=self.COLORS['text_dim'],
            bg=self.COLORS['bg']
        )
        self.status_label.pack(pady=(10, 0))
        
        # Эффекты при наведении на drop zone
        self.drop_frame.bind('<Enter>', self.on_drop_enter)
        self.drop_frame.bind('<Leave>', self.on_drop_leave)
        
    def setup_dnd(self):
        """Настройка drag & drop"""
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self.on_drop)
        self.drop_frame.dnd_bind('<<DragEnter>>', self.on_drag_enter)
        self.drop_frame.dnd_bind('<<DragLeave>>', self.on_drag_leave)
        
    def on_drop_enter(self, event):
        """При наведении мыши на зону drop"""
        self.drop_frame.configure(highlightbackground=self.COLORS['highlight'])
        
    def on_drop_leave(self, event):
        """При уходе мыши из зоны drop"""
        self.drop_frame.configure(highlightbackground=self.COLORS['border'])
        
    def on_drag_enter(self, event):
        """При начале перетаскивания над зоной"""
        self.drop_frame.configure(highlightbackground=self.COLORS['success'])
        self.drop_label.configure(text="Отпустите для загрузки")
        return event.action
        
    def on_drag_leave(self, event):
        """При уходе перетаскивания из зоны"""
        self.drop_frame.configure(highlightbackground=self.COLORS['border'])
        self.drop_label.configure(text="Перетащите изображения сюда")
        return event.action
        
    def on_drop(self, event):
        """Обработка drop"""
        self.drop_frame.configure(highlightbackground=self.COLORS['border'])
        self.drop_label.configure(text="Перетащите изображения сюда")
        
        # Парсим пути (могут быть в фигурных скобках)
        data = event.data
        paths = []
        
        # Обработка путей с пробелами в фигурных скобках
        if '{' in data:
            import re
            paths = re.findall(r'\{([^}]+)\}', data)
            # Добавляем пути без скобок
            remaining = re.sub(r'\{[^}]+\}', '', data).strip()
            if remaining:
                paths.extend(remaining.split())
        else:
            paths = data.split()
        
        # Берём первый валидный путь
        for path in paths:
            path = path.strip()
            if path and os.path.exists(path):
                p = Path(path)
                if p.is_dir() or (p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS):
                    self.set_selected_path(path)
                    break
        
        return event.action
            
    def open_files(self):
        """Открытие диалога выбора файлов"""
        filetypes = [
            ("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("Все файлы", "*.*"),
        ]
        
        # Используем последний путь
        initial_dir = self.config.get("last_path", "")
        if initial_dir and not os.path.isdir(initial_dir):
            initial_dir = os.path.dirname(initial_dir)
        
        files = filedialog.askopenfilenames(
            title="Выберите стереоизображения",
            filetypes=filetypes,
            initialdir=initial_dir if initial_dir else None
        )
        
        if files:
            # Берём первый файл - остальные из папки загрузятся автоматически
            self.set_selected_path(files[0])
            
    def open_folder(self):
        """Открытие диалога выбора папки"""
        # Используем последний путь
        initial_dir = self.config.get("last_path", "")
        if initial_dir and not os.path.isdir(initial_dir):
            initial_dir = os.path.dirname(initial_dir)
        
        folder = filedialog.askdirectory(
            title="Выберите папку с изображениями",
            initialdir=initial_dir if initial_dir else None
        )
        
        if folder:
            self.set_selected_path(folder)
            
    def start_vr(self):
        """Запуск VR просмотрщика"""
        if not self.selected_path:
            messagebox.showwarning("Внимание", "Сначала выберите изображения!")
            return
        
        self.status_label.configure(
            text="Запуск VR...",
            fg=self.COLORS['text_dim']
        )
        self.root.update()
        
        # Запускаем в отдельном потоке
        def run_viewer():
            try:
                script_dir = Path(__file__).parent
                viewer_script = script_dir / "sbs_viewer.py"
                
                # Передаём только путь - сканирование происходит при запуске VR
                cmd = [sys.executable, str(viewer_script), self.selected_path]
                
                # НЕ перехватываем stdout/stderr - иначе буфер заполнится и процесс зависнет!
                # Логи пишутся в файл vr_viewer.log
                self.vr_process = subprocess.Popen(
                    cmd,
                    cwd=str(script_dir),
                    stdout=None,  # Вывод в консоль (или никуда для .pyw)
                    stderr=None,
                    creationflags=subprocess.CREATE_NEW_CONSOLE  # Создаём новую консоль для вывода
                )
                
                self.root.after(0, lambda: self.status_label.configure(
                    text="VR Viewer запущен (см. vr_viewer.log)",
                    fg=self.COLORS['success']
                ))
                
                # Ждём завершения
                self.vr_process.wait()
                
                exit_code = self.vr_process.returncode
                if exit_code == 0:
                    self.root.after(0, lambda: self.status_label.configure(
                        text="VR Viewer завершён",
                        fg=self.COLORS['text_dim']
                    ))
                else:
                    self.root.after(0, lambda: self.status_label.configure(
                        text=f"VR Viewer завершён с ошибкой (код {exit_code})",
                        fg=self.COLORS['highlight']
                    ))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Ошибка",
                    f"Не удалось запустить VR Viewer:\n{e}"
                ))
                self.root.after(0, lambda: self.status_label.configure(
                    text=f"Ошибка: {e}",
                    fg=self.COLORS['highlight']
                ))
        
        thread = threading.Thread(target=run_viewer, daemon=True)
        thread.start()
    
    def load_config(self) -> dict:
        """Загрузка конфигурации"""
        config_path = Path(__file__).parent / self.CONFIG_FILE
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {"last_path": ""}
    
    def save_config(self):
        """Сохранение конфигурации"""
        config_path = Path(__file__).parent / self.CONFIG_FILE
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def restore_last_path(self):
        """Восстановление последнего использованного пути"""
        last_path = self.config.get("last_path", "")
        if last_path and os.path.exists(last_path):
            self.set_selected_path(last_path)
    
    def set_selected_path(self, path: str):
        """Установка выбранного пути"""
        if not os.path.exists(path):
            return
            
        self.selected_path = path
        
        # Определяем тип (файл или папка)
        if os.path.isdir(path):
            # Считаем изображения в папке
            count = sum(1 for f in os.listdir(path) 
                       if os.path.isfile(os.path.join(path, f)) 
                       and os.path.splitext(f)[1].lower() in self.SUPPORTED_EXTENSIONS)
            self.files_count_label.configure(text=f"📁 Папка: {count} изображений")
            self.drop_icon.configure(text="📁")
            display_path = path
        else:
            # Один файл - покажем что загрузится вся папка
            folder = os.path.dirname(path)
            count = sum(1 for f in os.listdir(folder) 
                       if os.path.isfile(os.path.join(folder, f)) 
                       and os.path.splitext(f)[1].lower() in self.SUPPORTED_EXTENSIONS)
            self.files_count_label.configure(text=f"📄 Файл выбран ({count} в папке)")
            self.drop_icon.configure(text="🖼️")
            display_path = path
        
        # Показываем путь (сокращённо если длинный)
        if len(display_path) > 60:
            display_path = "..." + display_path[-57:]
        self.path_label.configure(text=display_path)
        
        # Активируем кнопку запуска
        self.start_btn.configure(state='normal')
        
        # Сохраняем в конфиг
        self.config["last_path"] = path
        self.save_config()
        
    def run(self):
        """Запуск приложения"""
        self.root.mainloop()


def main():
    # Проверяем наличие tkinterdnd2
    if not HAS_DND:
        print("Предупреждение: tkinterdnd2 не установлен")
        print("Drag & Drop будет недоступен")
        print("Установите: pip install tkinterdnd2")
        print()
    
    app = VRLauncher()
    app.run()


if __name__ == '__main__':
    main()

