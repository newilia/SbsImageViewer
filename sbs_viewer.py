"""
VR Stereo Image Viewer using OpenXR
Просмотрщик стереоизображений для VR-шлемов

Поддерживает форматы:
- Side-by-Side (SBS) - левое и правое изображение рядом
- Отдельные файлы для левого и правого глаза
"""

import os
import sys
import ctypes
import argparse
import logging
import time
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple
import tkinter as tk
from tkinter import filedialog

try:
    from send2trash import send2trash
    HAS_SEND2TRASH = True
except ImportError:
    HAS_SEND2TRASH = False

# Файл конфигурации
CONFIG_FILE = Path(__file__).parent / "vr_viewer_settings.json"

import numpy as np
from PIL import Image


# ============== НАСТРОЙКА ЛОГИРОВАНИЯ ==============
class FlushingHandler(logging.StreamHandler):
    """Handler который сразу сбрасывает буфер"""
    def emit(self, record):
        super().emit(record)
        self.flush()

class FlushingFileHandler(logging.FileHandler):
    """FileHandler который сразу сбрасывает буфер"""
    def emit(self, record):
        super().emit(record)
        self.flush()

def setup_logging(log_file: str = "vr_viewer.log", console_level=logging.INFO):
    """Настройка системы логирования"""
    
    # Форматтер с временными метками
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Логгер
    logger = logging.getLogger('VRViewer')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # Консольный вывод (с немедленным flush)
    console_handler = FlushingHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Файловый вывод (с немедленным flush)
    file_handler = FlushingFileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Глобальный логгер
log = setup_logging()

# OpenXR imports
import xr
from xr import (
    Instance,
    Session,
    SystemId,
    Space,
    Swapchain,
    SwapchainCreateInfo,
    SwapchainUsageFlags,
    ViewConfigurationType,
    EnvironmentBlendMode,
    ReferenceSpaceType,
    SessionState,
    StructureType,
    Result,
    ActionSet,
    Action,
    ActionType,
    ActionCreateInfo,
    ActionSetCreateInfo,
    ActionStateGetInfo,
    ActionsSyncInfo,
    ActiveActionSet,
    InteractionProfileSuggestedBinding,
    ActionSuggestedBinding,
    SessionActionSetsAttachInfo,
)

# OpenGL imports
from OpenGL.GL import *
from OpenGL.GL import shaders
import platform
if platform.system() == "Windows":
    from OpenGL import WGL

import glfw

from linear import Matrix4x4f


class StereoImage:
    """Класс для хранения стереопары изображений"""
    
    def __init__(self, left: np.ndarray, right: np.ndarray, name: str = "", path: str = ""):
        self.left = left
        self.right = right
        self.name = name
        self.path = path  # Полный путь к файлу
        self.left_texture: Optional[int] = None
        self.right_texture: Optional[int] = None
        self.name_texture: Optional[int] = None
        self.name_aspect: float = 1.0
    
    @classmethod
    def from_sbs(cls, image_path: str) -> 'StereoImage':
        """Загрузка SBS (side-by-side) изображения"""
        img = Image.open(image_path)
        
        # Конвертируем только если нужно (RGB быстрее чем RGBA)
        if img.mode == 'RGBA':
            pass  # Уже RGBA
        elif img.mode == 'RGB':
            img = img.convert('RGBA')  # Добавляем альфа-канал
        else:
            img = img.convert('RGBA')
        
        width, height = img.size
        
        # Разделяем изображение пополам
        left_img = img.crop((0, 0, width // 2, height))
        right_img = img.crop((width // 2, 0, width, height))
        
        left = np.array(left_img, dtype=np.uint8)
        right = np.array(right_img, dtype=np.uint8)
        
        return cls(left, right, Path(image_path).name, os.path.abspath(image_path))
    
    @classmethod
    def from_separate_files(cls, left_path: str, right_path: str) -> 'StereoImage':
        """Загрузка из отдельных файлов для левого и правого глаза"""
        left_img = Image.open(left_path).convert('RGBA')
        right_img = Image.open(right_path).convert('RGBA')
        
        left = np.array(left_img, dtype=np.uint8)
        right = np.array(right_img, dtype=np.uint8)
        
        name = f"{Path(left_path).stem} / {Path(right_path).stem}"
        return cls(left, right, name, os.path.abspath(left_path))
    
    def create_textures(self):
        """Создание OpenGL текстур"""
        self.left_texture = self._create_texture(self.left)
        self.right_texture = self._create_texture(self.right)
        self._create_name_texture()
    
    def _create_name_texture(self):
        """Создание текстуры с именем файла"""
        from PIL import ImageDraw, ImageFont
        
        # Размеры текстуры
        text_height = 64
        
        # Создаём изображение для текста
        try:
            font = ImageFont.truetype("arial.ttf", 48)
        except:
            font = ImageFont.load_default()
        
        # Измеряем размер текста
        dummy_img = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), self.name, font=font)
        text_width = bbox[2] - bbox[0] + 20
        text_height = bbox[3] - bbox[1] + 10
        
        # Создаём изображение
        img = Image.new('RGBA', (text_width, text_height), (0, 0, 0, 180))
        draw = ImageDraw.Draw(img)
        draw.text((10, 0), self.name, fill=(255, 255, 255, 255), font=font)
        
        # Сохраняем соотношение сторон
        self.name_aspect = text_width / text_height
        
        # Создаём текстуру
        self.name_texture = self._create_texture(np.array(img, dtype=np.uint8))
    
    def _create_texture(self, image_data: np.ndarray) -> int:
        """Создание одной OpenGL текстуры"""
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        
        # Качественная фильтрация с mipmaps для сглаживания
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        height, width = image_data.shape[:2]
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA8,
            width, height, 0,
            GL_RGBA, GL_UNSIGNED_BYTE,
            image_data
        )
        glGenerateMipmap(GL_TEXTURE_2D)
        
        return texture
    
    def delete_textures(self):
        """Удаление OpenGL текстур"""
        if self.left_texture:
            glDeleteTextures(1, [self.left_texture])
            self.left_texture = None
        if self.right_texture:
            glDeleteTextures(1, [self.right_texture])
            self.right_texture = None
        if self.name_texture:
            glDeleteTextures(1, [self.name_texture])
            self.name_texture = None


class VRStereoViewer:
    """Основной класс просмотрщика VR стереоизображений"""
    
    # Простой вершинный шейдер
    VERTEX_SHADER = """
    #version 410
    in vec3 VertexPos;
    in vec2 VertexUV;
    
    uniform mat4 ModelViewProjection;
    
    out vec2 TexCoord;
    
    void main() {
        gl_Position = ModelViewProjection * vec4(VertexPos, 1.0);
        TexCoord = VertexUV;
    }
    """
    
    # Простой фрагментный шейдер (белый цвет или текстура)
    FRAGMENT_SHADER = """
    #version 410
    in vec2 TexCoord;
    out vec4 FragColor;
    
    uniform sampler2D uTexture;
    uniform int uUseTexture;
    
    void main() {
        if (uUseTexture == 1) {
            FragColor = texture(uTexture, TexCoord);
        } else {
            FragColor = vec4(1.0, 1.0, 1.0, 1.0);  // Белый цвет
        }
    }
    """
    
    def __init__(self, image_paths: List[str], sbs_mode: bool = True):
        self.image_paths = image_paths
        self.sbs_mode = sbs_mode
        self.current_index = 0
        self.images: List[StereoImage] = []
        
        # OpenXR объекты
        self.instance: Optional[Instance] = None
        self.system_id: Optional[SystemId] = None
        self.session: Optional[Session] = None
        self.local_space: Optional[Space] = None
        self.view_space: Optional[Space] = None
        self.swapchains: List[Swapchain] = []
        self.swapchain_images: List[List] = []
        self.framebuffers: List[List[int]] = []
        
        # Функция получения требований к графике
        self.pxrGetOpenGLGraphicsRequirementsKHR = None
        self.graphics_requirements = xr.GraphicsRequirementsOpenGLKHR()
        self.graphics_binding = xr.GraphicsBindingOpenGLWin32KHR()
        
        # OpenGL объекты
        self.shader_program: Optional[int] = None
        self.quad_vao: Optional[int] = None
        self.quad_vbo: Optional[int] = None
        self.window = None
        
        # Состояние
        self.session_running = False
        self.should_quit = False
        self.views = []
        self.view_configs = []
        self.render_target_size = None
        
        # Контроллеры (Meta Quest 3 / Oculus Touch)
        self.action_set: Optional[ActionSet] = None
        self.thumbstick_y_action: Optional[Action] = None  # Thumbstick Y - расстояние
        self.thumbstick_x_action: Optional[Action] = None  # Thumbstick X - масштаб
        self.next_action: Optional[Action] = None  # A/X кнопки - следующее фото
        self.prev_action: Optional[Action] = None  # B/Y кнопки - предыдущее фото
        self.menu_action: Optional[Action] = None  # Menu кнопка - выход
        self.trigger_action: Optional[Action] = None  # Триггеры
        self.grip_action: Optional[Action] = None  # Grip/Squeeze (бампер под средним пальцем)
        self.pose_action: Optional[Action] = None  # Поза контроллера
        self.hand_paths = []  # Пути к левой и правой руке
        self.hand_spaces = [None, None]  # Пространства для отслеживания позиции рук
        
        # Состояние контроллеров
        self.last_thumbstick_y = [0.0, 0.0]  # [left, right]
        self.thumbstick_deadzone = 0.2
        self.thumbstick_speed_distance = 1.5  # Скорость изменения расстояния (экспоненциальная)
        self.thumbstick_speed_scale = 0.5  # Скорость изменения масштаба
        self.button_cooldown = 0.3  # Задержка между нажатиями (секунды)
        self.last_next_press = 0.0
        self.last_prev_press = 0.0
        
        # Состояние для смещения изображения
        self.image_offset_x = 0.0  # Смещение по горизонтали (метры)
        self.image_offset_y = 0.0  # Смещение по вертикали (метры)
        self.controller_grab_rot = [None, None]  # Ориентация контроллера при захвате [left, right]
        self.translation_sensitivity = 0.05  # Чувствительность перемещения (метры/градус)
        self.predicted_display_time = 0  # Время для locate_space
        
        # Параметры отображения (загружаем из конфига)
        settings = self.load_settings()
        self.quad_distance = settings.get("distance", 10.0)
        self.quad_scale = settings.get("scale", 1.0)
        self.base_size = 1.0  # Базовый физический размер при расстоянии 1м
        self.distance_texture: Optional[int] = None
        self.distance_aspect: float = 1.0
        self.counter_texture: Optional[int] = None
        self.counter_aspect: float = 1.0
        self.head_height: Optional[float] = None  # Высота головы (центр между глазами)
        self.watch_folder: Optional[str] = None  # Папка для мониторинга
        self.last_folder_check: float = 0  # Время последней проверки папки
        self.folder_check_interval: float = 2.0  # Интервал проверки (секунды)
        self.cross_eyed_mode: bool = settings.get("cross_eyed", False)  # Режим просмотра: False = parallel, True = cross-eyed
        self.ipd_offset: float = settings.get("ipd_offset", 0.0)  # Смещение IPD (межзрачковое расстояние), в метрах
        self.ipd_step: float = 0.04  # Шаг изменения IPD (40 мм)
        
        mode_name = "Cross-eyed" if self.cross_eyed_mode else "Parallel"
        log.info(f"Загружены настройки: масштаб={self.quad_scale:.2f}, IPD={self.ipd_offset * 1000:+.1f}мм, режим={mode_name}")
        
    def load_images(self):
        """Подготовка списка изображений (ленивая загрузка)"""
        # Если передан один файл - загружаем все изображения из его папки
        if len(self.image_paths) == 1 and os.path.isfile(self.image_paths[0]):
            single_file = self.image_paths[0]
            folder = os.path.dirname(single_file)
            if folder:
                all_files = find_images(folder)
                if all_files:
                    self.image_paths = all_files
                    # Находим индекс исходного файла
                    try:
                        start_index = [os.path.normpath(p) for p in all_files].index(os.path.normpath(single_file))
                        self.current_index = start_index
                    except ValueError:
                        self.current_index = 0
        
        # Фильтруем пути (убираем _right файлы для режима separate)
        filtered_paths = []
        for path in self.image_paths:
            if not self.sbs_mode and '_right' in path.lower():
                continue
            filtered_paths.append(path)
        self.image_paths = filtered_paths
        
        log.info(f"Найдено {len(self.image_paths)} изображений")
        
        # Создаём placeholder-объекты (загрузка будет при показе)
        for path in self.image_paths:
            # Создаём пустой объект с путём
            img = StereoImage(np.array([]), np.array([]), Path(path).name, os.path.abspath(path))
            img._loaded = False  # Флаг загрузки
            self.images.append(img)
        
        # Загружаем только первое изображение сразу
        if self.images:
            self._load_image_data(self.current_index)
            log.info(f"Загружено: {self.images[self.current_index].name}")
            # Предзагружаем соседние в фоне
            self._preload_nearby()
        
        # Запоминаем папку для мониторинга
        if self.image_paths:
            first_path = self.image_paths[0]
            if os.path.isfile(first_path):
                self.watch_folder = os.path.dirname(first_path)
            else:
                self.watch_folder = first_path
    
    def _load_image_data(self, index: int):
        """Загрузка данных изображения по индексу"""
        if index < 0 or index >= len(self.images):
            return
        
        img = self.images[index]
        if hasattr(img, '_loaded') and img._loaded:
            return  # Уже загружено
        
        path = img.path
        try:
            if self.sbs_mode:
                loaded = StereoImage.from_sbs(path)
            else:
                if '_left' in path.lower():
                    right_path = path.replace('_left', '_right').replace('_Left', '_Right')
                    if os.path.exists(right_path):
                        loaded = StereoImage.from_separate_files(path, right_path)
                    else:
                        loaded = StereoImage.from_sbs(path)
                else:
                    loaded = StereoImage.from_sbs(path)
            
            # Копируем данные
            img.left = loaded.left
            img.right = loaded.right
            img._loaded = True
        except Exception as e:
            log.error(f"Ошибка загрузки {path}: {e}")
            # Создаём пустое изображение чтобы не падать
            img.left = np.zeros((100, 100, 4), dtype=np.uint8)
            img.right = np.zeros((100, 100, 4), dtype=np.uint8)
            img._loaded = True
    
    def _preload_nearby(self):
        """Предзагрузка соседних изображений в фоне"""
        if len(self.images) <= 1:
            return
        
        # Индексы для предзагрузки: следующее и предыдущее
        indices_to_preload = [
            (self.current_index + 1) % len(self.images),
            (self.current_index - 1) % len(self.images),
        ]
        
        for idx in indices_to_preload:
            if idx != self.current_index:
                img = self.images[idx]
                if not hasattr(img, '_loaded') or not img._loaded:
                    # Загружаем в фоновом потоке
                    thread = threading.Thread(
                        target=self._load_image_data,
                        args=(idx,),
                        daemon=True
                    )
                    thread.start()
    
    def check_for_new_files(self):
        """Проверка появления новых файлов в папке"""
        current_time = time.time()
        if current_time - self.last_folder_check < self.folder_check_interval:
            return
        
        self.last_folder_check = current_time
        
        if not self.watch_folder or not os.path.isdir(self.watch_folder):
            return
        
        # Получаем текущий список файлов
        current_files = set(find_images(self.watch_folder))
        known_files = set(os.path.normpath(img.path) for img in self.images if img.path)
        
        # Находим новые файлы
        new_files = current_files - known_files
        
        if new_files:
            log.info(f"🆕 Обнаружено {len(new_files)} новых файлов")
            
            # Добавляем новые файлы
            for path in sorted(new_files):
                norm_path = os.path.normpath(path)
                if not self.sbs_mode and '_right' in norm_path.lower():
                    continue
                
                img = StereoImage(np.array([]), np.array([]), Path(path).name, os.path.abspath(path))
                img._loaded = False
                self.images.append(img)
                self.image_paths.append(path)
            
            log.info(f"Всего изображений: {len(self.images)}")
            self.update_counter_texture()
    
    def initialize_openxr_instance(self):
        """Инициализация OpenXR Instance и получение требований к графике"""
        log.info("=" * 50)
        log.info("Инициализация OpenXR...")
        
        # Получаем доступные расширения
        log.debug("Получение списка расширений OpenXR...")
        discovered_extensions = xr.enumerate_instance_extension_properties()
        log.debug(f"Доступно расширений: {len(discovered_extensions)}")
        
        # Проверяем наличие OpenGL расширения
        requested_extensions = [xr.KHR_OPENGL_ENABLE_EXTENSION_NAME]
        for ext in requested_extensions:
            if ext not in discovered_extensions:
                log.error(f"  ✗ Расширение {ext} не найдено!")
                raise RuntimeError(f"Расширение {ext} не поддерживается")
        log.info("  ✓ Расширение OpenGL найдено")
        
        # Создаём экземпляр OpenXR
        log.debug("Создание OpenXR Instance...")
        app_info = xr.ApplicationInfo(
            application_name="SBS Stereo Viewer",
            application_version=0,
            engine_name="pyopenxr",
            engine_version=xr.PYOPENXR_CURRENT_API_VERSION,
            api_version=xr.Version(1, 0, xr.XR_VERSION_PATCH),
        )
        
        create_info = xr.InstanceCreateInfo(
            application_info=app_info,
            enabled_extension_names=requested_extensions,
        )
        
        self.instance = xr.create_instance(create_info)
        log.info(f"  ✓ OpenXR Instance создан")
        
        # ВАЖНО: Получаем функцию xrGetOpenGLGraphicsRequirementsKHR
        log.debug("Получение функции xrGetOpenGLGraphicsRequirementsKHR...")
        self.pxrGetOpenGLGraphicsRequirementsKHR = ctypes.cast(
            xr.get_instance_proc_addr(
                self.instance,
                "xrGetOpenGLGraphicsRequirementsKHR",
            ),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR
        )
        log.debug("  ✓ Функция получена")
        
        # Получаем систему (HMD)
        log.debug("Поиск VR шлема (HMD)...")
        get_info = xr.SystemGetInfo(xr.FormFactor.HEAD_MOUNTED_DISPLAY)
        
        try:
            self.system_id = xr.get_system(self.instance, get_info)
            log.info(f"  ✓ System ID: {self.system_id}")
        except xr.FormFactorUnavailableError:
            log.error("  ✗ VR шлем не найден! Убедитесь, что шлем подключён и включён.")
            raise RuntimeError("VR шлем не обнаружен")
        
        # Получаем конфигурации видов
        log.debug("Получение конфигураций видов...")
        view_config_views = xr.enumerate_view_configuration_views(
            self.instance, self.system_id, xr.ViewConfigurationType.PRIMARY_STEREO)
        
        if len(view_config_views) >= 2:
            self.render_target_size = (
                view_config_views[0].recommended_image_rect_width * 2,
                view_config_views[0].recommended_image_rect_height
            )
            log.info(f"  ✓ Размер рендера: {self.render_target_size[0]}x{self.render_target_size[1]}")
        
        # ОБЯЗАТЕЛЬНО: Вызываем xrGetOpenGLGraphicsRequirementsKHR
        log.debug("Получение требований к графике OpenGL...")
        result = self.pxrGetOpenGLGraphicsRequirementsKHR(
            self.instance, 
            self.system_id, 
            ctypes.byref(self.graphics_requirements)
        )
        result = xr.exception.check_result(xr.Result(result))
        if result.is_exception():
            log.error(f"  ✗ Ошибка получения требований: {result}")
            raise result
        log.info("  ✓ Требования к графике получены")
        
    def initialize_glfw(self):
        """Инициализация GLFW для OpenGL контекста"""
        log.info("=" * 50)
        log.info("Инициализация GLFW и OpenGL...")
        
        if not glfw.init():
            log.error("  ✗ Не удалось инициализировать GLFW")
            raise RuntimeError("Не удалось инициализировать GLFW")
        log.debug("  GLFW инициализирован")
        
        # Настройки окна как в примере pyopenxr
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)  # Видимое окно
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.FALSE)  # Без двойной буферизации!
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        # Размер окна
        window_width = 400
        window_height = 200
        
        log.debug("  Создание окна GLFW...")
        self.window = glfw.create_window(window_width, window_height, "VR Stereo Viewer - Перетащите файлы сюда", None, None)
        if not self.window:
            glfw.terminate()
            log.error("  ✗ Не удалось создать окно GLFW")
            raise RuntimeError("Не удалось создать окно GLFW")
        log.debug("  ✓ Окно GLFW создано")
        
        # Центрируем окно
        log.debug("  Центрирование окна...")
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        glfw.set_window_pos(self.window, (mode.size.width - window_width) // 2, (mode.size.height - window_height) // 2)
        log.debug("  ✓ Окно отцентрировано")
        
        log.debug("  Активация OpenGL контекста...")
        glfw.make_context_current(self.window)
        
        # Отключаем vsync чтобы не мешал OpenXR
        glfw.swap_interval(0)
        log.debug("  ✓ OpenGL контекст активирован")
        
        log.debug("  Получение информации OpenGL...")
        gl_version = glGetString(GL_VERSION).decode()
        gl_vendor = glGetString(GL_VENDOR).decode()
        gl_renderer = glGetString(GL_RENDERER).decode()
        
        log.info(f"  ✓ OpenGL версия: {gl_version}")
        log.info(f"  ✓ GPU: {gl_renderer}")
        log.debug(f"    Vendor: {gl_vendor}")
        
        # Обрабатываем события чтобы окно не зависало
        glfw.poll_events()
        
    def create_session(self):
        """Создание OpenXR сессии"""
        log.info("=" * 50)
        log.info("Создание OpenXR сессии...")
        glfw.poll_events()
        
        # Получаем DC и GLRC через WGL (как в примере pyopenxr)
        log.debug("Получение OpenGL контекста через WGL...")
        self.graphics_binding.h_dc = WGL.wglGetCurrentDC()
        self.graphics_binding.h_glrc = WGL.wglGetCurrentContext()
        
        log.debug(f"  HDC: {self.graphics_binding.h_dc}")
        log.debug(f"  HGLRC: {self.graphics_binding.h_glrc}")
        
        if not self.graphics_binding.h_glrc:
            log.error("  ✗ OpenGL контекст не найден!")
            raise RuntimeError("OpenGL контекст не создан")
        log.debug("  ✓ OpenGL контекст получен")
        
        # Создаём сессию
        log.debug("Создание сессии OpenXR...")
        pp = ctypes.cast(ctypes.pointer(self.graphics_binding), ctypes.c_void_p)
        session_create_info = xr.SessionCreateInfo(
            create_flags=xr.SessionCreateFlags.NONE,
            system_id=self.system_id,
            next=pp,
        )
        
        try:
            self.session = xr.create_session(self.instance, session_create_info)
            log.info("  ✓ Сессия OpenXR создана")
        except Exception as e:
            log.error(f"  ✗ Ошибка создания сессии: {e}")
            raise
        
        # Создаём референсные пространства
        log.debug("Создание референсных пространств...")
        
        # Пробуем STAGE, если не поддерживается - LOCAL
        try:
            stage_space_info = xr.ReferenceSpaceCreateInfo(
                reference_space_type=xr.ReferenceSpaceType.STAGE,
                pose_in_reference_space=xr.Posef(xr.Quaternionf(0, 0, 0, 1), xr.Vector3f(0, 0, 0)),
            )
            self.local_space = xr.create_reference_space(self.session, stage_space_info)
            log.debug("  ✓ STAGE space создан")
        except:
            local_space_info = xr.ReferenceSpaceCreateInfo(
                reference_space_type=xr.ReferenceSpaceType.LOCAL,
                pose_in_reference_space=xr.Posef(xr.Quaternionf(0, 0, 0, 1), xr.Vector3f(0, 0, 0)),
            )
            self.local_space = xr.create_reference_space(self.session, local_space_info)
            log.debug("  ✓ LOCAL space создан")
        
        view_space_info = xr.ReferenceSpaceCreateInfo(
            reference_space_type=xr.ReferenceSpaceType.VIEW,
            pose_in_reference_space=xr.Posef(xr.Quaternionf(0, 0, 0, 1), xr.Vector3f(0, 0, 0)),
        )
        self.view_space = xr.create_reference_space(self.session, view_space_info)
        log.debug("  ✓ VIEW space создан")
        
        # Инициализируем контроллеры
        self.initialize_controller_actions()
        
    def initialize_controller_actions(self):
        """Инициализация системы действий для контроллеров Meta Quest 3"""
        log.info("=" * 50)
        log.info("Инициализация контроллеров...")
        
        try:
            # Создаём набор действий
            action_set_info = xr.ActionSetCreateInfo(
                action_set_name="viewer_controls",
                localized_action_set_name="Viewer Controls",
                priority=0,
            )
            self.action_set = xr.create_action_set(self.instance, action_set_info)
            log.debug("  ✓ Action set создан")
            
            # Пути к рукам
            self.hand_paths = (xr.Path * 2)(
                xr.string_to_path(self.instance, "/user/hand/left"),
                xr.string_to_path(self.instance, "/user/hand/right"),
            )
            
            # Действие для thumbstick Y (вперёд-назад = расстояние)
            self.thumbstick_y_action = xr.create_action(
                action_set=self.action_set,
                create_info=xr.ActionCreateInfo(
                    action_type=xr.ActionType.FLOAT_INPUT,
                    action_name="thumbstick_y",
                    localized_action_name="Thumbstick Y (Distance)",
                    count_subaction_paths=len(self.hand_paths),
                    subaction_paths=self.hand_paths,
                ),
            )
            log.debug("  ✓ Thumbstick Y action создан")
            
            # Действие для thumbstick X (влево-вправо = масштаб)
            self.thumbstick_x_action = xr.create_action(
                action_set=self.action_set,
                create_info=xr.ActionCreateInfo(
                    action_type=xr.ActionType.FLOAT_INPUT,
                    action_name="thumbstick_x",
                    localized_action_name="Thumbstick X (Scale)",
                    count_subaction_paths=len(self.hand_paths),
                    subaction_paths=self.hand_paths,
                ),
            )
            log.debug("  ✓ Thumbstick X action создан")
            
            # Действие для кнопки "следующее фото" (A на правом, X на левом)
            self.next_action = xr.create_action(
                action_set=self.action_set,
                create_info=xr.ActionCreateInfo(
                    action_type=xr.ActionType.BOOLEAN_INPUT,
                    action_name="next_image",
                    localized_action_name="Next Image",
                    count_subaction_paths=0,
                    subaction_paths=None,
                ),
            )
            log.debug("  ✓ Next action создан")
            
            # Действие для кнопки "предыдущее фото" (B на правом, Y на левом)
            self.prev_action = xr.create_action(
                action_set=self.action_set,
                create_info=xr.ActionCreateInfo(
                    action_type=xr.ActionType.BOOLEAN_INPUT,
                    action_name="prev_image",
                    localized_action_name="Previous Image",
                    count_subaction_paths=0,
                    subaction_paths=None,
                ),
            )
            log.debug("  ✓ Prev action создан")
            
            # Действие для кнопки выхода (Menu)
            self.menu_action = xr.create_action(
                action_set=self.action_set,
                create_info=xr.ActionCreateInfo(
                    action_type=xr.ActionType.BOOLEAN_INPUT,
                    action_name="menu_exit",
                    localized_action_name="Menu/Exit",
                    count_subaction_paths=0,
                    subaction_paths=None,
                ),
            )
            log.debug("  ✓ Menu action создан")
            
            # Действие для триггера (для сброса смещения вместе с grip)
            self.trigger_action = xr.create_action(
                action_set=self.action_set,
                create_info=xr.ActionCreateInfo(
                    action_type=xr.ActionType.FLOAT_INPUT,
                    action_name="trigger",
                    localized_action_name="Trigger",
                    count_subaction_paths=len(self.hand_paths),
                    subaction_paths=self.hand_paths,
                ),
            )
            log.debug("  ✓ Trigger action создан")
            
            # Действие для grip/squeeze (бампер - для перемещения изображения)
            self.grip_action = xr.create_action(
                action_set=self.action_set,
                create_info=xr.ActionCreateInfo(
                    action_type=xr.ActionType.FLOAT_INPUT,
                    action_name="grip",
                    localized_action_name="Grip",
                    count_subaction_paths=len(self.hand_paths),
                    subaction_paths=self.hand_paths,
                ),
            )
            log.debug("  ✓ Grip action создан")
            
            # Действие для позы контроллера (отслеживание положения)
            self.pose_action = xr.create_action(
                action_set=self.action_set,
                create_info=xr.ActionCreateInfo(
                    action_type=xr.ActionType.POSE_INPUT,
                    action_name="hand_pose",
                    localized_action_name="Hand Pose",
                    count_subaction_paths=len(self.hand_paths),
                    subaction_paths=self.hand_paths,
                ),
            )
            log.debug("  ✓ Pose action создан")
            
            # === Привязки для Oculus Touch (Meta Quest 3) ===
            # Пути к элементам управления
            thumbstick_y_path = [
                xr.string_to_path(self.instance, "/user/hand/left/input/thumbstick/y"),
                xr.string_to_path(self.instance, "/user/hand/right/input/thumbstick/y"),
            ]
            thumbstick_x_path = [
                xr.string_to_path(self.instance, "/user/hand/left/input/thumbstick/x"),
                xr.string_to_path(self.instance, "/user/hand/right/input/thumbstick/x"),
            ]
            
            # A/X кнопки (нижние кнопки)
            a_click_path = xr.string_to_path(self.instance, "/user/hand/right/input/a/click")
            x_click_path = xr.string_to_path(self.instance, "/user/hand/left/input/x/click")
            
            # B/Y кнопки (верхние кнопки)
            b_click_path = xr.string_to_path(self.instance, "/user/hand/right/input/b/click")
            y_click_path = xr.string_to_path(self.instance, "/user/hand/left/input/y/click")
            
            # Menu кнопка (только на левом контроллере у Oculus)
            menu_click_path = xr.string_to_path(self.instance, "/user/hand/left/input/menu/click")
            
            # Триггеры
            trigger_path = [
                xr.string_to_path(self.instance, "/user/hand/left/input/trigger/value"),
                xr.string_to_path(self.instance, "/user/hand/right/input/trigger/value"),
            ]
            
            # Grip/Squeeze (бампер)
            grip_path = [
                xr.string_to_path(self.instance, "/user/hand/left/input/squeeze/value"),
                xr.string_to_path(self.instance, "/user/hand/right/input/squeeze/value"),
            ]
            
            # Поза контроллера
            pose_path = [
                xr.string_to_path(self.instance, "/user/hand/left/input/grip/pose"),
                xr.string_to_path(self.instance, "/user/hand/right/input/grip/pose"),
            ]
            
            # Создаём привязки для Oculus Touch
            oculus_bindings = [
                # Thumbstick Y (вперёд-назад = расстояние) - оба контроллера
                xr.ActionSuggestedBinding(self.thumbstick_y_action, thumbstick_y_path[0]),
                xr.ActionSuggestedBinding(self.thumbstick_y_action, thumbstick_y_path[1]),
                # Thumbstick X (влево-вправо = масштаб) - оба контроллера
                xr.ActionSuggestedBinding(self.thumbstick_x_action, thumbstick_x_path[0]),
                xr.ActionSuggestedBinding(self.thumbstick_x_action, thumbstick_x_path[1]),
                # Next: A и X
                xr.ActionSuggestedBinding(self.next_action, a_click_path),
                xr.ActionSuggestedBinding(self.next_action, x_click_path),
                # Prev: B и Y
                xr.ActionSuggestedBinding(self.prev_action, b_click_path),
                xr.ActionSuggestedBinding(self.prev_action, y_click_path),
                # Menu
                xr.ActionSuggestedBinding(self.menu_action, menu_click_path),
                # Triggers
                xr.ActionSuggestedBinding(self.trigger_action, trigger_path[0]),
                xr.ActionSuggestedBinding(self.trigger_action, trigger_path[1]),
                # Grip
                xr.ActionSuggestedBinding(self.grip_action, grip_path[0]),
                xr.ActionSuggestedBinding(self.grip_action, grip_path[1]),
                # Pose
                xr.ActionSuggestedBinding(self.pose_action, pose_path[0]),
                xr.ActionSuggestedBinding(self.pose_action, pose_path[1]),
            ]
            
            # Регистрируем для Oculus Touch контроллера
            xr.suggest_interaction_profile_bindings(
                instance=self.instance,
                suggested_bindings=xr.InteractionProfileSuggestedBinding(
                    interaction_profile=xr.string_to_path(
                        self.instance,
                        "/interaction_profiles/oculus/touch_controller",
                    ),
                    count_suggested_bindings=len(oculus_bindings),
                    suggested_bindings=(xr.ActionSuggestedBinding * len(oculus_bindings))(*oculus_bindings),
                ),
            )
            log.info("  ✓ Привязки Oculus Touch зарегистрированы")
            
            # === Привязки для KHR Simple Controller (fallback) ===
            select_path = [
                xr.string_to_path(self.instance, "/user/hand/left/input/select/click"),
                xr.string_to_path(self.instance, "/user/hand/right/input/select/click"),
            ]
            simple_menu_path = [
                xr.string_to_path(self.instance, "/user/hand/left/input/menu/click"),
                xr.string_to_path(self.instance, "/user/hand/right/input/menu/click"),
            ]
            simple_pose_path = [
                xr.string_to_path(self.instance, "/user/hand/left/input/grip/pose"),
                xr.string_to_path(self.instance, "/user/hand/right/input/grip/pose"),
            ]
            
            simple_bindings = [
                # Next: select на обоих
                xr.ActionSuggestedBinding(self.next_action, select_path[0]),
                xr.ActionSuggestedBinding(self.next_action, select_path[1]),
                # Menu
                xr.ActionSuggestedBinding(self.menu_action, simple_menu_path[0]),
                xr.ActionSuggestedBinding(self.menu_action, simple_menu_path[1]),
                # Pose
                xr.ActionSuggestedBinding(self.pose_action, simple_pose_path[0]),
                xr.ActionSuggestedBinding(self.pose_action, simple_pose_path[1]),
            ]
            
            xr.suggest_interaction_profile_bindings(
                instance=self.instance,
                suggested_bindings=xr.InteractionProfileSuggestedBinding(
                    interaction_profile=xr.string_to_path(
                        self.instance,
                        "/interaction_profiles/khr/simple_controller",
                    ),
                    count_suggested_bindings=len(simple_bindings),
                    suggested_bindings=(xr.ActionSuggestedBinding * len(simple_bindings))(*simple_bindings),
                ),
            )
            log.debug("  ✓ Привязки Simple Controller зарегистрированы")
            
            # Присоединяем action set к сессии
            xr.attach_session_action_sets(
                session=self.session,
                attach_info=xr.SessionActionSetsAttachInfo(
                    count_action_sets=1,
                    action_sets=ctypes.pointer(self.action_set),
                ),
            )
            log.info("  ✓ Action set присоединён к сессии")
            
            # Создаём пространства для отслеживания позиции рук
            for hand_idx in [0, 1]:
                self.hand_spaces[hand_idx] = xr.create_action_space(
                    session=self.session,
                    create_info=xr.ActionSpaceCreateInfo(
                        action=self.pose_action,
                        subaction_path=self.hand_paths[hand_idx],
                    ),
                )
            log.info("  ✓ Hand spaces созданы")
            
            log.info("=" * 50)
            log.info("Управление контроллерами:")
            log.info("  Стики ↑↓ - IPD | ←→ - масштаб")
            log.info("  A/X - следующее | B/Y - предыдущее")
            log.info("  Grip + вращение запястья - смещение изображения")
            log.info("  Trigger + Grip - сброс смещения | Menu - выход")
            log.info("=" * 50)
            
        except Exception as e:
            log.warning(f"Не удалось инициализировать контроллеры: {e}")
            log.warning("Управление будет только с клавиатуры")
            self.action_set = None
        
    def poll_controller_actions(self):
        """Опрос состояния контроллеров и обработка ввода"""
        if self.action_set is None:
            return
        
        try:
            # Синхронизируем действия
            active_action_set = xr.ActiveActionSet(
                action_set=self.action_set,
                subaction_path=xr.NULL_PATH,
            )
            xr.sync_actions(
                session=self.session,
                sync_info=xr.ActionsSyncInfo(
                    count_active_action_sets=1,
                    active_action_sets=ctypes.pointer(active_action_set),
                ),
            )
            
            current_time = time.time()
            
            # === Обработка thumbstick с толерантностью к отклонениям ===
            # Считываем оба значения (X и Y) для каждого контроллера
            # и активируем только доминирующую ось
            ipd_changed = False
            scale_changed = False
            
            for hand_idx in [0, 1]:  # left, right
                # Получаем значения обеих осей
                stick_y = xr.get_action_state_float(
                    self.session,
                    xr.ActionStateGetInfo(
                        action=self.thumbstick_y_action,
                        subaction_path=self.hand_paths[hand_idx],
                    ),
                )
                stick_x = xr.get_action_state_float(
                    self.session,
                    xr.ActionStateGetInfo(
                        action=self.thumbstick_x_action,
                        subaction_path=self.hand_paths[hand_idx],
                    ),
                )
                
                y_val = stick_y.current_state if stick_y.is_active else 0.0
                x_val = stick_x.current_state if stick_x.is_active else 0.0
                
                abs_y = abs(y_val)
                abs_x = abs(x_val)
                
                # Определяем доминирующую ось (должна быть значительно больше другой)
                # Коэффициент 1.5 означает, что доминирующая ось должна быть в 1.5 раза больше
                dominance_ratio = 1.5
                
                # Y ось (IPD) - активируем только если Y доминирует
                # Линейное изменение IPD
                if abs_y > self.thumbstick_deadzone and abs_y > abs_x * dominance_ratio:
                    ipd_delta = y_val * self.ipd_step * 0.5  # Плавное изменение
                    self.ipd_offset += ipd_delta
                    ipd_changed = True
                
                # X ось (масштаб) - активируем только если X доминирует
                if abs_x > self.thumbstick_deadzone and abs_x > abs_y * dominance_ratio:
                    scale_delta = 1.0 + (x_val * self.thumbstick_speed_scale * 0.016)
                    self.quad_scale = max(0.1, min(5.0, self.quad_scale * scale_delta))
                    scale_changed = True
            
            if ipd_changed:
                self.update_distance_texture()
                self.save_settings()
            
            if scale_changed:
                self.save_settings()
            
            # === Обработка кнопок навигации ===
            # Следующее фото (A/X)
            next_state = xr.get_action_state_boolean(
                session=self.session,
                get_info=xr.ActionStateGetInfo(
                    action=self.next_action,
                    subaction_path=xr.NULL_PATH,
                ),
            )
            if (next_state.is_active and 
                next_state.current_state and 
                next_state.changed_since_last_sync and
                current_time - self.last_next_press > self.button_cooldown):
                self.next_image()
                self.last_next_press = current_time
                log.debug("Controller: Next image")
            
            # Предыдущее фото (B/Y)
            prev_state = xr.get_action_state_boolean(
                session=self.session,
                get_info=xr.ActionStateGetInfo(
                    action=self.prev_action,
                    subaction_path=xr.NULL_PATH,
                ),
            )
            if (prev_state.is_active and 
                prev_state.current_state and 
                prev_state.changed_since_last_sync and
                current_time - self.last_prev_press > self.button_cooldown):
                self.prev_image()
                self.last_prev_press = current_time
                log.debug("Controller: Prev image")
            
            # === Кнопка выхода (Menu) ===
            menu_state = xr.get_action_state_boolean(
                session=self.session,
                get_info=xr.ActionStateGetInfo(
                    action=self.menu_action,
                    subaction_path=xr.NULL_PATH,
                ),
            )
            if menu_state.is_active and menu_state.current_state and menu_state.changed_since_last_sync:
                log.info("Controller: Menu pressed - выход")
                self.should_quit = True
            
            # === Получаем состояние триггеров и grip ===
            trigger_values = [0.0, 0.0]
            grip_values = [0.0, 0.0]
            
            for hand_idx in [0, 1]:
                trigger_state = xr.get_action_state_float(
                    self.session,
                    xr.ActionStateGetInfo(
                        action=self.trigger_action,
                        subaction_path=self.hand_paths[hand_idx],
                    ),
                )
                if trigger_state.is_active:
                    trigger_values[hand_idx] = trigger_state.current_state
                
                grip_state = xr.get_action_state_float(
                    self.session,
                    xr.ActionStateGetInfo(
                        action=self.grip_action,
                        subaction_path=self.hand_paths[hand_idx],
                    ),
                )
                if grip_state.is_active:
                    grip_values[hand_idx] = grip_state.current_state
            
            # Логируем значения если что-то зажато (для отладки)
            # if trigger_values[0] > 0.5 or trigger_values[1] > 0.5:
            #     log.debug(f"Триггеры: L={trigger_values[0]:.2f} R={trigger_values[1]:.2f}")
            # if grip_values[0] > 0.5 or grip_values[1] > 0.5:
            #     log.debug(f"Grip: L={grip_values[0]:.2f} R={grip_values[1]:.2f}")
            
            # === Перемещение изображения (grip + вращение контроллера) ===
            for hand_idx in [0, 1]:
                grip_held = grip_values[hand_idx] > 0.5
                trigger_held = trigger_values[hand_idx] > 0.5
                
                # Сброс смещения (триггер + grip одновременно на любой руке)
                if grip_held and trigger_held:
                    if not hasattr(self, '_reset_held'):
                        self._reset_held = False
                    if not self._reset_held:
                        self._reset_held = True
                        self.image_offset_x = 0.0
                        self.image_offset_y = 0.0
                        log.info("Controller: Сброс смещения изображения")
                    self.controller_grab_rot[hand_idx] = None
                    continue
                
                if not grip_held:
                    # Grip не зажат - сбрасываем начальную ориентацию
                    self.controller_grab_rot[hand_idx] = None
                    continue
                
                # Получаем ориентацию контроллера
                if self.hand_spaces[hand_idx] is None:
                    continue
                    
                try:
                    pose_state = xr.get_action_state_pose(
                        session=self.session,
                        get_info=xr.ActionStateGetInfo(
                            action=self.pose_action,
                            subaction_path=self.hand_paths[hand_idx],
                        ),
                    )
                    
                    if not pose_state.is_active:
                        continue
                    
                    space_location = xr.locate_space(
                        space=self.hand_spaces[hand_idx],
                        base_space=self.local_space,
                        time=self.predicted_display_time,
                    )
                    
                    if not (space_location.location_flags & xr.SPACE_LOCATION_ORIENTATION_VALID_BIT):
                        continue
                    
                    # Получаем ориентацию контроллера (кватернион)
                    q = space_location.pose.orientation
                    
                    # Конвертируем кватернион в углы Эйлера (yaw, pitch)
                    sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
                    cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
                    pitch = np.arctan2(sinr_cosp, cosr_cosp)  # Вращение вокруг X
                    
                    siny_cosp = 2 * (q.w * q.y - q.z * q.x)
                    yaw = np.arcsin(np.clip(siny_cosp, -1, 1))  # Вращение вокруг Y
                    
                    current_rot = (np.degrees(yaw), np.degrees(pitch))
                    
                    # Grip зажат - перемещение изображения
                    if self.controller_grab_rot[hand_idx] is not None:
                        # Вычисляем дельту вращения контроллера
                        delta_yaw = current_rot[0] - self.controller_grab_rot[hand_idx][0]
                        delta_pitch = current_rot[1] - self.controller_grab_rot[hand_idx][1]
                        
                        # Обрабатываем переход через 180/-180
                        if delta_yaw > 90:
                            delta_yaw -= 180
                        elif delta_yaw < -90:
                            delta_yaw += 180
                        
                        # Перемещаем изображение
                        self.image_offset_x -= delta_yaw * self.translation_sensitivity
                        self.image_offset_y += delta_pitch * self.translation_sensitivity
                    
                    # Обновляем референсную точку
                    self.controller_grab_rot[hand_idx] = current_rot
                        
                except Exception as e:
                    log.debug(f"Controller error [{hand_idx}]: {e}")
            
            # Сброс флага reset когда отпустили
            if not any(grip_values[i] > 0.5 and trigger_values[i] > 0.5 for i in [0, 1]):
                self._reset_held = False
                
        except Exception as e:
            # Логируем ошибки контроллеров
            log.debug(f"Controller error: {e}")
        
    def create_swapchains(self):
        """Создание swapchain для каждого вида"""
        log.info("=" * 50)
        log.info("Создание swapchains...")
        glfw.poll_events()
        
        # Получаем конфигурации видов
        log.debug("Получение конфигураций видов...")
        self.view_configs = xr.enumerate_view_configuration_views(
            self.instance,
            self.system_id,
            xr.ViewConfigurationType.PRIMARY_STEREO,
        )
        log.info(f"  Количество видов: {len(self.view_configs)}")
        
        # Получаем поддерживаемые форматы
        swapchain_formats = xr.enumerate_swapchain_formats(self.session)
        log.debug(f"  Поддерживаемых форматов: {len(swapchain_formats)}")
        
        # Предпочитаем SRGB формат
        preferred_formats = [GL_SRGB8_ALPHA8, GL_RGBA8]
        selected_format = GL_RGBA8
        for fmt in preferred_formats:
            if fmt in swapchain_formats:
                selected_format = fmt
                break
        log.debug(f"  Выбранный формат: {selected_format}")
        
        for i, view_config in enumerate(self.view_configs):
            log.info(f"  View {i}: {view_config.recommended_image_rect_width}x{view_config.recommended_image_rect_height}")
            
            # Создаём swapchain
            swapchain_info = xr.SwapchainCreateInfo(
                usage_flags=xr.SwapchainUsageFlags.SAMPLED_BIT | xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT,
                format=selected_format,
                sample_count=1,
                width=view_config.recommended_image_rect_width,
                height=view_config.recommended_image_rect_height,
                face_count=1,
                array_size=1,
                mip_count=1,
            )
            
            swapchain = xr.create_swapchain(self.session, swapchain_info)
            self.swapchains.append(swapchain)
            
            # Получаем изображения swapchain
            images = xr.enumerate_swapchain_images(swapchain, xr.SwapchainImageOpenGLKHR)
            self.swapchain_images.append(images)
            
            # Создаём framebuffer для каждого изображения
            framebuffers = []
            for img in images:
                fb = glGenFramebuffers(1)
                glBindFramebuffer(GL_FRAMEBUFFER, fb)
                
                # Создаём depth buffer
                depth_buffer = glGenRenderbuffers(1)
                glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer)
                glRenderbufferStorage(
                    GL_RENDERBUFFER, GL_DEPTH24_STENCIL8,
                    view_config.recommended_image_rect_width,
                    view_config.recommended_image_rect_height
                )
                glFramebufferRenderbuffer(
                    GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                    GL_RENDERBUFFER, depth_buffer
                )
                
                glFramebufferTexture2D(
                    GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                    GL_TEXTURE_2D, img.image, 0
                )
                
                framebuffers.append(fb)
            
            self.framebuffers.append(framebuffers)
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
    def create_shaders(self):
        """Создание шейдерной программы"""
        # Компилируем вершинный шейдер
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, self.VERTEX_SHADER)
        glCompileShader(vertex_shader)
        if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
            raise RuntimeError(f"Vertex shader error: {glGetShaderInfoLog(vertex_shader)}")
        
        # Компилируем фрагментный шейдер
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, self.FRAGMENT_SHADER)
        glCompileShader(fragment_shader)
        if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
            raise RuntimeError(f"Fragment shader error: {glGetShaderInfoLog(fragment_shader)}")
        
        # Создаём программу
        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, vertex_shader)
        glAttachShader(self.shader_program, fragment_shader)
        glLinkProgram(self.shader_program)
        if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
            raise RuntimeError(f"Program link error: {glGetProgramInfoLog(self.shader_program)}")
        
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        
        # Получаем locations атрибутов
        self.vertex_pos_loc = glGetAttribLocation(self.shader_program, "VertexPos")
        self.vertex_uv_loc = glGetAttribLocation(self.shader_program, "VertexUV")
        
    def create_quad(self):
        """Создание четырёхугольника для отображения изображений"""
        # Прямоугольник 1x1 метр в плоскости XY
        # Позиция (x, y, z), Текстура (u, v)
        vertices = np.array([
            # Треугольник 1
            -0.5, -0.5, 0.0,  0.0, 1.0,
             0.5, -0.5, 0.0,  1.0, 1.0,
             0.5,  0.5, 0.0,  1.0, 0.0,
            # Треугольник 2
            -0.5, -0.5, 0.0,  0.0, 1.0,
             0.5,  0.5, 0.0,  1.0, 0.0,
            -0.5,  0.5, 0.0,  0.0, 0.0,
        ], dtype=np.float32)
        
        self.quad_vao = glGenVertexArrays(1)
        self.quad_vbo = glGenBuffers(1)
        
        glBindVertexArray(self.quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Атрибут позиции (location из шейдера)
        glEnableVertexAttribArray(self.vertex_pos_loc)
        glVertexAttribPointer(self.vertex_pos_loc, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
        
        # Атрибут текстурных координат
        glEnableVertexAttribArray(self.vertex_uv_loc)
        glVertexAttribPointer(self.vertex_uv_loc, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
        
        glBindVertexArray(0)
        
    def create_textures(self):
        """Создание текстур (только для текущего изображения)"""
        # Создаём текстуру только для текущего изображения (ленивая загрузка)
        if self.images:
            self.ensure_current_texture()
        self.update_distance_texture()
        self.update_counter_texture()
    
    def ensure_current_texture(self):
        """Убедиться, что текстура текущего изображения создана"""
        if not self.images:
            return
        
        # Сначала загружаем данные изображения если нужно
        self._load_image_data(self.current_index)
        
        current = self.images[self.current_index]
        if current.left_texture is None:
            current.create_textures()
    
    def update_distance_texture(self):
        """Обновление текстуры с расстоянием и режимом просмотра"""
        from PIL import ImageDraw, ImageFont
        
        # Удаляем старую текстуру
        if self.distance_texture:
            glDeleteTextures(1, [self.distance_texture])
        
        # Текст с режимом и IPD
        mode_name = "Cross" if self.cross_eyed_mode else "Parallel"
        ipd_mm = self.ipd_offset * 1000  # Конвертируем в миллиметры
        text = f"{mode_name} | IPD: {ipd_mm:+.1f} мм"
        
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except:
            font = ImageFont.load_default()
        
        # Измеряем размер текста
        dummy_img = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0] + 16
        text_height = bbox[3] - bbox[1] + 8
        
        # Создаём изображение
        img = Image.new('RGBA', (text_width, text_height), (0, 0, 0, 150))
        draw = ImageDraw.Draw(img)
        draw.text((8, 0), text, fill=(200, 200, 200, 255), font=font)
        
        self.distance_aspect = text_width / text_height
        
        # Создаём текстуру
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        img_data = np.array(img, dtype=np.uint8)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_width, text_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        
        self.distance_texture = texture
    
    def update_counter_texture(self):
        """Обновление текстуры со счётчиком изображений"""
        from PIL import ImageDraw, ImageFont
        
        # Удаляем старую текстуру
        if self.counter_texture:
            glDeleteTextures(1, [self.counter_texture])
            self.counter_texture = None
        
        if not self.images:
            return
        
        # Текст со счётчиком
        text = f"({self.current_index + 1}/{len(self.images)})"
        
        try:
            font = ImageFont.truetype("arial.ttf", 48)
        except:
            font = ImageFont.load_default()
        
        # Измеряем размер текста
        dummy_img = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0] + 20
        text_height = bbox[3] - bbox[1] + 10
        
        # Создаём изображение
        img = Image.new('RGBA', (text_width, text_height), (0, 0, 0, 180))
        draw = ImageDraw.Draw(img)
        draw.text((10, 0), text, fill=(255, 255, 255, 255), font=font)
        
        self.counter_aspect = text_width / text_height
        
        # Создаём текстуру
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        img_data = np.array(img, dtype=np.uint8)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_width, text_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        
        self.counter_texture = texture
            
    def create_projection_matrix(self, fov: xr.Fovf, near: float = 0.1, far: float = 100.0) -> np.ndarray:
        """Создание матрицы проекции из FOV"""
        tan_left = np.tan(fov.angle_left)
        tan_right = np.tan(fov.angle_right)
        tan_up = np.tan(fov.angle_up)
        tan_down = np.tan(fov.angle_down)
        
        tan_width = tan_right - tan_left
        tan_height = tan_up - tan_down
        
        matrix = np.zeros((4, 4), dtype=np.float32)
        matrix[0, 0] = 2.0 / tan_width
        matrix[1, 1] = 2.0 / tan_height
        matrix[0, 2] = (tan_right + tan_left) / tan_width
        matrix[1, 2] = (tan_up + tan_down) / tan_height
        matrix[2, 2] = -(far + near) / (far - near)
        matrix[2, 3] = -1.0
        matrix[3, 2] = -(2.0 * far * near) / (far - near)
        
        return matrix
        
    def create_view_matrix(self, pose: xr.Posef) -> np.ndarray:
        """Создание матрицы вида из позы"""
        # Статичное изображение - единичная матрица (изображение следует за головой)
        return np.eye(4, dtype=np.float32)
        
    def create_model_matrix(self) -> np.ndarray:
        """Создание матрицы модели для плоскости изображения"""
        # Получаем соотношение сторон изображения
        if self.images:
            img = self.images[self.current_index]
            height, width = img.left.shape[:2]
            aspect = width / height
        else:
            aspect = 16.0 / 9.0
        
        # Масштаб с учётом соотношения сторон
        scale_x = self.image_scale * aspect
        scale_y = self.image_scale
        
        model = np.array([
            [scale_x, 0, 0, 0],
            [0, scale_y, 0, 0],
            [0, 0, 1, 0],
            [0, 0, -self.image_distance, 1],  # Позиция перед пользователем
        ], dtype=np.float32)
        
        return model
        
    def render_eye(self, view_index: int, view: xr.View, swapchain_image_index: int):
        """Рендеринг для одного глаза"""
        if not self.images:
            return
            
        view_config = self.view_configs[view_index]
        vp_width = view_config.recommended_image_rect_width
        vp_height = view_config.recommended_image_rect_height
        
        # Привязываем framebuffer
        fb = self.framebuffers[view_index][swapchain_image_index]
        glBindFramebuffer(GL_FRAMEBUFFER, fb)
        glViewport(0, 0, vp_width, vp_height)
        
        # Очищаем буферы (тёмно-серый фон)
        glClearColor(0.05, 0.05, 0.05, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Включаем depth test
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        
        # Используем шейдер
        glUseProgram(self.shader_program)
        
        # === МАТРИЦЫ КАК В ПРИМЕРЕ PYOPENXR ===
        
        # 1. Матрица проекции из FOV
        proj = Matrix4x4f.create_projection_fov(view.fov, 0.05, 100.0)
        
        # 2. Матрица вида (инвертированная поза камеры)
        pose = view.pose
        scale_one = xr.Vector3f(1, 1, 1)
        to_view = Matrix4x4f.create_translation_rotation_scale(pose.position, pose.orientation, scale_one)
        view_matrix = to_view.invert_rigid_body()
        
        # 3. Получаем размеры изображения для пропорций
        current_image = self.images[self.current_index]
        img_height, img_width = current_image.left.shape[:2]
        aspect_ratio = img_width / img_height
        
        # 4. Матрица модели - прямоугольник перед пользователем
        # Позиция фиксированная, вращение "себя" уже применено к view matrix
        if self.head_height is None:
            self.head_height = pose.position.y
        eye_height = self.head_height
        
        # Позиция с учётом смещения (без вращения - оно в view matrix)
        # IPD: левый глаз (view_index=0) смещаем влево, правый (view_index=1) вправо
        ipd_shift = self.ipd_offset / 2 * (-1 if view_index == 0 else 1)
        quad_pos = xr.Vector3f(self.image_offset_x + ipd_shift, eye_height + self.image_offset_y, -self.quad_distance)
        
        # Ориентация: изображение смотрит на пользователя (без вращения)
        quad_rot = xr.Quaternionf(0, 0, 0, 1)
        
        # Физический размер = base_size * quad_scale * расстояние (для сохранения углового размера)
        physical_scale = self.base_size * self.quad_scale * self.quad_distance
        quad_scale = xr.Vector3f(physical_scale * aspect_ratio, physical_scale, 1)
        model = Matrix4x4f.create_translation_rotation_scale(quad_pos, quad_rot, quad_scale)
        
        # 5. MVP = Projection * View * Model
        vp = proj @ view_matrix
        mvp = vp @ model
        
        # Устанавливаем uniform матрицы
        mvp_loc = glGetUniformLocation(self.shader_program, "ModelViewProjection")
        glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp.as_numpy())
        
        # Используем текстуру
        use_tex_loc = glGetUniformLocation(self.shader_program, "uUseTexture")
        glUniform1i(use_tex_loc, 1)
        
        # Привязываем текстуру (левую для левого глаза, правую для правого)
        # В режиме cross-eyed текстуры меняются местами
        if self.cross_eyed_mode:
            texture = current_image.right_texture if view_index == 0 else current_image.left_texture
        else:
            texture = current_image.left_texture if view_index == 0 else current_image.right_texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)
        tex_loc = glGetUniformLocation(self.shader_program, "uTexture")
        glUniform1i(tex_loc, 0)
        
        # Рисуем изображение
        glBindVertexArray(self.quad_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        
        # === РИСУЕМ НАЗВАНИЕ ФАЙЛА И РАССТОЯНИЕ ПОД ИЗОБРАЖЕНИЕМ ===
        # Включаем прозрачность
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Физический размер изображения = base_size * quad_scale * distance (для сохранения углового размера)
        physical_scale = self.base_size * self.quad_scale * self.quad_distance
        
        # Коэффициент масштабирования надписей (сохраняем угловой размер)
        label_scale_factor = self.quad_distance
        
        # Позиция под изображением (с учётом смещения)
        label_base_y = eye_height + self.image_offset_y - (physical_scale * 0.5) - 0.02 * label_scale_factor
        label_x = self.image_offset_x
        label_z = -self.quad_distance + 0.01
        
        current_label_offset = 0.0
        
        # 1. Название файла и счётчик
        if current_image.name_texture:
            text_height = 0.03 * label_scale_factor  # Угловой размер ~1.7°
            text_width = text_height * current_image.name_aspect
            
            # Если есть счётчик, рисуем имя файла левее центра
            counter_width = 0.0
            if self.counter_texture:
                counter_width = text_height * self.counter_aspect
            
            total_width = text_width + counter_width + 0.005 * label_scale_factor  # Отступ между именем и счётчиком
            
            current_label_offset -= text_height
            # Смещаем имя файла влево, чтобы центрировать всю строку
            name_x = label_x - total_width / 2 + text_width / 2
            text_pos = xr.Vector3f(name_x, label_base_y + current_label_offset, label_z)
            text_scale = xr.Vector3f(text_width, text_height, 1)
            text_model = Matrix4x4f.create_translation_rotation_scale(text_pos, quad_rot, text_scale)
            
            text_mvp = vp @ text_model
            glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, text_mvp.as_numpy())
            
            glBindTexture(GL_TEXTURE_2D, current_image.name_texture)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            
            # Рисуем счётчик справа от имени файла
            if self.counter_texture:
                counter_x = name_x + text_width / 2 + 0.005 * label_scale_factor + counter_width / 2
                counter_pos = xr.Vector3f(counter_x, label_base_y + current_label_offset, label_z)
                counter_scale = xr.Vector3f(counter_width, text_height, 1)
                counter_model = Matrix4x4f.create_translation_rotation_scale(counter_pos, quad_rot, counter_scale)
                
                counter_mvp = vp @ counter_model
                glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, counter_mvp.as_numpy())
                
                glBindTexture(GL_TEXTURE_2D, self.counter_texture)
                glDrawArrays(GL_TRIANGLES, 0, 6)
            
            current_label_offset -= 0.005 * label_scale_factor  # Отступ между названием и расстоянием
        
        # 2. Расстояние
        if self.distance_texture:
            dist_height = 0.02 * label_scale_factor  # Угловой размер ~1.1°
            dist_width = dist_height * self.distance_aspect
            
            current_label_offset -= dist_height
            dist_pos = xr.Vector3f(label_x, label_base_y + current_label_offset, label_z)
            dist_scale = xr.Vector3f(dist_width, dist_height, 1)
            dist_model = Matrix4x4f.create_translation_rotation_scale(dist_pos, quad_rot, dist_scale)
            
            dist_mvp = vp @ dist_model
            glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, dist_mvp.as_numpy())
            
            glBindTexture(GL_TEXTURE_2D, self.distance_texture)
            glDrawArrays(GL_TRIANGLES, 0, 6)
        
        glDisable(GL_BLEND)
        
        glBindVertexArray(0)
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
    def render_frame(self):
        """Рендеринг одного кадра"""
        # Ожидаем кадр (это блокирующий вызов!)
        frame_state = xr.wait_frame(self.session)
        
        # Сохраняем время для использования в poll_controller_actions
        self.predicted_display_time = frame_state.predicted_display_time
        
        # Начинаем кадр
        xr.begin_frame(self.session)
        
        layers = []
        
        if frame_state.should_render:
            # Получаем положение видов
            view_state, views = xr.locate_views(
                self.session,
                xr.ViewLocateInfo(
                    view_configuration_type=xr.ViewConfigurationType.PRIMARY_STEREO,
                    display_time=frame_state.predicted_display_time,
                    space=self.local_space,
                ),
            )
            
            projection_views = []
            
            for i, view in enumerate(views):
                # Получаем индекс swapchain изображения
                swapchain_index = xr.acquire_swapchain_image(
                    self.swapchains[i],
                    xr.SwapchainImageAcquireInfo(),
                )
                
                # Таймаут 1 секунда вместо бесконечного ожидания
                timeout_ns = 1_000_000_000  # 1 секунда в наносекундах
                xr.wait_swapchain_image(
                    self.swapchains[i],
                    xr.SwapchainImageWaitInfo(timeout=timeout_ns),
                )
                
                # Рендерим
                self.render_eye(i, view, swapchain_index)
                
                # Освобождаем swapchain изображение
                xr.release_swapchain_image(
                    self.swapchains[i],
                    xr.SwapchainImageReleaseInfo(),
                )
                
                # Добавляем projection view
                view_config = self.view_configs[i]
                projection_views.append(
                    xr.CompositionLayerProjectionView(
                        pose=view.pose,
                        fov=view.fov,
                        sub_image=xr.SwapchainSubImage(
                            swapchain=self.swapchains[i],
                            image_rect=xr.Rect2Di(
                                offset=xr.Offset2Di(0, 0),
                                extent=xr.Extent2Di(
                                    view_config.recommended_image_rect_width,
                                    view_config.recommended_image_rect_height,
                                ),
                            ),
                            image_array_index=0,
                        ),
                    )
                )
            
            # Создаём projection layer
            projection_layer = xr.CompositionLayerProjection(
                space=self.local_space,
                views=projection_views,
            )
            layers.append(ctypes.byref(projection_layer))
        
        # Завершаем кадр
        xr.end_frame(
            self.session,
            xr.FrameEndInfo(
                display_time=frame_state.predicted_display_time,
                environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
                layers=layers,
            ),
        )
        
    def handle_session_state_change(self, state):
        """Обработка изменения состояния сессии"""
        # Преобразуем int в SessionState если нужно
        if isinstance(state, int):
            state = xr.SessionState(state)
        
        log.info(f">>> Состояние сессии изменилось: {state.name}")
        
        if state == xr.SessionState.READY:
            log.info("  Сессия готова, начинаем...")
            begin_info = xr.SessionBeginInfo(
                primary_view_configuration_type=xr.ViewConfigurationType.PRIMARY_STEREO,
            )
            try:
                xr.begin_session(self.session, begin_info)
                self.session_running = True
                log.info("  ✓ Сессия запущена! Рендеринг активен.")
            except Exception as e:
                log.error(f"  ✗ Ошибка запуска сессии: {e}")
                raise
            
        elif state == xr.SessionState.SYNCHRONIZED:
            log.info("  Сессия синхронизирована с runtime")
            
        elif state == xr.SessionState.VISIBLE:
            log.info("  Сессия видима (но не в фокусе)")
            
        elif state == xr.SessionState.FOCUSED:
            log.info("  ✓ Сессия в фокусе - полный рендеринг")
            
        elif state == xr.SessionState.STOPPING:
            log.info("  Сессия останавливается...")
            try:
                xr.end_session(self.session)
                self.session_running = False
                log.info("  Сессия остановлена")
            except Exception as e:
                log.error(f"  Ошибка остановки сессии: {e}")
            
        elif state == xr.SessionState.EXITING:
            log.info("  Сессия завершается (EXITING)")
            self.should_quit = True
            
        elif state == xr.SessionState.LOSS_PENDING:
            log.warning("  ⚠ Потеря сессии (LOSS_PENDING)")
            self.should_quit = True
            
        elif state == xr.SessionState.IDLE:
            log.info("  Сессия в режиме ожидания (IDLE)")
            
    def poll_events(self):
        """Обработка событий OpenXR"""
        events_processed = 0
        while True:
            try:
                event = xr.poll_event(self.instance)
                if event is None:
                    break
                
                events_processed += 1
                log.debug(f"  OpenXR Event: {event.type}")
                    
                if event.type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                    session_state_event = ctypes.cast(
                        ctypes.byref(event),
                        ctypes.POINTER(xr.EventDataSessionStateChanged)
                    ).contents
                    self.handle_session_state_change(session_state_event.state)
                elif event.type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                    log.error("  ✗ OpenXR Instance потерян!")
                    self.should_quit = True
                    
            except xr.EventUnavailable:
                break
        
        if events_processed > 0:
            log.debug(f"  Обработано событий: {events_processed}")
                
    def load_settings(self) -> dict:
        """Загрузка настроек из файла"""
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            log.debug(f"Не удалось загрузить настройки: {e}")
        return {}
    
    def save_settings(self):
        """Сохранение настроек в файл"""
        try:
            settings = {
                "distance": self.quad_distance,
                "scale": self.quad_scale,
                "ipd_offset": self.ipd_offset,
                "cross_eyed": self.cross_eyed_mode
            }
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            log.debug(f"Не удалось сохранить настройки: {e}")
    
    def next_image(self):
        """Переход к следующему изображению"""
        if self.images and len(self.images) > 1:
            self.current_index = (self.current_index + 1) % len(self.images)
            self.ensure_current_texture()
            self.update_counter_texture()
            self._preload_nearby()
            
    def prev_image(self):
        """Переход к предыдущему изображению"""
        if self.images and len(self.images) > 1:
            self.current_index = (self.current_index - 1) % len(self.images)
            self.ensure_current_texture()
            self.update_counter_texture()
            self._preload_nearby()
    
    def delete_current_image(self):
        """Удаление текущего изображения в корзину"""
        log.info("Попытка удаления изображения...")
        
        if not self.images:
            log.warning("Нет изображений для удаления")
            return
        
        if not HAS_SEND2TRASH:
            log.error("send2trash не установлен! Выполните: pip install send2trash")
            return
        
        current_image = self.images[self.current_index]
        image_path = current_image.path  # Используем сохранённый путь
        
        log.info(f"Удаление: {current_image.name}")
        log.info(f"Путь: {image_path}")
        
        if not image_path:
            log.error("Путь к файлу не сохранён в объекте изображения")
            return
            
        if not os.path.exists(image_path):
            log.error(f"Файл не существует: {image_path}")
            return
        
        try:
            # Удаляем текстуры
            current_image.delete_textures()
            
            # Отправляем в корзину
            send2trash(image_path)
            log.info(f"🗑️ Удалено в корзину: {current_image.name}")
            
            # Удаляем из списков
            self.images.pop(self.current_index)
            
            # Удаляем из image_paths
            norm_path = os.path.normpath(image_path)
            self.image_paths = [p for p in self.image_paths if os.path.normpath(p) != norm_path]
            
            # Корректируем индекс
            if self.images:
                if self.current_index >= len(self.images):
                    self.current_index = len(self.images) - 1
                self.update_counter_texture()
            else:
                log.info("Все изображения удалены")
                self.update_counter_texture()
                
        except Exception as e:
            log.error(f"Ошибка удаления: {e}")
            import traceback
            log.error(traceback.format_exc())
    
    def refresh_images(self):
        """Обновление списка файлов из текущей директории"""
        if not self.image_paths:
            log.warning("Нет пути для обновления")
            return
        
        # Определяем папку из первого пути
        first_path = self.image_paths[0]
        if os.path.isfile(first_path):
            folder = os.path.dirname(first_path)
        else:
            folder = first_path
        
        if not folder or not os.path.isdir(folder):
            log.warning(f"Папка не найдена: {folder}")
            return
        
        # Запоминаем текущее изображение
        current_name = self.images[self.current_index].name if self.images else None
        
        # Удаляем старые текстуры
        for img in self.images:
            img.delete_textures()
        self.images.clear()
        
        # Сканируем папку заново
        new_paths = find_images(folder)
        if not new_paths:
            log.warning(f"Изображения не найдены в: {folder}")
            return
        
        # Фильтруем _right файлы
        if not self.sbs_mode:
            new_paths = [p for p in new_paths if '_right' not in p.lower()]
        
        self.image_paths = new_paths
        log.info(f"🔄 Обновлено: {len(new_paths)} файлов")
        
        # Создаём placeholder-объекты (ленивая загрузка)
        for path in self.image_paths:
            img = StereoImage(np.array([]), np.array([]), Path(path).name, os.path.abspath(path))
            img._loaded = False
            self.images.append(img)
        
        # Восстанавливаем позицию на том же изображении если возможно
        self.current_index = 0
        if current_name:
            for i, img in enumerate(self.images):
                if img.name == current_name:
                    self.current_index = i
                    break
        
        # Загружаем текущее изображение
        if self.images:
            self.ensure_current_texture()
            self.update_counter_texture()
    
    def add_images_from_paths(self, paths: List[str], replace: bool = False):
        """
        Добавление новых изображений в просмотрщик.
        
        Args:
            paths: Список путей к изображениям
            replace: Если True, заменить текущие изображения, иначе добавить
        """
        if replace:
            # Удаляем старые текстуры
            for img in self.images:
                img.delete_textures()
            self.images.clear()
            self.current_index = 0
        
        for path in paths:
            try:
                if self.sbs_mode:
                    img = StereoImage.from_sbs(path)
                else:
                    if '_left' in path.lower():
                        right_path = path.lower().replace('_left', '_right')
                        if os.path.exists(right_path):
                            img = StereoImage.from_separate_files(path, right_path)
                        else:
                            continue
                    elif '_right' in path.lower():
                        continue
                    else:
                        img = StereoImage.from_sbs(path)
                
                img.create_textures()
                self.images.append(img)
                print(f"  Добавлено: {img.name}")
            except Exception as e:
                print(f"  Ошибка загрузки {path}: {e}")
        
        if self.images:
            print(f"Всего изображений: {len(self.images)}")
    
    def open_files_dialog(self, replace: bool = True):
        """Открыть диалог выбора файлов"""
        print("\nОткрытие диалога выбора файлов...")
        files = open_file_dialog(
            title="Выберите стереоизображения (SBS)",
            multiple=True
        )
        if files:
            self.add_images_from_paths(files, replace=replace)
    
    def open_folder_dialog(self):
        """Открыть диалог выбора папки"""
        print("\nОткрытие диалога выбора папки...")
        folder = open_folder_dialog(
            title="Выберите папку с изображениями"
        )
        if folder:
            images = find_images(folder)
            if images:
                self.add_images_from_paths(images, replace=True)
            else:
                print("В выбранной папке нет изображений!")
            
    def run(self):
        """Главный цикл приложения"""
        log.info("=" * 60)
        log.info("      VR STEREO IMAGE VIEWER")
        log.info("=" * 60)
        log.info(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log.info(f"Python: {sys.version}")
        log.info(f"Платформа: {sys.platform}")
        log.info("")
        
        # Инициализация
        self.load_images()
        
        if not self.images:
            log.error("Нет изображений для отображения!")
            return
        
        try:
            # ВАЖНО: Порядок инициализации для OpenXR + OpenGL:
            # 1. Сначала OpenXR instance (без сессии)
            # 2. Потом GLFW + OpenGL контекст
            # 3. Потом OpenXR сессия
            self.initialize_openxr_instance()
            self.initialize_glfw()
            self.create_session()
            self.create_swapchains()
            
            log.info("=" * 50)
            log.info("Создание OpenGL ресурсов...")
            self.create_shaders()
            log.debug("  ✓ Шейдеры созданы")
            self.create_quad()
            log.debug("  ✓ Геометрия создана")
            self.create_textures()
            log.info(f"  ✓ Текстура создана")
            
            log.info("=" * 50)
            log.info("ИНИЦИАЛИЗАЦИЯ ЗАВЕРШЕНА")
            log.info("=" * 50)
            log.info("")
            log.info("Управление клавиатурой:")
            log.info("  Перетащите файлы на окно для загрузки")
            log.info("  O - открыть файлы | F - открыть папку")
            log.info("  ←/→ или E/Q - переключение изображений")
            log.info("  +/- или D/A - масштаб")
            log.info("  W/S или 1/3 - IPD ±40мм | 2 - сброс IPD")
            log.info("  C - cross-eyed/parallel | Home - сброс смещения")
            log.info("  Delete - удалить фото | ESC - выход")
            log.info("")
            log.info("Управление контроллерами Meta Quest 3:")
            log.info("  Стики ↑↓ - IPD | ←→ - масштаб")
            log.info("  Grip + вращение запястья - смещение изображения")
            log.info("  A/X - след. | B/Y - пред. | Menu - выход")
            log.info("  Trigger + Grip - сброс смещения")
            log.info("")
            log.info("Ожидание готовности VR сессии...")
            
            # Устанавливаем callback для клавиатуры
            def key_callback(window, key, scancode, action, mods):
                if action == glfw.PRESS or action == glfw.REPEAT:
                    if key == glfw.KEY_ESCAPE:
                        self.should_quit = True
                    elif key == glfw.KEY_O:
                        self.open_files_dialog(replace=True)
                    elif key == glfw.KEY_F:
                        self.open_folder_dialog()
                    elif key == glfw.KEY_RIGHT or key == glfw.KEY_E:
                        self.next_image()
                    elif key == glfw.KEY_LEFT or key == glfw.KEY_Q:
                        self.prev_image()
                    elif key == glfw.KEY_EQUAL or key == glfw.KEY_KP_ADD or key == glfw.KEY_D:
                        self.quad_scale = min(5.0, self.quad_scale * 1.1)
                        self.save_settings()
                    elif key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT or key == glfw.KEY_A:
                        self.quad_scale = max(0.1, self.quad_scale / 1.1)
                        self.save_settings()
                    elif key == glfw.KEY_W:
                        # Увеличить IPD (изображения расходятся) - то же что клавиша 3
                        self.ipd_offset += self.ipd_step
                        log.info(f"IPD: {self.ipd_offset * 1000:+.1f} мм")
                        self.update_distance_texture()
                        self.save_settings()
                    elif key == glfw.KEY_S:
                        # Уменьшить IPD (изображения сходятся) - то же что клавиша 1
                        self.ipd_offset -= self.ipd_step
                        log.info(f"IPD: {self.ipd_offset * 1000:+.1f} мм")
                        self.update_distance_texture()
                        self.save_settings()
                    elif key == glfw.KEY_R:
                        self.head_height = None
                    elif key == glfw.KEY_C:
                        # Переключение режима cross-eyed / parallel
                        self.cross_eyed_mode = not self.cross_eyed_mode
                        mode_name = "Cross-eyed" if self.cross_eyed_mode else "Parallel"
                        log.info(f"Режим просмотра: {mode_name}")
                        self.update_distance_texture()
                        self.save_settings()
                    elif key == glfw.KEY_1 or key == glfw.KEY_KP_1:
                        # Уменьшить IPD (изображения сходятся)
                        self.ipd_offset -= self.ipd_step
                        log.info(f"IPD: {self.ipd_offset * 1000:+.1f} мм")
                        self.update_distance_texture()
                        self.save_settings()
                    elif key == glfw.KEY_2 or key == glfw.KEY_KP_2:
                        # Сброс IPD
                        self.ipd_offset = 0.0
                        log.info("IPD сброшен в 0")
                        self.update_distance_texture()
                        self.save_settings()
                    elif key == glfw.KEY_3 or key == glfw.KEY_KP_3:
                        # Увеличить IPD (изображения расходятся)
                        self.ipd_offset += self.ipd_step
                        log.info(f"IPD: {self.ipd_offset * 1000:+.1f} мм")
                        self.update_distance_texture()
                        self.save_settings()
                    elif key == glfw.KEY_HOME:
                        # Сброс смещения изображения
                        self.image_offset_x = 0.0
                        self.image_offset_y = 0.0
                        log.info("Сброс смещения изображения")
                    elif key == glfw.KEY_DELETE:
                        # Удаление текущего изображения в корзину
                        self.delete_current_image()
                    elif key == glfw.KEY_F5:
                        # Обновление списка файлов
                        self.refresh_images()
            
            glfw.set_key_callback(self.window, key_callback)
            
            # Устанавливаем callback для drag & drop
            def drop_callback(window, paths):
                """Обработка перетаскиваемых файлов"""
                if not paths:
                    return
                
                # Фильтруем только изображения
                extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                image_files = []
                
                for path in paths:
                    p = Path(path)
                    if p.is_file() and p.suffix.lower() in extensions:
                        image_files.append(path)
                    elif p.is_dir():
                        # Если перетащили папку - загружаем все изображения из неё
                        image_files.extend(find_images(path))
                
                if image_files:
                    print(f"\n📁 Перетащено {len(image_files)} файлов")
                    self.add_images_from_paths(image_files, replace=True)
                else:
                    print("\n⚠ Перетащенные файлы не являются изображениями")
            
            glfw.set_drop_callback(self.window, drop_callback)
            
            # Ждём готовности VR сессии
            log.info("")
            log.info("=" * 50)
            log.info("Ожидание готовности VR сессии...")
            log.info("  Убедитесь что VR шлем надет и активен!")
            log.info("  (Нажмите Q или ESC для выхода)")
            log.info("=" * 50)
            
            wait_start = time.time()
            wait_logged = False
            
            while not self.session_running and not self.should_quit:
                glfw.poll_events()
                self.poll_events()
                
                # Логируем каждые 2 секунды что ждём
                elapsed = time.time() - wait_start
                if elapsed > 2 and not wait_logged:
                    log.warning("  Всё ещё ждём... Проверьте:")
                    log.warning("    1. VR шлем включён и подключён")
                    log.warning("    2. SteamVR/Oculus запущен")
                    log.warning("    3. Шлем надет (датчик присутствия)")
                    wait_logged = True
                
                if elapsed > 30:
                    log.error("  Таймаут ожидания VR сессии (30 сек)")
                    self.should_quit = True
                    break
                    
                time.sleep(0.1)
            
            if self.should_quit:
                log.info("Выход до начала рендеринга")
                return
            
            # Главный цикл рендеринга
            log.info("")
            log.info(">>> РЕНДЕРИНГ ЗАПУЩЕН <<<")
            log.info(f"  Изображение: {self.images[self.current_index].name}")
            
            frame_count = 0
            last_log_time = time.time()
            
            while not self.should_quit and not glfw.window_should_close(self.window):
                glfw.poll_events()
                self.poll_events()
                
                # Проверяем появление новых файлов
                self.check_for_new_files()
                
                if self.session_running:
                    try:
                        self.render_frame()
                        # Опрос контроллеров после render_frame (нужен predicted_display_time)
                        self.poll_controller_actions()
                        frame_count += 1
                        
                        # Счётчик кадров (можно использовать для отладки)
                        current_time = time.time()
                        if current_time - last_log_time >= 5.0:
                            frame_count = 0
                            last_log_time = current_time
                            
                    except Exception as e:
                        log.error(f"Ошибка рендеринга: {e}")
                        import traceback
                        log.error(traceback.format_exc())
                        # Небольшая пауза чтобы не спамить ошибками
                        time.sleep(0.1)
                else:
                    # Сессия остановилась
                    log.warning("  Сессия не активна, ждём...")
                    time.sleep(0.1)
            
            log.info(">>> РЕНДЕРИНГ ЗАВЕРШЁН <<<")
            log.info(f"  should_quit: {self.should_quit}")
            log.info(f"  window_should_close: {glfw.window_should_close(self.window) if self.window else 'N/A'}")
                    
        except Exception as e:
            log.error(f"КРИТИЧЕСКАЯ ОШИБКА: {e}")
            import traceback
            log.error(traceback.format_exc())
            
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Очистка ресурсов"""
        log.info("")
        log.info("=" * 50)
        log.info("Очистка ресурсов...")
        
        try:
            # Удаляем текстуры
            for img in self.images:
                img.delete_textures()
            if self.distance_texture:
                glDeleteTextures(1, [self.distance_texture])
            if self.counter_texture:
                glDeleteTextures(1, [self.counter_texture])
            log.debug("  ✓ Текстуры удалены")
            
            # Удаляем OpenGL объекты
            if self.quad_vao:
                glDeleteVertexArrays(1, [self.quad_vao])
            if self.quad_vbo:
                glDeleteBuffers(1, [self.quad_vbo])
            if self.shader_program:
                glDeleteProgram(self.shader_program)
            log.debug("  ✓ OpenGL объекты удалены")
                
            # Удаляем framebuffers
            for fb_list in self.framebuffers:
                for fb in fb_list:
                    glDeleteFramebuffers(1, [fb])
            log.debug("  ✓ Framebuffers удалены")
            
            # Удаляем hand spaces
            for space in self.hand_spaces:
                if space is not None:
                    try:
                        xr.destroy_space(space)
                    except:
                        pass
            log.debug("  ✓ Hand spaces удалены")
            
            # Удаляем action set контроллеров
            if self.action_set:
                try:
                    xr.destroy_action_set(self.action_set)
                    log.debug("  ✓ Action set удалён")
                except:
                    pass
            
            # Удаляем OpenXR объекты
            for swapchain in self.swapchains:
                xr.destroy_swapchain(swapchain)
            log.debug("  ✓ Swapchains удалены")
                
            if self.view_space:
                xr.destroy_space(self.view_space)
            if self.local_space:
                xr.destroy_space(self.local_space)
            log.debug("  ✓ Пространства удалены")
            
            if self.session:
                xr.destroy_session(self.session)
                log.debug("  ✓ Сессия удалена")
            if self.instance:
                xr.destroy_instance(self.instance)
                log.debug("  ✓ Instance удалён")
                
            glfw.terminate()
            log.debug("  ✓ GLFW завершён")
            
        except Exception as e:
            log.error(f"Ошибка при очистке: {e}")
        
        log.info("=" * 50)
        log.info("Приложение завершено")
        log.info(f"Лог сохранён в: vr_viewer.log")
        log.info("=" * 50)


def find_images(directory: str) -> List[str]:
    """Поиск изображений в директории"""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    images = []
    
    path = Path(directory)
    if path.is_file():
        return [str(path)]
    
    for file in sorted(path.iterdir()):
        if file.is_file() and file.suffix.lower() in extensions:
            images.append(str(file))
    
    return images


def open_file_dialog(title: str = "Выберите стереоизображения", 
                     multiple: bool = True) -> List[str]:
    """
    Открывает диалог выбора файлов.
    
    Args:
        title: Заголовок диалога
        multiple: Разрешить выбор нескольких файлов
    
    Returns:
        Список путей к выбранным файлам
    """
    # Создаём скрытое окно tkinter
    root = tk.Tk()
    root.withdraw()  # Скрываем главное окно
    root.attributes('-topmost', True)  # Поверх других окон
    
    filetypes = [
        ("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
        ("JPEG", "*.jpg *.jpeg"),
        ("PNG", "*.png"),
        ("BMP", "*.bmp"),
        ("TIFF", "*.tiff *.tif"),
        ("Все файлы", "*.*"),
    ]
    
    if multiple:
        files = filedialog.askopenfilenames(
            title=title,
            filetypes=filetypes,
        )
        result = list(files) if files else []
    else:
        file = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes,
        )
        result = [file] if file else []
    
    root.destroy()
    return result


def open_folder_dialog(title: str = "Выберите папку с изображениями") -> str:
    """
    Открывает диалог выбора папки.
    
    Args:
        title: Заголовок диалога
    
    Returns:
        Путь к выбранной папке или пустая строка
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    folder = filedialog.askdirectory(title=title)
    
    root.destroy()
    return folder if folder else ""


def main():
    parser = argparse.ArgumentParser(
        description='VR Stereo Image Viewer - просмотрщик стереоизображений для VR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s                              # Открыть диалог выбора файлов
  %(prog)s image.jpg                    # Открыть одно SBS изображение
  %(prog)s *.jpg                        # Открыть все JPG файлы
  %(prog)s ./stereo_photos/             # Открыть все изображения в папке
  %(prog)s --separate left.jpg right.jpg  # Открыть пару изображений

Во время работы нажмите O для открытия новых файлов или F для выбора папки.
        """
    )
    
    parser.add_argument(
        'images',
        nargs='*',  # Теперь аргументы опциональны
        help='Путь к изображениям или директории (опционально)'
    )
    
    parser.add_argument(
        '--separate', '-s',
        action='store_true',
        help='Режим раздельных файлов (left/right вместо SBS)'
    )
    
    parser.add_argument(
        '--distance', '-d',
        type=float,
        default=2.0,
        help='Начальное расстояние до изображения в метрах (по умолчанию: 2.0)'
    )
    
    parser.add_argument(
        '--scale',
        type=float,
        default=1.5,
        help='Начальный масштаб изображения (по умолчанию: 1.5)'
    )
    
    args = parser.parse_args()
    
    # Собираем список файлов
    all_images = []
    
    if args.images:
        # Если указаны файлы в командной строке
        for path in args.images:
            if os.path.isdir(path):
                all_images.extend(find_images(path))
            elif os.path.isfile(path):
                all_images.append(path)
            else:
                # Возможно glob pattern
                from glob import glob
                all_images.extend(glob(path))
    else:
        # Если файлы не указаны - открываем диалог выбора
        print("=== VR Stereo Image Viewer ===")
        print("\nВыберите способ открытия изображений:")
        print("  1. Выбрать файлы")
        print("  2. Выбрать папку")
        print("  3. Выход")
        print()
        
        choice = input("Ваш выбор (1/2/3): ").strip()
        
        if choice == '1':
            all_images = open_file_dialog(
                title="Выберите стереоизображения (SBS)",
                multiple=True
            )
        elif choice == '2':
            folder = open_folder_dialog(
                title="Выберите папку с изображениями"
            )
            if folder:
                all_images = find_images(folder)
        elif choice == '3':
            print("Выход.")
            sys.exit(0)
        else:
            # По умолчанию открываем диалог выбора файлов
            all_images = open_file_dialog(
                title="Выберите стереоизображения (SBS)",
                multiple=True
            )
    
    if not all_images:
        print("Изображения не выбраны!")
        sys.exit(1)
    
    # Создаём и запускаем просмотрщик
    viewer = VRStereoViewer(
        image_paths=all_images,
        sbs_mode=not args.separate
    )
    viewer.image_distance = args.distance
    viewer.image_scale = args.scale
    
    viewer.run()


if __name__ == '__main__':
    main()

