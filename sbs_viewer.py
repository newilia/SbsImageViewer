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
        img = Image.open(image_path).convert('RGBA')
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
        
        # Параметры отображения
        self.quad_distance = 2.0  # Расстояние до прямоугольника в метрах
        self.quad_scale = 1.0  # Масштаб прямоугольника (угловой размер)
        self.base_size = 1.0  # Базовый физический размер при расстоянии 1м
        self.distance_texture: Optional[int] = None
        self.distance_aspect: float = 1.0
        self.head_height: Optional[float] = None  # Высота головы (центр между глазами)
        
    def load_images(self):
        """Загрузка всех изображений"""
        # Если передан один файл - загружаем все изображения из его папки
        if len(self.image_paths) == 1 and os.path.isfile(self.image_paths[0]):
            single_file = self.image_paths[0]
            folder = os.path.dirname(single_file)
            if folder:
                log.info(f"Загрузка всех изображений из папки: {folder}")
                all_files = find_images(folder)
                if all_files:
                    self.image_paths = all_files
                    # Находим индекс исходного файла
                    try:
                        start_index = [os.path.normpath(p) for p in all_files].index(os.path.normpath(single_file))
                        self.current_index = start_index
                    except ValueError:
                        self.current_index = 0
        
        log.info(f"Загрузка {len(self.image_paths)} изображений...")
        
        for path in self.image_paths:
            try:
                log.debug(f"  Обработка: {path}")
                if self.sbs_mode:
                    img = StereoImage.from_sbs(path)
                else:
                    # Для режима отдельных файлов ожидаем пары _left/_right
                    if '_left' in path.lower():
                        right_path = path.lower().replace('_left', '_right')
                        for orig_path in self.image_paths:
                            if orig_path.lower() == right_path:
                                img = StereoImage.from_separate_files(path, orig_path)
                                break
                        else:
                            continue
                    elif '_right' in path.lower():
                        continue  # Пропускаем, уже обработано с _left
                    else:
                        # Загружаем как SBS по умолчанию
                        img = StereoImage.from_sbs(path)
                
                self.images.append(img)
                log.info(f"  ✓ Загружено: {img.name} ({img.left.shape[1]}x{img.left.shape[0]})")
            except Exception as e:
                log.error(f"  ✗ Ошибка загрузки {path}: {e}")
        
        log.info(f"Всего загружено {len(self.images)} стереопар")
        if self.images:
            log.info(f"Текущее изображение: {self.images[self.current_index].name}")
    
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
        """Создание текстур для всех загруженных изображений"""
        for img in self.images:
            img.create_textures()
        self.update_distance_texture()
    
    def update_distance_texture(self):
        """Обновление текстуры с расстоянием"""
        from PIL import ImageDraw, ImageFont
        
        # Удаляем старую текстуру
        if self.distance_texture:
            glDeleteTextures(1, [self.distance_texture])
        
        # Текст с расстоянием
        text = f"{self.quad_distance:.1f} м"
        
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
        # Позиция: перед пользователем на расстоянии quad_distance
        # Y = фиксированная высота головы (одинаковая для обоих глаз!)
        # При первом кадре запоминаем высоту
        if self.head_height is None:
            self.head_height = pose.position.y
        eye_height = self.head_height
        quad_pos = xr.Vector3f(0, eye_height, -self.quad_distance)
        quad_rot = xr.Quaternionf(0, 0, 0, 1)  # Без вращения
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
        
        # Позиция под изображением
        label_y = eye_height - (physical_scale * 0.5) - 0.02 * label_scale_factor
        
        # 1. Название файла
        if current_image.name_texture:
            text_height = 0.03 * label_scale_factor  # Угловой размер ~1.7°
            text_width = text_height * current_image.name_aspect
            
            label_y -= text_height
            text_pos = xr.Vector3f(0, label_y, -self.quad_distance + 0.01)
            text_scale = xr.Vector3f(text_width, text_height, 1)
            text_model = Matrix4x4f.create_translation_rotation_scale(text_pos, quad_rot, text_scale)
            
            text_mvp = vp @ text_model
            glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, text_mvp.as_numpy())
            
            glBindTexture(GL_TEXTURE_2D, current_image.name_texture)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            
            label_y -= 0.005 * label_scale_factor  # Отступ между названием и расстоянием
        
        # 2. Расстояние
        if self.distance_texture:
            dist_height = 0.02 * label_scale_factor  # Угловой размер ~1.1°
            dist_width = dist_height * self.distance_aspect
            
            label_y -= dist_height
            dist_pos = xr.Vector3f(0, label_y, -self.quad_distance + 0.01)
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
                
    def next_image(self):
        """Переход к следующему изображению"""
        if self.images and len(self.images) > 1:
            self.current_index = (self.current_index + 1) % len(self.images)
            log.info(f"Изображение [{self.current_index + 1}/{len(self.images)}]: {self.images[self.current_index].name}")
            
    def prev_image(self):
        """Переход к предыдущему изображению"""
        if self.images and len(self.images) > 1:
            self.current_index = (self.current_index - 1) % len(self.images)
            log.info(f"Изображение [{self.current_index + 1}/{len(self.images)}]: {self.images[self.current_index].name}")
    
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
                log.info(f"Изображение [{self.current_index + 1}/{len(self.images)}]: {self.images[self.current_index].name}")
            else:
                log.info("Все изображения удалены")
                
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
        
        self.image_paths = new_paths
        log.info(f"🔄 Обновлено: найдено {len(new_paths)} файлов в {folder}")
        
        # Загружаем изображения
        for path in self.image_paths:
            try:
                if self.sbs_mode:
                    img = StereoImage.from_sbs(path)
                else:
                    if '_left' in path.lower():
                        right_path = path.lower().replace('_left', '_right')
                        for orig_path in self.image_paths:
                            if orig_path.lower() == right_path:
                                img = StereoImage.from_separate_files(path, orig_path)
                                break
                        else:
                            continue
                    elif '_right' in path.lower():
                        continue
                    else:
                        img = StereoImage.from_sbs(path)
                
                img.create_textures()
                self.images.append(img)
            except Exception as e:
                log.debug(f"Ошибка загрузки {path}: {e}")
        
        # Восстанавливаем позицию на том же изображении если возможно
        self.current_index = 0
        if current_name:
            for i, img in enumerate(self.images):
                if img.name == current_name:
                    self.current_index = i
                    break
        
        if self.images:
            log.info(f"Изображение [{self.current_index + 1}/{len(self.images)}]: {self.images[self.current_index].name}")
    
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
            log.info(f"  ✓ Текстуры созданы ({len(self.images)} изображений)")
            
            log.info("=" * 50)
            log.info("ИНИЦИАЛИЗАЦИЯ ЗАВЕРШЕНА")
            log.info("=" * 50)
            log.info("")
            log.info("Управление:")
            log.info("  Перетащите файлы на окно для загрузки")
            log.info("  O - открыть файлы | F - открыть папку")
            log.info("  ←/→ или A/D - переключение изображений")
            log.info("  +/- - масштаб | W/S - расстояние")
            log.info("  ESC/Q - выход")
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
                        log.info(f"  Угловой размер: {self.quad_scale:.2f}")
                    elif key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT or key == glfw.KEY_A:
                        self.quad_scale = max(0.1, self.quad_scale / 1.1)
                        log.info(f"  Угловой размер: {self.quad_scale:.2f}")
                    elif key == glfw.KEY_S:
                        # Логарифмическое увеличение расстояния (дальше)
                        self.quad_distance = min(50.0, self.quad_distance * 1.15)
                        self.update_distance_texture()
                        log.info(f"  Расстояние: {self.quad_distance:.1f} м")
                    elif key == glfw.KEY_W:
                        # Логарифмическое уменьшение расстояния (ближе)
                        self.quad_distance = max(0.3, self.quad_distance / 1.15)
                        self.update_distance_texture()
                        log.info(f"  Расстояние: {self.quad_distance:.1f} м")
                    elif key == glfw.KEY_R:
                        # Сброс высоты головы (перецентровка)
                        self.head_height = None
                        log.info("  Высота головы сброшена")
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
                
                if self.session_running:
                    try:
                        self.render_frame()
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

