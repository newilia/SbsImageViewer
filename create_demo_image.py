"""
Скрипт для создания демонстрационного SBS стереоизображения
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os


def create_demo_stereo_image(output_path: str = "demo_stereo.png"):
    """
    Создаёт демонстрационное SBS стереоизображение с геометрическими фигурами.
    
    Изображение создаётся с небольшим смещением между левым и правым видом,
    что создаёт эффект глубины при просмотре в VR.
    """
    
    # Размеры одного глаза
    width = 1920
    height = 1080
    
    # Создаём изображения для каждого глаза
    left_img = Image.new('RGB', (width, height), (20, 25, 35))
    right_img = Image.new('RGB', (width, height), (20, 25, 35))
    
    left_draw = ImageDraw.Draw(left_img)
    right_draw = ImageDraw.Draw(right_img)
    
    # Параметры стереоэффекта
    # Положительное смещение = объект ближе к зрителю
    # Отрицательное смещение = объект дальше
    
    objects = [
        # (центр_x, центр_y, размер, цвет, смещение, тип)
        (width//2, height//2, 150, (70, 130, 200), 30, 'circle'),  # Центральный круг - ближе
        (width//4, height//3, 100, (200, 100, 100), -20, 'circle'),  # Левый круг - дальше
        (3*width//4, height//3, 100, (100, 200, 100), 15, 'circle'),  # Правый круг - ближе
        (width//3, 2*height//3, 120, (200, 180, 50), 0, 'square'),  # Квадрат - на плоскости
        (2*width//3, 2*height//3, 80, (180, 80, 180), 40, 'triangle'),  # Треугольник - очень близко
    ]
    
    # Рисуем фоновую сетку (дальний план)
    grid_offset = -15  # Сетка на заднем плане
    grid_color = (40, 50, 60)
    
    for x in range(0, width, 100):
        left_draw.line([(x - grid_offset//2, 0), (x - grid_offset//2, height)], fill=grid_color, width=1)
        right_draw.line([(x + grid_offset//2, 0), (x + grid_offset//2, height)], fill=grid_color, width=1)
    
    for y in range(0, height, 100):
        left_draw.line([(0, y), (width, y)], fill=grid_color, width=1)
        right_draw.line([(0, y), (width, y)], fill=grid_color, width=1)
    
    # Рисуем объекты
    for cx, cy, size, color, stereo_offset, obj_type in objects:
        # Смещение для левого и правого глаза
        left_x = cx - stereo_offset // 2
        right_x = cx + stereo_offset // 2
        
        if obj_type == 'circle':
            left_draw.ellipse(
                [left_x - size, cy - size, left_x + size, cy + size],
                fill=color, outline=(255, 255, 255), width=3
            )
            right_draw.ellipse(
                [right_x - size, cy - size, right_x + size, cy + size],
                fill=color, outline=(255, 255, 255), width=3
            )
            
        elif obj_type == 'square':
            left_draw.rectangle(
                [left_x - size, cy - size, left_x + size, cy + size],
                fill=color, outline=(255, 255, 255), width=3
            )
            right_draw.rectangle(
                [right_x - size, cy - size, right_x + size, cy + size],
                fill=color, outline=(255, 255, 255), width=3
            )
            
        elif obj_type == 'triangle':
            left_points = [
                (left_x, cy - size),
                (left_x - size, cy + size),
                (left_x + size, cy + size)
            ]
            right_points = [
                (right_x, cy - size),
                (right_x - size, cy + size),
                (right_x + size, cy + size)
            ]
            left_draw.polygon(left_points, fill=color, outline=(255, 255, 255))
            right_draw.polygon(right_points, fill=color, outline=(255, 255, 255))
    
    # Добавляем текст
    try:
        font = ImageFont.truetype("arial.ttf", 48)
        small_font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    text = "VR Stereo Demo"
    text_offset = 25
    
    # Текст на переднем плане (большое смещение)
    left_draw.text((width//2 - 180 - text_offset//2, 50), text, fill=(255, 255, 255), font=font)
    right_draw.text((width//2 - 180 + text_offset//2, 50), text, fill=(255, 255, 255), font=font)
    
    # Инструкции внизу
    instructions = "Наденьте VR-шлем для просмотра стереоэффекта"
    left_draw.text((width//2 - 280, height - 60), instructions, fill=(150, 150, 150), font=small_font)
    right_draw.text((width//2 - 280, height - 60), instructions, fill=(150, 150, 150), font=small_font)
    
    # Объединяем в SBS формат
    sbs_img = Image.new('RGB', (width * 2, height))
    sbs_img.paste(left_img, (0, 0))
    sbs_img.paste(right_img, (width, 0))
    
    # Сохраняем
    sbs_img.save(output_path, quality=95)
    print(f"Демо изображение сохранено: {output_path}")
    print(f"Размер: {width * 2}x{height} (SBS формат)")
    
    return output_path


def create_depth_gradient_image(output_path: str = "demo_depth.png"):
    """
    Создаёт стереоизображение с градиентом глубины.
    Демонстрирует, как разные смещения создают ощущение глубины.
    """
    
    width = 1920
    height = 1080
    
    left_img = Image.new('RGB', (width, height), (15, 20, 30))
    right_img = Image.new('RGB', (width, height), (15, 20, 30))
    
    left_draw = ImageDraw.Draw(left_img)
    right_draw = ImageDraw.Draw(right_img)
    
    # Создаём ряды кругов с разной глубиной
    num_rows = 5
    num_cols = 8
    
    for row in range(num_rows):
        # Глубина увеличивается от верхнего ряда к нижнему
        depth = row - num_rows // 2  # От -2 до +2
        stereo_offset = depth * 15  # Смещение пропорционально глубине
        
        # Цвет зависит от глубины
        brightness = 100 + row * 30
        color = (brightness, brightness // 2, 255 - row * 30)
        
        y = height // (num_rows + 1) * (row + 1)
        
        for col in range(num_cols):
            x = width // (num_cols + 1) * (col + 1)
            size = 30 + row * 5
            
            left_x = x - stereo_offset // 2
            right_x = x + stereo_offset // 2
            
            left_draw.ellipse(
                [left_x - size, y - size, left_x + size, y + size],
                fill=color, outline=(255, 255, 255), width=2
            )
            right_draw.ellipse(
                [right_x - size, y - size, right_x + size, y + size],
                fill=color, outline=(255, 255, 255), width=2
            )
    
    # Добавляем метки глубины
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    depths = ["Далеко", "Дальше", "Плоскость", "Ближе", "Близко"]
    for row, label in enumerate(depths):
        y = height // (num_rows + 1) * (row + 1)
        left_draw.text((20, y - 10), label, fill=(200, 200, 200), font=font)
        right_draw.text((20, y - 10), label, fill=(200, 200, 200), font=font)
    
    # Объединяем
    sbs_img = Image.new('RGB', (width * 2, height))
    sbs_img.paste(left_img, (0, 0))
    sbs_img.paste(right_img, (width, 0))
    
    sbs_img.save(output_path, quality=95)
    print(f"Демо изображение с глубиной сохранено: {output_path}")
    
    return output_path


if __name__ == '__main__':
    # Создаём демонстрационные изображения
    create_demo_stereo_image()
    create_depth_gradient_image()
    
    print("\nДля просмотра в VR запустите:")
    print("  python sbs_viewer.py demo_stereo.png demo_depth.png")

