"""Проверка доступных функций pyopenxr"""
import xr

# Ищем функции связанные с graphics requirements
funcs = [x for x in dir(xr) if 'requirement' in x.lower() or 'graphic' in x.lower()]
print("Функции с 'requirement' или 'graphic':")
for f in funcs:
    print(f"  {f}")

# Ищем всё связанное с OpenGL
gl_funcs = [x for x in dir(xr) if 'gl' in x.lower()]
print("\nФункции с 'gl':")
for f in gl_funcs:
    print(f"  {f}")

# Версия pyopenxr
print(f"\npyopenxr version: {xr.__version__ if hasattr(xr, '__version__') else 'unknown'}")

