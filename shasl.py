#!/usr/bin/env python3
"""
2D Модель прогрева шашлыка с вращением над углями
=================================================

Физика:
- 2D сечение цилиндра (r, φ) - радиально-угловая сетка
- Вращение шампура с заданной скоростью
- Неравномерный нагрев: снизу - огонь (излучение + конвекция),
                        сверху - воздух (только конвекция, охлаждение)
- Радиальная и тангенциальная теплопроводность

Уравнение теплопроводности в цилиндрических координатах:
ρc * ∂T/∂t = (1/r) * ∂/∂r(k*r * ∂T/∂r) + (1/r²) * ∂/∂φ(k * ∂T/∂φ)

Граничные условия:
- Снизу (угли): Q = w(φ,t) * [ε*σ*(T_fire⁴ - T_s⁴) + h_hot*(T_fire - T_s)]
- Сверху (воздух): Q = (1-w(φ,t)) * h_cold*(T_air - T_s)
- w(φ,t) - весовая функция положения сектора относительно углей
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
import json
import pickle
import time
from datetime import datetime
from numba import jit, prange
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Физические константы
STEFAN_BOLTZMANN = 5.67e-8  # Вт/(м²·К⁴)


# =============================================================================
# Конфигурация
# =============================================================================

@dataclass
class MeatProperties:
    """Теплофизические свойства мяса"""
    name: str = "Говядина"
    k: float = 0.45  # Теплопроводность, Вт/(м·К)
    rho: float = 1050.0  # Плотность, кг/м³
    c: float = 3500.0  # Удельная теплоёмкость, Дж/(кг·К)

    @property
    def alpha(self) -> float:
        """Температуропроводность, м²/с"""
        return self.k / (self.rho * self.c)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CylinderGeometry:
    """Геометрия цилиндра (шашлыка)"""
    radius: float = 0.015  # Радиус, м (15 мм)
    length: float = 0.03  # Длина куска, м (30 мм)
    n_radial: int = 25  # Количество ячеек по радиусу
    n_angular: int = 60  # Количество секторов по углу (360°/60 = 6°)

    def to_dict(self) -> dict:
        return asdict(self)


class RotationStrategy:
    """Перечисление стратегий вращения"""
    CONTINUOUS = "continuous"  # Постоянное вращение
    FLIP_90 = "flip_90"  # Переворот на 90° через интервал
    FLIP_180 = "flip_180"  # Переворот на 180° через интервал
    STATIC = "static"  # Без вращения (статика)


@dataclass
class RotationConfig:
    """Параметры вращения шампура"""
    strategy: str = RotationStrategy.CONTINUOUS  # Стратегия вращения
    rotation_period: float = 10.0  # Период одного оборота (для CONTINUOUS), с
    flip_interval: float = 30.0  # Интервал между переворотами (для FLIP), с
    auto_rotate: bool = True  # Автоматическое вращение

    @property
    def omega(self) -> float:
        """Угловая скорость, рад/с (только для CONTINUOUS)"""
        if self.strategy == RotationStrategy.CONTINUOUS and self.rotation_period > 0:
            return 2 * np.pi / self.rotation_period
        return 0.0

    @property
    def rpm(self) -> float:
        """Обороты в минуту (только для CONTINUOUS)"""
        if self.strategy == RotationStrategy.CONTINUOUS and self.rotation_period > 0:
            return 60.0 / self.rotation_period
        return 0.0

    def get_rotation_angle(self, t: float) -> float:
        """
        Вычисление угла поворота в момент времени t

        Returns:
            Угол в радианах
        """
        if not self.auto_rotate:
            return 0.0

        if self.strategy == RotationStrategy.CONTINUOUS:
            # Постоянное плавное вращение
            return self.omega * t

        elif self.strategy == RotationStrategy.FLIP_90:
            # Дискретные перевороты на 90°
            n_flips = int(t / self.flip_interval)
            return n_flips * (np.pi / 2)

        elif self.strategy == RotationStrategy.FLIP_180:
            # Дискретные перевороты на 180°
            n_flips = int(t / self.flip_interval)
            return n_flips * np.pi

        elif self.strategy == RotationStrategy.STATIC:
            # Без вращения
            return 0.0

        return 0.0

    def get_strategy_description(self) -> str:
        """Описание стратегии для вывода"""
        if self.strategy == RotationStrategy.CONTINUOUS:
            return f"Постоянное вращение (период {self.rotation_period:.1f}с, {self.rpm:.1f} об/мин)"
        elif self.strategy == RotationStrategy.FLIP_90:
            return f"Переворот на 90° каждые {self.flip_interval:.0f}с"
        elif self.strategy == RotationStrategy.FLIP_180:
            return f"Переворот на 180° каждые {self.flip_interval:.0f}с"
        elif self.strategy == RotationStrategy.STATIC:
            return "Без вращения (статика)"
        return "Неизвестная стратегия"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HeatSourceConfig:
    """Конфигурация источников тепла"""
    # Угли (снизу)
    T_fire: float = 573.0  # Температура углей, К (300°C)
    h_conv_hot: float = 35.0  # Конвекция от углей, Вт/(м²·К)
    epsilon: float = 0.95  # Степень черноты мяса

    # Воздух (сверху и с боков)
    T_air: float = 313.0  # Температура воздуха над мангалом, К (40°C)
    h_conv_cold: float = 10.0  # Конвекция воздуха, Вт/(м²·К)

    # Геометрия зоны нагрева
    fire_angle_width: float = 120.0  # Угловая ширина зоны огня, градусы
    fire_direction: float = 270.0  # Направление на угли (270° = снизу)
    transition_width: float = 30.0  # Ширина переходной зоны, градусы

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CookingConditions:
    """Условия приготовления"""
    T_initial: float = 278.0  # Начальная температура мяса, К (5°C)

    # Критерии готовности
    T_ready: float = 343.0  # Температура готовности в центре, К (70°C)
    t_hold: float = 60.0  # Время удержания температуры, с

    # Параметры симуляции
    t_max: float = 900.0  # Максимальное время расчёта, с (15 минут)
    save_interval: float = 1.0  # Интервал сохранения данных, с

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimulationConfig2D:
    """Полная конфигурация 2D симуляции"""
    meat: MeatProperties = field(default_factory=MeatProperties)
    geometry: CylinderGeometry = field(default_factory=CylinderGeometry)
    rotation: RotationConfig = field(default_factory=RotationConfig)
    heat_source: HeatSourceConfig = field(default_factory=HeatSourceConfig)
    cooking: CookingConditions = field(default_factory=CookingConditions)

    def to_dict(self) -> dict:
        return {
            'meat': self.meat.to_dict(),
            'geometry': self.geometry.to_dict(),
            'rotation': self.rotation.to_dict(),
            'heat_source': self.heat_source.to_dict(),
            'cooking': self.cooking.to_dict()
        }

    def print_summary(self):
        """Вывод сводки параметров"""
        print("\n" + "=" * 65)
        print("  2D МОДЕЛЬ ПРОГРЕВА ШАШЛЫКА С ВРАЩЕНИЕМ НАД УГЛЯМИ")
        print("=" * 65)

        print(f"\nГеометрия ({self.meat.name}):")
        print(f"   Радиус: {self.geometry.radius * 1000:.1f} мм")
        print(f"   Длина куска: {self.geometry.length * 1000:.1f} мм")
        print(f"   Сетка: {self.geometry.n_radial} × {self.geometry.n_angular} "
              f"(радиус × угол)")
        print(f"   Всего ячеек: {self.geometry.n_radial * self.geometry.n_angular}")

        print(f"\nТеплофизические свойства:")
        print(f"   k = {self.meat.k:.3f} Вт/(м·К)")
        print(f"   ρ = {self.meat.rho:.0f} кг/м³")
        print(f"   c = {self.meat.c:.0f} Дж/(кг·К)")
        print(f"   α = {self.meat.alpha:.2e} м²/с")
        print(f"   Начальная температура мяса = {self.cooking.T_initial - 273:.0f} °C")

        print(f"\nВращение:")
        print(f"   Стратегия: {self.rotation.get_strategy_description()}")
        if self.rotation.strategy == RotationStrategy.CONTINUOUS:
            print(f"   ω = {self.rotation.omega:.3f} рад/с")

        print(f"\nИсточники тепла:")
        print(f"   Угли (снизу):")
        print(f"      T_fire = {self.heat_source.T_fire - 273:.0f}°C")
        print(f"      h_hot = {self.heat_source.h_conv_hot:.0f} Вт/(м²·К)")
        print(f"      ε = {self.heat_source.epsilon:.2f}")
        print(f"      Угол зоны огня: {self.heat_source.fire_angle_width:.0f}°")
        print(f"   Воздух (сверху):")
        print(f"      T_air = {self.heat_source.T_air - 273:.0f}°C")
        print(f"      h_cold = {self.heat_source.h_conv_cold:.0f} Вт/(м²·К)")

        print(f"\nКритерии готовности:")
        print(f"   T центра ≥ {self.cooking.T_ready - 273:.0f}°C")
        print(f"   Удержание: {self.cooking.t_hold:.0f} с")
        print("=" * 65)


# =============================================================================
# Хранилище данных
# =============================================================================

@dataclass
class TimeSnapshot2D:
    """Снимок состояния в момент времени (2D)"""
    time: float
    rotation_angle: float  # Текущий угол поворота, рад
    T_field: np.ndarray  # 2D поле температуры [n_radial, n_angular]
    T_center: float  # Средняя температура в центре
    T_surface_min: float  # Минимальная температура поверхности
    T_surface_max: float  # Максимальная температура поверхности
    T_surface_avg: float  # Средняя температура поверхности
    Q_total_in: float  # Суммарный входящий поток
    is_ready: bool


@dataclass
class SimulationResults2D:
    """Результаты 2D симуляции"""
    config: SimulationConfig2D
    snapshots: List[TimeSnapshot2D] = field(default_factory=list)

    total_time: float = 0.0
    cooking_time: float = 0.0
    is_cooked: bool = False
    computation_time: float = 0.0

    # Массивы для быстрого доступа
    times: np.ndarray = None
    rotation_angles: np.ndarray = None
    T_center_history: np.ndarray = None
    T_surface_max_history: np.ndarray = None
    T_surface_min_history: np.ndarray = None
    radii: np.ndarray = None
    angles: np.ndarray = None

    def finalize(self):
        """Преобразование в массивы"""
        if self.snapshots:
            self.times = np.array([s.time for s in self.snapshots])
            self.rotation_angles = np.array([s.rotation_angle for s in self.snapshots])
            self.T_center_history = np.array([s.T_center for s in self.snapshots])
            self.T_surface_max_history = np.array([s.T_surface_max for s in self.snapshots])
            self.T_surface_min_history = np.array([s.T_surface_min for s in self.snapshots])

    def save(self, filename: str):
        """Сохранение результатов"""
        self.finalize()

        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(self, f)

        # JSON сводка
        json_data = {
            'config': self.config.to_dict(),
            'summary': {
                'total_time': self.total_time,
                'cooking_time': self.cooking_time,
                'is_cooked': self.is_cooked,
                'computation_time': self.computation_time,
                'n_snapshots': len(self.snapshots)
            }
        }

        with open(filename + '.json', 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Данные сохранены: {filename}.pkl, {filename}.json")

    @classmethod
    def load(cls, filename: str) -> 'SimulationResults2D':
        with open(filename + '.pkl', 'rb') as f:
            return pickle.load(f)


# =============================================================================
# Numba-оптимизированное ядро расчёта
# =============================================================================

@jit(nopython=True, parallel=True)
def compute_weight_function(phi_sector: float, phi_rotation: float,
                            fire_direction: float, fire_width: float,
                            transition_width: float) -> float:
    """
    Вычисление весовой функции w(φ, t) для сектора

    w = 1: сектор полностью над углями (огонь)
    w = 0: сектор полностью над воздухом (охлаждение)
    0 < w < 1: переходная зона

    phi_sector: угол сектора в системе координат мяса
    phi_rotation: текущий угол поворота шампура
    fire_direction: направление на угли (обычно 270° = снизу)
    fire_width: угловая ширина зоны огня
    transition_width: ширина переходной зоны
    """
    # Абсолютный угол сектора в неподвижной системе координат
    phi_abs = phi_sector + phi_rotation

    # Нормализация угла к [0, 2π]
    phi_abs = phi_abs % (2 * np.pi)

    # Угол до направления на огонь
    fire_dir_rad = fire_direction * np.pi / 180.0
    delta_phi = phi_abs - fire_dir_rad

    # Нормализация к [-π, π]
    while delta_phi > np.pi:
        delta_phi -= 2 * np.pi
    while delta_phi < -np.pi:
        delta_phi += 2 * np.pi

    delta_phi = abs(delta_phi)

    # Половина ширины зоны огня в радианах
    half_fire = (fire_width / 2) * np.pi / 180.0
    trans_rad = transition_width * np.pi / 180.0

    if delta_phi <= half_fire:
        # Полностью в зоне огня
        return 1.0
    elif delta_phi >= half_fire + trans_rad:
        # Полностью вне зоны огня
        return 0.0
    else:
        # Переходная зона - плавный косинусный переход
        t = (delta_phi - half_fire) / trans_rad
        return 0.5 * (1.0 + np.cos(np.pi * t))



@jit(nopython=True, parallel=True)
def compute_rhs_2d(T, n_r, n_phi, dr, dphi, r_centers,
                        A_radial_in, A_radial_out, A_tangential, V,
                        k, rho, c, phi_rotation,
                        T_fire, T_air, h_hot, h_cold, epsilon,
                        fire_direction, fire_width, transition_width,
                        sigma=5.67e-8):

    dTdt = np.zeros((n_r, n_phi))

    # Параллелизуем по УГЛАМ, а не по радиусам!
    for j in prange(n_phi):      # ВНЕШНИЙ цикл по углам (параллельный)
        for i in range(n_r):     # Внутренний цикл по радиусам (последовательный)
            Q_total = 0.0

            # Индексы соседей по углу
            j_prev = (j - 1) % n_phi
            j_next = (j + 1) % n_phi

            # 1. РАДИАЛЬНЫЙ ТЕПЛООБМЕН (от центра к поверхности)
            if i > 0:
                Q_in = k * A_radial_in[i, j] * (T[i-1, j] - T[i, j]) / dr
                Q_total += Q_in

            if i < n_r - 1:
                Q_out = k * A_radial_out[i, j] * (T[i, j] - T[i+1, j]) / dr
                Q_total -= Q_out
            else:
              # ГРАНИЧНОЕ УСЛОВИЕ НА ПОВЕРХНОСТИ
                T_s = T[i, j]
                phi_sector = j * dphi

                # Весовая функция: где находится этот сектор?
                w = compute_weight_function(
                    phi_sector, phi_rotation,
                    fire_direction, fire_width, transition_width
                )

                A_ext = A_radial_out[i, j]

                # Поток от огня (взвешенный)
                if w > 0:
                    Q_rad = epsilon * sigma * A_ext * (T_fire**4 - T_s**4)
                    Q_conv_hot = h_hot * A_ext * (T_fire - T_s)
                    Q_fire = w * (Q_rad + Q_conv_hot)
                else:
                    Q_fire = 0.0

                # Поток от воздуха (охлаждение)
                Q_air = (1 - w) * h_cold * A_ext * (T_air - T_s)

                Q_total += Q_fire + Q_air


            # 2. ТАНГЕНЦИАЛЬНЫЙ ТЕПЛООБМЕН
            if n_phi > 1:
                r_i = r_centers[i]
                if r_i > 0:
                    arc_length = r_i * dphi
                    Q_tang_prev = k * A_tangential[i] * (T[i, j_prev] - T[i, j]) / arc_length
                    Q_tang_next = k * A_tangential[i] * (T[i, j] - T[i, j_next]) / arc_length
                    Q_total += Q_tang_prev - Q_tang_next

            # dT/dt
            C_ij = rho * c * V[i, j]
            if C_ij > 0:
                dTdt[i, j] = Q_total / C_ij

    return dTdt



class CylindricalHeatTransfer2D:
    """
    2D модель теплопередачи в цилиндре с вращением
    """

    def __init__(self, config: SimulationConfig2D):
        self.config = config
        self.meat = config.meat
        self.geom = config.geometry
        self.rot = config.rotation
        self.heat = config.heat_source
        self.cook = config.cooking

        self._build_mesh()
        self._init_temperature()

        self.results = SimulationResults2D(config=config)
        self.results.radii = self.r_centers.copy()
        self.results.angles = self.phi_centers.copy()

    def _build_mesh(self):
        """Построение 2D сетки (r, φ)"""
        R = self.geom.radius
        n_r = self.geom.n_radial
        n_phi = self.geom.n_angular
        L = self.geom.length

        self.n_r = n_r
        self.n_phi = n_phi
        self.dr = R / n_r
        self.dphi = 2 * np.pi / n_phi

        # Центры ячеек
        self.r_centers = np.array([(i + 0.5) * self.dr for i in range(n_r)])
        self.phi_centers = np.array([j * self.dphi for j in range(n_phi)])

        # Геометрические параметры для каждого сектора
        self.V = np.zeros((n_r, n_phi))  # Объём сектора
        self.A_radial_in = np.zeros((n_r, n_phi))  # Площадь внутренней радиальной грани
        self.A_radial_out = np.zeros((n_r, n_phi))  # Площадь внешней радиальной грани
        self.A_tangential = np.zeros(n_r)  # Площадь тангенциальной грани (одинакова для всех j)

        for i in range(n_r):
            r_in = i * self.dr
            r_out = (i + 1) * self.dr

            # Объём сектора: V = (1/n_phi) * π * (r_out² - r_in²) * L
            # Это "кусок пиццы"
            sector_area = (self.dphi / (2 * np.pi)) * np.pi * (r_out**2 - r_in**2)
            V_sector = sector_area * L

            # Площадь внутренней радиальной грани (дуга × длина)
            # A_in = r_in * dphi * L
            A_in = r_in * self.dphi * L

            # Площадь внешней радиальной грани
            A_out = r_out * self.dphi * L

            # Площадь тангенциальной грани (dr × L)
            A_tang = self.dr * L

            for j in range(n_phi):
                self.V[i, j] = V_sector
                self.A_radial_in[i, j] = A_in
                self.A_radial_out[i, j] = A_out

            self.A_tangential[i] = A_tang

        print(f"   Сетка построена: {n_r}×{n_phi} = {n_r * n_phi} ячеек")

    def _init_temperature(self):
        """Инициализация температурного поля"""
        self.T = np.full((self.n_r, self.n_phi), self.cook.T_initial)
        self.phi_rotation = 0.0  # Текущий угол поворота

    def get_rotation_angle(self, t: float) -> float:
        """Угол поворота шампура в момент времени t (использует стратегию из config)"""
        return self.rot.get_rotation_angle(t)

    def compute_rhs(self, T: np.ndarray, phi_rot: float) -> np.ndarray:
        """Вычисление правой части"""
        return compute_rhs_2d(
            T, self.n_r, self.n_phi, self.dr, self.dphi,
            self.r_centers, self.A_radial_in, self.A_radial_out,
            self.A_tangential, self.V,
            self.meat.k, self.meat.rho, self.meat.c,
            phi_rot,
            self.heat.T_fire, self.heat.T_air,
            self.heat.h_conv_hot, self.heat.h_conv_cold,
            self.heat.epsilon,
            self.heat.fire_direction, self.heat.fire_angle_width,
            self.heat.transition_width
        )

    def estimate_stable_dt(self) -> float:
        """Оценка устойчивого шага по времени"""
        alpha = self.meat.alpha

        # Критерий Фурье для радиального направления
        dt_r = 0.25 * self.dr**2 / alpha

        # Критерий для тангенциального направления
        r_min = self.r_centers[0]
        if r_min > 0:
            arc_min = r_min * self.dphi
            dt_phi = 0.25 * arc_min**2 / alpha
        else:
            dt_phi = dt_r

        return min(dt_r, dt_phi) * 0.5

    def solve(self, verbose: bool = True) -> SimulationResults2D:
        """Основной цикл решения с прогрессбаром"""
        dt = self.estimate_stable_dt()
        t_max = self.cook.t_max
        save_interval = self.cook.save_interval

        # Оценка количества шагов
        total_steps = int(t_max / dt)

        if verbose:
            print(f"\nЗапуск 2D расчёта...")
            print(f"   dt = {dt:.5f} с")
            print(f"   Шагов: ~{total_steps}")
            print(f"   Период вращения: {self.rot.rotation_period:.1f} с")

        t = 0.0
        last_save = -save_interval
        time_at_ready = 0.0
        ready_time_start = None

        start_time = time.time()
        step = 0

        # Создаём прогрессбар
        pbar = tqdm(
            total=total_steps,
            desc="Готовим шашлык",
            unit="шаг",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            colour="red"
        )

        # Добавляем постфикс с температурой
        T_center_init = np.mean(self.T[0, :]) - 273
        pbar.set_postfix({
            "T_центр": f"{T_center_init:.1f}°C",
            "t": f"{t:.0f}с"
        })

        while t < t_max:
            phi_rot = self.get_rotation_angle(t)

            # RK4
            k1 = self.compute_rhs(self.T, phi_rot)
            k2 = self.compute_rhs(self.T + 0.5 * dt * k1, phi_rot + 0.5 * dt * self.rot.omega)
            k3 = self.compute_rhs(self.T + 0.5 * dt * k2, phi_rot + 0.5 * dt * self.rot.omega)
            k4 = self.compute_rhs(self.T + dt * k3, phi_rot + dt * self.rot.omega)

            self.T = self.T + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
            step += 1

            # Статистика
            T_center = np.mean(self.T[0, :])  # Среднее по центральным секторам
            T_surface = self.T[-1, :]
            T_surf_min = np.min(T_surface)
            T_surf_max = np.max(T_surface)
            T_surf_avg = np.mean(T_surface)

            # Проверка готовности
            is_at_ready_temp = T_center >= self.cook.T_ready

            if is_at_ready_temp:
                if ready_time_start is None:
                    ready_time_start = t
                time_at_ready = t - ready_time_start
            else:
                ready_time_start = None
                time_at_ready = 0.0

            is_ready = time_at_ready >= self.cook.t_hold

            # Обновляем прогрессбар
            pbar.update(1)
            if step % 100 == 0:  # Обновляем постфикс каждые 100 шагов для производительности
                T_ready_celsius = self.cook.T_ready - 273
                progress_to_ready = min(100, (T_center - 273) / (T_ready_celsius - 5) * 100)
                pbar.set_postfix({
                    "T_центр": f"{T_center - 273:.1f}°C",
                    "t": f"{t:.0f}с",
                    "готовность": f"{progress_to_ready:.0f}%"
                })

            # Сохранение
            if t - last_save >= save_interval or is_ready:
                snapshot = TimeSnapshot2D(
                    time=t,
                    rotation_angle=phi_rot,
                    T_field=self.T.copy(),
                    T_center=T_center,
                    T_surface_min=T_surf_min,
                    T_surface_max=T_surf_max,
                    T_surface_avg=T_surf_avg,
                    Q_total_in=0.0,  # TODO
                    is_ready=is_ready
                )
                self.results.snapshots.append(snapshot)
                last_save = t

            if is_ready:
                self.results.is_cooked = True
                self.results.cooking_time = t
                pbar.set_postfix({
                    "T_центр": f"{T_center - 273:.1f}°C",
                    "t": f"{t:.0f}с",
                    "статус": "ГОТОВО!"
                })
                pbar.close()
                if verbose:
                    print(f"\nГОТОВО! Время: {t:.0f}с ({t / 60:.1f} мин)")
                break

        if not is_ready:
            pbar.close()

        self.results.total_time = t
        self.results.computation_time = time.time() - start_time
        self.results.finalize()

        if verbose:
            if not self.results.is_cooked:
                print(f"\nНе достигнута готовность за {t_max:.0f}с")
                print(f"   T_центр = {T_center - 273:.1f}°C")
            print(f"Время расчёта: {self.results.computation_time:.2f}с")

        return self.results

# =============================================================================
# Визуализация
# =============================================================================

class Visualizer2D:
    """Визуализация 2D модели"""

    def __init__(self, results: SimulationResults2D):
        self.results = results
        self.config = results.config

    def plot_cross_section(self, time_idx: int = -1, ax=None,
                           show_mesh: bool = False, show_fire_zone: bool = True):
        """
        Полярная тепловая карта сечения
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
        else:
            fig = ax.figure

        snapshot = self.results.snapshots[time_idx]
        T = snapshot.T_field
        t = snapshot.time
        phi_rot = snapshot.rotation_angle

        n_r, n_phi = T.shape
        R = self.config.geometry.radius
        dr = R / n_r
        dphi = 2 * np.pi / n_phi

        # Сетка для pcolormesh
        r_edges = np.linspace(0, R, n_r + 1) * 1000  # в мм
        phi_edges = np.linspace(0, 2 * np.pi, n_phi + 1)

        # Цветовая карта
        T_min = self.config.cooking.T_initial - 273
        T_max = min(self.config.heat_source.T_fire - 273, 250)

        # T имеет размер [n_r, n_phi]
        # Для полярного pcolormesh(theta, r, C) нужно C размера (n_r, n_phi)
        T_celsius = T - 273

        # Отображение
        mesh = ax.pcolormesh(phi_edges, r_edges, T_celsius,
                             cmap='hot', vmin=T_min, vmax=T_max, shading='auto')

        # Сетка
        if show_mesh:
            for r in r_edges[::5]:
                ax.plot(np.linspace(0, 2 * np.pi, 100), [r] * 100, 'k-', alpha=0.2, lw=0.5)
            for phi in phi_edges[::10]:
                ax.plot([phi, phi], [0, R * 1000], 'k-', alpha=0.2, lw=0.5)

        # Зона огня
        if show_fire_zone:
            fire_dir = self.config.heat_source.fire_direction * np.pi / 180
            fire_width = self.config.heat_source.fire_angle_width * np.pi / 180
            
            # Корректируем положение зоны огня относительно вращающегося мяса
            fire_dir_relative = fire_dir - phi_rot

            # Стрелка к углям
            ax.annotate('', xy=(fire_dir_relative, R * 1000 * 1.15), xytext=(fire_dir_relative, R * 1000 * 1.3),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))
            ax.text(fire_dir_relative, R * 1000 * 1.4, '', ha='center', va='center', fontsize=16)

            # Дуга зоны нагрева
            phi_fire = np.linspace(fire_dir_relative - fire_width / 2, fire_dir_relative + fire_width / 2, 50)
            ax.plot(phi_fire, [R * 1000 * 1.05] * 50, 'r-', lw=3, alpha=0.7)


        # Настройки
        ax.set_theta_zero_location('E')  # 0° справа
        ax.set_theta_direction(1)  # По часовой стрелке
        ax.set_ylim(0, R * 1000 * 1.1)
        ax.set_title(f't = {t:.1f}с ({t / 60:.1f} мин) | '
                     f'φ = {(phi_rot * 180 / np.pi) % 360:.0f}°\n'
                     f'T_центр = {snapshot.T_center - 273:.1f}°C | '
                     f'T_пов = [{snapshot.T_surface_min - 273:.0f}..{snapshot.T_surface_max - 273:.0f}]°C',
                     pad=20)

        # Colorbar
        cbar = plt.colorbar(mesh, ax=ax, shrink=0.8, pad=0.1)
        cbar.set_label('Температура, °C')


        return fig, ax

    def plot_cross_section_cartesian(self, time_idx: int = -1, ax=None):
        """
        Картезианская тепловая карта (как imshow)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.figure

        snapshot = self.results.snapshots[time_idx]
        T = snapshot.T_field
        t = snapshot.time

        n_r, n_phi = T.shape
        R = self.config.geometry.radius * 1000

        # Создаём картезианскую сетку
        resolution = 200
        x = np.linspace(-R, R, resolution)
        y = np.linspace(-R, R, resolution)
        X, Y = np.meshgrid(x, y)

        # Преобразование в полярные
        R_grid = np.sqrt(X**2 + Y**2)
        Phi_grid = np.arctan2(Y, X) % (2 * np.pi)

        # Интерполяция
        dr = R / n_r
        dphi = 2 * np.pi / n_phi

        T_cart = np.zeros_like(R_grid)
        T_cart[:] = np.nan

        for ix in range(resolution):
            for iy in range(resolution):
                r = R_grid[iy, ix]
                phi = Phi_grid[iy, ix]

                if r <= R:
                    i_r = min(int(r / dr), n_r - 1)
                    i_phi = int(phi / dphi) % n_phi
                    T_cart[iy, ix] = T[i_r, i_phi] - 273

        # Отображение
        T_min = self.config.cooking.T_initial - 273
        T_max = min(self.config.heat_source.T_fire - 273, 250)

        im = ax.imshow(T_cart, extent=[-R, R, -R, R], origin='lower',
                       cmap='hot', vmin=T_min, vmax=T_max)

        # Контур
        circle = plt.Circle((0, 0), R, fill=False, color='black', lw=2)
        ax.add_patch(circle)

        # Центр (шампур)
        ax.plot(0, 0, 'ko', markersize=5)

        # Зона огня (снизу)
        fire_dir = self.config.heat_source.fire_direction
        ax.annotate('УГЛИ', xy=(0, -R * 1.15), fontsize=12, ha='center', color='red')

        ax.set_xlim(-R * 1.2, R * 1.2)
        ax.set_ylim(-R * 1.3, R * 1.2)
        ax.set_aspect('equal')
        ax.set_xlabel('x, мм')
        ax.set_ylabel('y, мм')
        ax.set_title(f'Разрез | t = {t:.1f}с | T_центр = {snapshot.T_center - 273:.1f}°C')

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Температура, °C')

        return fig, ax

    def plot_temperature_history(self, ax=None):
        """График температуры во времени"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.figure

        times = self.results.times / 60
        T_center = self.results.T_center_history - 273
        T_max = self.results.T_surface_max_history - 273
        T_min = self.results.T_surface_min_history - 273

        ax.fill_between(times, T_min, T_max, alpha=0.3, color='red',
                        label='Диапазон поверхности')
        ax.plot(times, T_center, 'b-', lw=2, label='Центр')
        ax.plot(times, T_max, 'r--', lw=1, label='Макс. поверхности')
        ax.plot(times, T_min, 'orange', lw=1, linestyle='--', label='Мин. поверхности')

        T_ready = self.config.cooking.T_ready - 273
        ax.axhline(y=T_ready, color='green', linestyle=':', lw=2,
                   label=f'T готовности ({T_ready:.0f}°C)')

        if self.results.is_cooked:
            ax.axvline(x=self.results.cooking_time / 60, color='green',
                       linestyle='--', alpha=0.7)

        ax.set_xlabel('Время, мин')
        ax.set_ylabel('Температура, °C')
        ax.set_title('Динамика прогрева шашлыка с вращением')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        return fig, ax

    def plot_surface_temperature_map(self, ax=None):
        """
        Карта температуры поверхности во времени (развёртка)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 6))
        else:
            fig = ax.figure

        n_times = len(self.results.snapshots)
        n_phi = self.config.geometry.n_angular

        T_surface_map = np.zeros((n_times, n_phi))
        times = np.zeros(n_times)

        for i, snap in enumerate(self.results.snapshots):
            T_surface_map[i, :] = snap.T_field[-1, :] - 273
            times[i] = snap.time

        # Отображение
        phi_deg = np.linspace(0, 360, n_phi + 1)
        time_edges = np.concatenate([[0], times])

        T_min = self.config.cooking.T_initial - 273
        T_max = min(self.config.heat_source.T_fire - 273, 250)

        im = ax.pcolormesh(phi_deg[:-1], times / 60, T_surface_map,
                           cmap='hot', vmin=T_min, vmax=T_max, shading='auto')

        ax.set_xlabel('Угол φ, градусы')
        ax.set_ylabel('Время, мин')
        ax.set_title('Температура поверхности (развёртка по углу)')

        # Линия положения огня
        fire_dir = self.config.heat_source.fire_direction
        ax.axvline(x=fire_dir, color='white', linestyle='--', lw=1, alpha=0.7)
        ax.text(fire_dir, times[-1] / 60 * 1.02, '', fontsize=12, ha='center')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Температура, °C')

        return fig, ax

    def create_summary_figure(self):
        """Итоговая фигура"""
        fig = plt.figure(figsize=(18, 12))

        # Полярный разрез
        ax1 = fig.add_subplot(2, 3, 1, projection='polar')
        self.plot_cross_section(time_idx=-1, ax=ax1)

        # Картезианский разрез
        ax2 = fig.add_subplot(2, 3, 2)
        self.plot_cross_section_cartesian(time_idx=-1, ax=ax2)

        # Разрезы в разные моменты
        ax3 = fig.add_subplot(2, 3, 3, projection='polar')
        mid_idx = len(self.results.snapshots) // 2
        self.plot_cross_section(time_idx=mid_idx, ax=ax3, show_fire_zone=False)

        # История температуры
        ax4 = fig.add_subplot(2, 3, 4)
        self.plot_temperature_history(ax=ax4)

        # Карта поверхности
        ax5 = fig.add_subplot(2, 3, 5)
        self.plot_surface_temperature_map(ax=ax5)

        # Начальный момент
        ax6 = fig.add_subplot(2, 3, 6, projection='polar')
        self.plot_cross_section(time_idx=0, ax=ax6, show_fire_zone=False)

        plt.tight_layout()
        return fig

    def create_animation(self, interval: int = 50, skip: int = 1):
        """Создание анимации (возвращает объект FuncAnimation)"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

        frames = range(0, len(self.results.snapshots), skip)

        def update(frame_idx):
            ax.clear()
            self.plot_cross_section(time_idx=frame_idx, ax=ax)
            return []

        anim = FuncAnimation(fig, update, frames=frames,
                             interval=interval, blit=False)
        return anim, fig

    def save_animation_gif(self, filename: str = 'shashlik_animation.gif',
                           skip: int = 2, fps: int = 10, dpi: int = 100):
        """
        Корректное сохранение GIF анимации с прогрессбаром

        Parameters:
        -----------
        filename : str
            Имя выходного файла
        skip : int
            Пропускать каждый N-й кадр (для ускорения)
        fps : int
            Кадров в секунду
        dpi : int
            Разрешение (dots per inch)
        """
        import io
        from PIL import Image

        # Выбираем кадры
        frame_indices = list(range(0, len(self.results.snapshots), skip))
        n_frames = len(frame_indices)

        print(f"\nСоздание GIF анимации...")
        print(f"   Кадров: {n_frames}")
        print(f"   FPS: {fps}")
        print(f"   Файл: {filename}")

        frames_pil = []

        # Прогрессбар для создания кадров
        for i, frame_idx in enumerate(tqdm(frame_indices, desc="Рендеринг кадров",
                                           unit="кадр", colour="green")):
            # Создаём фигуру для каждого кадра
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

            # Рисуем кадр
            self.plot_cross_section(time_idx=frame_idx, ax=ax, show_fire_zone=True)

            # Конвертируем в PIL Image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            buf.seek(0)

            img = Image.open(buf).convert('RGB')
            frames_pil.append(img.copy())

            buf.close()
            plt.close(fig)

        # Сохраняем GIF
        print("💾 Сохранение GIF...")
        duration_ms = int(1000 / fps)

        frames_pil[0].save(
            filename,
            save_all=True,
            append_images=frames_pil[1:],
            duration=duration_ms,
            loop=0,  # 0 = бесконечный цикл
            optimize=True
        )

        # Размер файла
        import os
        file_size = os.path.getsize(filename) / (1024 * 1024)

        print(f"Анимация сохранена: {filename}")
        print(f"   Размер: {file_size:.2f} МБ")
        print(f"   Длительность: {n_frames / fps:.1f} с")

        return filename

    def save_animation_mp4(self, filename: str = 'shashlik_animation.mp4',
                           skip: int = 2, fps: int = 15, dpi: int = 120):
        """
        Сохранение MP4 анимации (если установлен ffmpeg)

        Parameters:
        -----------
        filename : str
            Имя выходного файла
        skip : int
            Пропускать каждый N-й кадр
        fps : int
            Кадров в секунду
        dpi : int
            Разрешение
        """
        from matplotlib.animation import FFMpegWriter

        frame_indices = list(range(0, len(self.results.snapshots), skip))
        n_frames = len(frame_indices)

        print(f"\nСоздание MP4 анимации...")
        print(f"   Кадров: {n_frames}")
        print(f"   FPS: {fps}")

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

        # Настройка writer
        writer = FFMpegWriter(fps=fps, metadata={'title': 'Shashlik Heat Transfer'})

        with writer.saving(fig, filename, dpi=dpi):
            for frame_idx in tqdm(frame_indices, desc="🎞  Рендеринг",
                                  unit="кадр", colour="blue"):
                ax.clear()
                self.plot_cross_section(time_idx=frame_idx, ax=ax, show_fire_zone=True)
                writer.grab_frame()

        plt.close(fig)

        import os
        file_size = os.path.getsize(filename) / (1024 * 1024)

        print(f"Видео сохранено: {filename}")
        print(f"   Размер: {file_size:.2f} МБ")

        return filename

    def create_rotating_animation(self, filename: str = 'shashlik_rotating.gif',
                                  skip: int = 1, fps: int = 12, dpi: int = 100):
        """
        Анимация с акцентом на вращение - показывает как мясо крутится над углями

        Включает:
        - Полярную тепловую карту
        - Индикатор угла поворота
        - Температурную шкалу
        - Метку времени
        """
        import io
        from PIL import Image

        frame_indices = list(range(0, len(self.results.snapshots), skip))
        n_frames = len(frame_indices)

        print(f"\nСоздание анимации вращения...")
        print(f"   Кадров: {n_frames}")

        R = self.config.geometry.radius * 1000
        T_min = self.config.cooking.T_initial - 273
        T_max = min(self.config.heat_source.T_fire - 273, 250)
        T_ready = self.config.cooking.T_ready - 273

        frames_pil = []

        for frame_idx in tqdm(frame_indices, desc="Рендеринг",
                              unit="кадр", colour="yellow"):
            snapshot = self.results.snapshots[frame_idx]
            T = snapshot.T_field
            t = snapshot.time
            phi_rot = snapshot.rotation_angle

            # Создаём фигуру с двумя subplot'ами
            fig = plt.figure(figsize=(12, 6))

            # Левая часть - полярная карта
            ax1 = fig.add_subplot(121, projection='polar')

            n_r, n_phi = T.shape
            r_edges = np.linspace(0, R, n_r + 1)
            phi_edges = np.linspace(0, 2 * np.pi, n_phi + 1)
            T_celsius = T - 273

            mesh = ax1.pcolormesh(phi_edges, r_edges, T_celsius,
                                  cmap='hot', vmin=T_min, vmax=T_max, shading='auto')

            # Зона огня (динамически с учетом вращения)
            fire_dir = self.config.heat_source.fire_direction * np.pi / 180
            fire_width = self.config.heat_source.fire_angle_width * np.pi / 180
            
            # Корректируем положение зоны огня относительно вращающегося мяса
            fire_dir_relative = fire_dir - phi_rot
            
            phi_fire = np.linspace(fire_dir_relative - fire_width / 2, fire_dir_relative + fire_width / 2, 50)
            ax1.plot(phi_fire, [R * 1.05] * 50, 'r-', lw=4, alpha=0.8)
            ax1.text(fire_dir_relative, R * 1.2, '', ha='center', va='center', fontsize=20)


            ax1.set_theta_zero_location('E')
            ax1.set_theta_direction(1)
            ax1.set_ylim(0, R * 1.1)
            ax1.set_title(f'Поперечное сечение', fontsize=12)

            # Правая часть - информация
            ax2 = fig.add_subplot(122)
            ax2.axis('off')

            # Статус
            status_color = 'green' if snapshot.is_ready else 'orange'
            status_text = 'ГОТОВ!' if snapshot.is_ready else 'Готовится...'

            info_text = f"""
            Время: {t:.0f} с ({t / 60:.1f} мин)

            Угол поворота: {(phi_rot * 180 / np.pi) % 360:.0f}°

            Температура центра: {snapshot.T_center - 273:.1f}°C

            Поверхность:
                мин: {snapshot.T_surface_min - 273:.0f}°C
                макс: {snapshot.T_surface_max - 273:.0f}°C

            Цель: {T_ready:.0f}°C

            Статус: {status_text}
            """

            ax2.text(0.1, 0.9, info_text, transform=ax2.transAxes,
                     fontsize=14, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Прогресс-бар готовности
            progress = min(1.0, (snapshot.T_center - 273) / T_ready)
            ax2.barh([0], [progress], height=0.3, color=status_color, alpha=0.7)
            ax2.barh([0], [1.0], height=0.3, fill=False, edgecolor='black', linewidth=2)
            ax2.set_xlim(0, 1.1)
            ax2.set_ylim(-0.5, 0.5)
            ax2.text(0.5, -0.35, f'Готовность: {progress * 100:.0f}%',
                     ha='center', fontsize=12, fontweight='bold')

            # Colorbar
            cbar_ax = fig.add_axes([0.08, 0.15, 0.02, 0.3])
            cbar = fig.colorbar(mesh, cax=cbar_ax)
            cbar.set_label('T, °C', fontsize=10)

            plt.tight_layout()

            # Конвертируем в PIL
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                        facecolor='white')
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            frames_pil.append(img.copy())
            buf.close()
            plt.close(fig)

        # Сохраняем GIF
        print("Сохранение GIF...")
        duration_ms = int(1000 / fps)

        frames_pil[0].save(
            filename,
            save_all=True,
            append_images=frames_pil[1:],
            duration=duration_ms,
            loop=0,
            optimize=True
        )

        import os
        file_size = os.path.getsize(filename) / (1024 * 1024)

        print(f"Анимация сохранена: {filename}")
        print(f"   Размер: {file_size:.2f} МБ")

        return filename

    def save_video_mp4(self, filename: str = 'shashlik_video.mp4',
                       skip: int = 1, fps: int = 30, dpi: int = 120,
                       show_info_panel: bool = True):
        """
        Создание длинного MP4 видео с использованием imageio

        Parameters:
        -----------
        filename : str
            Имя выходного файла
        skip : int
            Пропускать каждый N-й кадр (1 = все кадры)
        fps : int
            Кадров в секунду (30 рекомендуется для плавного видео)
        dpi : int
            Разрешение (120 = HD качество)
        show_info_panel : bool
            Показывать информационную панель справа
        """
        import io
        from PIL import Image

        try:
            import imageio
        except ImportError:
            print("Установка imageio...")
            import subprocess
            subprocess.run(['pip', 'install', 'imageio[ffmpeg]', '--break-system-packages', '-q'])
            import imageio

        frame_indices = list(range(0, len(self.results.snapshots), skip))
        n_frames = len(frame_indices)

        # Расчёт длительности видео
        video_duration = n_frames / fps
        sim_duration = self.results.snapshots[-1].time if self.results.snapshots else 0

        print(f"\n🎬 Создание MP4 видео...")
        print(f"   Кадров: {n_frames}")
        print(f"   FPS: {fps}")
        print(f"   Длительность видео: {video_duration:.1f}с ({video_duration / 60:.1f} мин)")
        print(f"   Симуляция: {sim_duration:.0f}с ({sim_duration / 60:.1f} мин)")
        print(f"   Файл: {filename}")

        # Параметры визуализации
        R = self.config.geometry.radius * 1000
        T_min = self.config.cooking.T_initial - 273
        T_max = min(self.config.heat_source.T_fire - 273, 250)
        T_ready = self.config.cooking.T_ready - 273

        # Получаем стратегию вращения
        strategy_desc = self.config.rotation.get_strategy_description()

        # Создаём writer
        writer = imageio.get_writer(filename, fps=fps, codec='libx264',
                                    quality=8, pixelformat='yuv420p')

        try:
            for frame_idx in tqdm(frame_indices, desc="🎥 Рендеринг видео",
                                  unit="кадр", colour="blue"):
                snapshot = self.results.snapshots[frame_idx]
                T = snapshot.T_field
                t = snapshot.time
                phi_rot = snapshot.rotation_angle

                if show_info_panel:
                    # Фигура с двумя панелями
                    fig = plt.figure(figsize=(14, 7))

                    # Левая часть - полярная карта
                    ax1 = fig.add_subplot(121, projection='polar')

                    n_r, n_phi = T.shape
                    r_edges = np.linspace(0, R, n_r + 1)
                    phi_edges = np.linspace(0, 2 * np.pi, n_phi + 1)
                    T_celsius = T - 273

                    mesh = ax1.pcolormesh(phi_edges, r_edges, T_celsius,
                                          cmap='hot', vmin=T_min, vmax=T_max, shading='auto')

                    # Зона огня (динамически с учетом вращения)
                    fire_dir = self.config.heat_source.fire_direction * np.pi / 180
                    fire_width = self.config.heat_source.fire_angle_width * np.pi / 180
                    
                    # Корректируем положение зоны огня относительно вращающегося мяса
                    fire_dir_relative = fire_dir - phi_rot
                    
                    phi_fire = np.linspace(fire_dir_relative - fire_width / 2, fire_dir_relative + fire_width / 2, 50)
                    ax1.plot(phi_fire, [R * 1.05] * 50, 'r-', lw=4, alpha=0.8)
                    ax1.text(fire_dir_relative, R * 1.25, '', ha='center', fontsize=14, color='red')


                    ax1.set_theta_zero_location('E')
                    ax1.set_theta_direction(1)
                    ax1.set_ylim(0, R * 1.15)

                    # Colorbar
                    cbar = plt.colorbar(mesh, ax=ax1, shrink=0.7, pad=0.1)
                    cbar.set_label('Температура, °C', fontsize=11)

                    # Правая часть - информация
                    ax2 = fig.add_subplot(122)
                    ax2.axis('off')

                    status_color = 'green' if snapshot.is_ready else 'darkorange'
                    status_text = 'ГОТОВ!' if snapshot.is_ready else 'Готовится...'

                    # Информационный текст
                    info_lines = [
                        f"Время: {t:.0f} с ({t / 60:.1f} мин)",
                        "",
                        f"Угол поворота: {(phi_rot * 180 / np.pi) % 360:.0f}°",
                        f"   Стратегия: {strategy_desc[:40]}",
                        "",
                        f"Температура центра: {snapshot.T_center - 273:.1f}°C",
                        "",
                        f"Поверхность:",
                        f"    мин: {snapshot.T_surface_min - 273:.0f}°C",
                        f"    макс: {snapshot.T_surface_max - 273:.0f}°C",
                        f"    перепад: {snapshot.T_surface_max - snapshot.T_surface_min:.0f}°C",
                        "",
                        f"Цель: {T_ready:.0f}°C",
                        "",
                        f"Статус: {status_text}"
                    ]

                    ax2.text(0.05, 0.95, '\n'.join(info_lines), transform=ax2.transAxes,
                             fontsize=13, verticalalignment='top', fontfamily='monospace',
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

                    # Прогресс-бар готовности
                    progress = min(1.0, max(0, (snapshot.T_center - 273 - T_min) / (T_ready - T_min)))

                    # Рисуем прогресс-бар
                    bar_y = 0.15
                    ax2.barh([bar_y], [progress], height=0.08, color=status_color, alpha=0.8)
                    ax2.barh([bar_y], [1.0], height=0.08, fill=False, edgecolor='black', linewidth=2)
                    ax2.set_xlim(-0.05, 1.1)
                    ax2.set_ylim(0, 1)
                    ax2.text(0.5, bar_y - 0.06, f'Прогрев: {progress * 100:.0f}%',
                             ha='center', fontsize=13, fontweight='bold',
                             transform=ax2.transAxes)

                else:
                    # Только полярная карта
                    fig, ax1 = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
                    self.plot_cross_section(time_idx=frame_idx, ax=ax1, show_fire_zone=True)

                plt.tight_layout()

                # Конвертируем в numpy array для imageio
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                buf.seek(0)

                img = Image.open(buf).convert('RGB')
                frame_array = np.array(img)

                writer.append_data(frame_array)

                buf.close()
                plt.close(fig)

        finally:
            writer.close()

        # Размер файла
        import os
        file_size = os.path.getsize(filename) / (1024 * 1024)

        print(f"\nВидео сохранено: {filename}")
        print(f"   Размер: {file_size:.2f} МБ")
        print(f"   Длительность: {video_duration:.1f}с")
        print(
            f"   Разрешение: {dpi * 14}x{dpi * 7} px" if show_info_panel else f"   Разрешение: {dpi * 10}x{dpi * 10} px")

        return filename

    def create_comparison_video(self, other_results: 'SimulationResults2D',
                                filename: str = 'comparison.mp4',
                                labels: tuple = ('Стратегия 1', 'Стратегия 2'),
                                skip: int = 1, fps: int = 24, dpi: int = 100):
        """
        Создание видео сравнения двух стратегий вращения

        Parameters:
        -----------
        other_results : SimulationResults2D
            Результаты второй симуляции для сравнения
        filename : str
            Имя выходного файла
        labels : tuple
            Названия стратегий
        """
        import io
        from PIL import Image

        try:
            import imageio
        except ImportError:
            import subprocess
            subprocess.run(['pip', 'install', 'imageio[ffmpeg]', '--break-system-packages', '-q'])
            import imageio

        # Определяем количество кадров (минимум из двух)
        n1 = len(self.results.snapshots)
        n2 = len(other_results.snapshots)
        n_frames = min(n1, n2)

        frame_indices = list(range(0, n_frames, skip))

        print(f"\nСоздание видео сравнения...")
        print(f"   Кадров: {len(frame_indices)}")
        print(f"   {labels[0]} vs {labels[1]}")

        R = self.config.geometry.radius * 1000
        T_min = self.config.cooking.T_initial - 273
        T_max = min(self.config.heat_source.T_fire - 273, 250)

        writer = imageio.get_writer(filename, fps=fps, codec='libx264', quality=8)

        try:
            for frame_idx in tqdm(frame_indices, desc="Рендеринг", colour="magenta"):
                snap1 = self.results.snapshots[frame_idx]
                snap2 = other_results.snapshots[frame_idx]

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7),
                                               subplot_kw={'projection': 'polar'})

                for ax, snap, label in [(ax1, snap1, labels[0]), (ax2, snap2, labels[1])]:
                    T = snap.T_field
                    n_r, n_phi = T.shape
                    r_edges = np.linspace(0, R, n_r + 1)
                    phi_edges = np.linspace(0, 2 * np.pi, n_phi + 1)

                    mesh = ax.pcolormesh(phi_edges, r_edges, T - 273,
                                         cmap='hot', vmin=T_min, vmax=T_max, shading='auto')

                    # Зона огня (динамически с учетом вращения)
                    fire_dir = 270 * np.pi / 180  # 270° = снизу
                    fire_width = self.config.heat_source.fire_angle_width * np.pi / 180
                    fire_dir_relative = fire_dir - snap.rotation_angle
                    
                    phi_fire = np.linspace(fire_dir_relative - fire_width / 2, fire_dir_relative + fire_width / 2, 50)
                    ax.plot(phi_fire, [R * 1.05] * 50, 'r-', lw=3, alpha=0.7)
                    ax.text(fire_dir_relative, R * 1.2, '', ha='center', fontsize=16)

                    ax.set_theta_zero_location('E')
                    ax.set_theta_direction(1)
                    ax.set_ylim(0, R * 1.1)
                    ax.set_title(f'{label}\nT_центр={snap.T_center - 273:.1f}°C | '
                                 f'φ={(snap.rotation_angle * 180 / np.pi) % 360:.0f}°',
                                 fontsize=12)

                fig.suptitle(f't = {snap1.time:.0f}с ({snap1.time / 60:.1f} мин)',
                             fontsize=14, fontweight='bold')
                plt.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
                buf.seek(0)

                img = Image.open(buf).convert('RGB')
                writer.append_data(np.array(img))

                buf.close()
                plt.close(fig)

        finally:
            writer.close()

        import os
        file_size = os.path.getsize(filename) / (1024 * 1024)
        print(f"Видео сравнения сохранено: {filename} ({file_size:.2f} МБ)")

        return filename


# =============================================================================
# Интерфейс
# =============================================================================

def create_config_interactive() -> SimulationConfig2D:
    """Интерактивный ввод параметров"""
    print("\n" + "=" * 60)
    print("  НАСТРОЙКА 2D МОДЕЛИ ШАШЛЫКА С ВРАЩЕНИЕМ")
    print("=" * 60)
    print("(Enter = значение по умолчанию)\n")

    def inp(prompt, default, t=float):
        try:
            v = input(f"{prompt} [{default}]: ").strip()
            return default if v == '' else t(v)
        except:
            return default

    meat_name = inp("   Мясо ", "Свинина",str)

    print("ГЕОМЕТРИЯ:")
    radius = inp("   Радиус (мм)", 15.0) / 1000
    L = inp("   Длина (мм)", 50.0) / 1000
    n_r = inp("   Ячеек по радиусу", 25, int)
    n_phi = inp("   Секторов по углу", 60, int)

    print("ТЕПЛОФИЗИЧЕСКИЕ СВОЙСТВА МЯСА:")
    k = inp("   k (Вт/м*К)", 0.33, float)
    p = inp("   ρ (кг/м³)", 1030, float)
    c = inp("   c (Дж/(кг·К))", 3056, float)
    T_init = inp("   Начальная температура мяса (°C)", 5, float) + 273

    print("\nСТРАТЕГИЯ ВРАЩЕНИЯ:")
    print("   1. Постоянное вращение (как на электрошампуре)")
    print("   2. Переворот на 90° через интервал (ручной)")
    print("   3. Переворот на 180° через интервал")
    print("   4. Без вращения (статика)")

    strat_choice = input("   Выбор [1]: ").strip() or '1'

    if strat_choice == '1':
        strategy = RotationStrategy.CONTINUOUS
        period = inp("   Период оборота (с)", 10.0)
        flip_interval = 30.0
    elif strat_choice == '2':
        strategy = RotationStrategy.FLIP_90
        period = 10.0
        flip_interval = inp("   Интервал между переворотами (с)", 30.0)
    elif strat_choice == '3':
        strategy = RotationStrategy.FLIP_180
        period = 10.0
        flip_interval = inp("   Интервал между переворотами (с)", 45.0)
    else:
        strategy = RotationStrategy.STATIC
        period = 10.0
        flip_interval = 30.0

    print("\nИСТОЧНИКИ ТЕПЛА:")
    T_fire = inp("   Температура углей (°C)", 300.0) + 273
    T_air = inp("   Температура воздуха (°C)", 40.0) + 273
    fire_width = inp("   Угловая ширина зоны огня (°)", 120.0)

    print("\nГОТОВНОСТЬ:")
    T_ready = inp("   Температура в центре (°C)", 70.0) + 273
    t_hold = inp("   Время удержания температуры в центре (с)", 60.0)
    t_max = inp("   Макс. время (с)", 900.0)

    config = SimulationConfig2D(
        meat=MeatProperties(name=meat_name,k=k,rho=p,c=c),
        geometry=CylinderGeometry(radius=radius,length=L, n_radial=n_r, n_angular=n_phi),
        rotation=RotationConfig(strategy=strategy, rotation_period=period, flip_interval=flip_interval),
        heat_source=HeatSourceConfig(T_fire=T_fire, T_air=T_air, fire_angle_width=fire_width),
        cooking=CookingConditions(T_initial=T_init, T_ready=T_ready,t_hold=t_hold, t_max=t_max)
    )

    return config


def create_default_config() -> SimulationConfig2D:
    """Стандартная конфигурация"""
    return SimulationConfig2D()


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 65)
    print("  2D МОДЕЛЬ ПРОГРЕВА ШАШЛЫКА С ВРАЩЕНИЕМ НАД УГЛЯМИ")
    print("=" * 65)

    print("\nРежим:")
    print("  1. Стандартные параметры")
    print("  2. Настроить параметры")
    print("  3. Загрузить результаты")

    choice = input("\nВыбор [1]: ").strip() or '1'

    results = None

    if choice == '1':
        config = create_default_config()
    elif choice == '2':
        config = create_config_interactive()
    elif choice == '3':
        fname = input("Файл (без .pkl): ").strip()
        results = SimulationResults2D.load(fname)
        print(f"Загружено: {fname}")
    else:
        config = create_default_config()

    if results is None:
        config.print_summary()
        input("\nEnter для запуска...")

        model = CylindricalHeatTransfer2D(config)
        results = model.solve(verbose=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"shashlik_2d_{timestamp}"
        results.save(fname)

    # Визуализация
    print("\nВизуализация...")
    viz = Visualizer2D(results)

    fig = viz.create_summary_figure()
    fig.savefig('shashlik_2d_summary.png', dpi=150, bbox_inches='tight')
    print("   Сохранено: shashlik_2d_summary.png")

    # Анимация/Видео
    print("\nСоздание анимации/видео:")
    print("  1. Простая GIF (быстро, ~1 МБ)")
    print("  2. Расширенная GIF с информацией")
    print("  3. MP4 видео (HD качество)")
    print("  4. Все варианты")
    print("  5. Пропустить")

    anim_choice = input("\nВыбор [5]: ").strip() or '5'

    if anim_choice in ['1', '4']:
        viz.save_animation_gif(
            filename='shashlik_simple.gif',
            skip=2,
            fps=10,
            dpi=100
        )

    if anim_choice in ['2', '4']:
        viz.create_rotating_animation(
            filename='shashlik_detailed.gif',
            skip=2,
            fps=12,
            dpi=100
        )

    if anim_choice in ['3', '4']:
        viz.save_video_mp4(
            filename='shashlik_video.mp4',
            skip=1,
            fps=24,
            dpi=100,
            show_info_panel=True
        )

    plt.show()
    print("\nГотово!")

    return results


def run_comparison(strategy1: str = RotationStrategy.CONTINUOUS,
                   strategy2: str = RotationStrategy.FLIP_90,
                   t_max: float = 300.0):
    """
    Запуск сравнения двух стратегий вращения

    Parameters:
    -----------
    strategy1, strategy2 : str
        Стратегии для сравнения (из RotationStrategy)
    t_max : float
        Время симуляции в секундах
    """
    print("\n" + "=" * 65)
    print("  СРАВНЕНИЕ СТРАТЕГИЙ ВРАЩЕНИЯ")
    print("=" * 65)

    # Базовая конфигурация
    base_config = {
        'geometry': CylinderGeometry(n_radial=20, n_angular=48),
        'heat_source': HeatSourceConfig(),
        'cooking': CookingConditions(t_max=t_max, save_interval=1.0)
    }

    # Конфигурация 1
    config1 = SimulationConfig2D(
        **base_config,
        rotation=RotationConfig(strategy=strategy1, rotation_period=8.0, flip_interval=30.0)
    )

    # Конфигурация 2
    config2 = SimulationConfig2D(
        **base_config,
        rotation=RotationConfig(strategy=strategy2, rotation_period=8.0, flip_interval=30.0)
    )

    print(f"\nСтратегия 1: {config1.rotation.get_strategy_description()}")
    print(f"Стратегия 2: {config2.rotation.get_strategy_description()}")
    print(f"Время симуляции: {t_max:.0f}с\n")

    # Запуск симуляций
    print("Запуск симуляции 1...")
    model1 = CylindricalHeatTransfer2D(config1)
    results1 = model1.solve(verbose=False)

    print("Запуск симуляции 2...")
    model2 = CylindricalHeatTransfer2D(config2)
    results2 = model2.solve(verbose=False)

    # Результаты
    print("\nРЕЗУЛЬТАТЫ:")
    print(f"   Стратегия 1 - T_центр: {results1.snapshots[-1].T_center - 273:.1f}°C")
    print(f"   Стратегия 2 - T_центр: {results2.snapshots[-1].T_center - 273:.1f}°C")

    # Создание видео сравнения
    viz1 = Visualizer2D(results1)
    viz1.create_comparison_video(
        results2,
        filename='comparison.mp4',
        labels=(config1.rotation.get_strategy_description()[:25],
                config2.rotation.get_strategy_description()[:25]),
        skip=2,
        fps=20
    )

    return results1, results2


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        # Режим сравнения
        run_comparison()
    else:
        # Обычный режим
        results = main()