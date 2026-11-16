# geometric_lie_arm

Модуль (`src/arm`, `src/algorithms/arm_*`, `launch_arm.py`) реализует управление 7-степенным манипулятором Franka Panda в физическом симуляторе MuJoCo.

В отличие от существующих 6-DOF контроллеров (`src/algorithms/{pid,lqr,mpc}_v*`), работающих с упрощённой ODE-моделью твёрдого тела, новый модуль использует полноценную модель робота (`scene.xml`, `panda.xml`) с учётом его кинематики и динамики.

Поддерживается управление как **в пространстве суставов**, так и **в декартовом пространстве** (положением и ориентацией рабочего органа). Управляющие воздействия передаются через низкоуровневые PD-актуаторы манипулятора.

### Связь с SE(3)-контроллерами

В проекте реализованы PID, LQR и MPC-контроллеры (pid_v*, lqr_v*, mpc_v*) для управления свободным твёрдым телом с 6 степенями свободы (6-DOF).

Эти контроллеры нельзя напрямую применить к манипулятору Franka Panda, поскольку робот имеет 7 управляемых суставов, а его кинематика и динамика существенно отличаются от модели свободного твёрдого тела.

Поэтому в новом модуле повторно используются математические принципы управления, адаптированные к роботизированной руке:
- SE(3) — для представления положения и ориентации рабочего органа и вычисления ошибки относительно целевой позы.
- PID — для регулирования ошибки положения и ориентации рабочего органа.
- LQR — для оптимального управления на основе локальной линейной модели ошибки позы и скорости.

Якобиан манипулятора — для преобразования декартовых управляющих команд в скорости суставов с использованием Damped Least Squares (DLS).

Таким образом, PID и LQR работают с 6-мерной ошибкой позы рабочего органа в SE(3), а полученные управляющие команды преобразуются в целевые положения семи суставов.

В отличие от исходных контроллеров, физическая динамика робота рассчитывается непосредственно в MuJoCo, а выполнение команд обеспечивается суставными PD-актуаторами. Для этого используется отдельный цикл симуляции ArmRunner вместо существующего ODE-интегратора.

```
q, dq ∈ R⁷  ──FK──→  X_ee(q) ∈ SE(3)
                         │
X_target ∈ SE(3) ─────────┤
                         ▼
                 Ψ = X_target⁻¹ X_ee
                         │
                        log
                         ▼
                ψ, ψ̇ ∈ R⁶
                         │
                     PID / LQR
                         ▼
                u_ee ∈ R⁶
                         │
                   J_b# (DLS)
                         ▼
                dq_cmd ∈ R⁷
                         │
                  интеграция
                         ▼
                q_target ∈ R⁷
                         │
                  PD-актуаторы
                         ▼
                       MuJoCo
```

SE(3) остаётся **только на уровне ошибки рабочего органа**, а динамика и
управление живут в суставах.

![Демонстрация управления Franka Panda для SE(3) контроллера](arm_se3_pid.gif)

---

## 1. Архитектура

```
                              ┌─────────────────────────┐
                              │  launch_arm.py          │
                              │  (launch_arm.yaml)      │
                              └────────────┬────────────┘
                                           │
                  ┌────────────────────────┼────────────────────────┐
                  ▼                        ▼                        ▼
        ┌─────────────────┐      ┌──────────────────┐     ┌──────────────────┐
        │  src.arm.World  │      │   src.arm.Task   │     │   src.algorithms │
        │  ArmWorld       │      │   ReachPose,     │     │   arm_joint_pd   │
        │  (обёртка над   │      │   FollowEELine,  │     │   arm_jacobian_ik│
        │  MuJoCo)        │      │   PickAndPlace,  │     │   arm_osc        │
        │                 │      │   HoldHome       │     │   …              │
        └─────────────────┘      └──────────────────┘     └──────────────────┘
                  │                        │                        │
                  └───────────┬────────────┴────────────┬───────────┘
                              ▼                         ▼
                       ┌─────────────────────────────────────────┐
                       │            src.arm.ArmRunner            │
                       │  for k in range(N):                     │
                       │      desired = task.desired(t, world)   │
                       │      q, gripper = ctrl.step(world, …)   │
                       │      world.set_arm_ctrl(q)              │
                       │      world.set_gripper_norm(gripper)    │
                       │      world.step()                       │
                       └─────────────────────────────────────────┘
                                          │
                                          ▼
                              ┌─────────────────────┐
                              │   history.npz       │
                              │   (q, ee, e_pos…)   │
                              └─────────────────────┘
```

Слои разделены так, чтобы можно было независимо:
- менять **сцену** (XML),
- менять **задачу** (что должен делать робот),
- менять **алгоритм управления** (как именно),
- логировать одинаковую историю и сравнивать алгоритмы.

---

## 2. Каркас сцены

`scene.xml` подключает `panda.xml` и добавляет пол + куб со свободным
шарниром:

```xml
<body name="cube" pos="0.60 0.25 0.035">
  <freejoint name="cube_freejoint"/>
  <geom name="cube_geom" type="box" size="0.035 0.035 0.035"
        rgba="1 0.2 0.1 1" mass="0.05" friction="1 0.005 0.0001"/>
</body>
```

```
joint1, joint2, ..., joint7 - в модели есть 7 основных суставов (они соответствуют 7 степеням свободы руки Panda)

qpos — где робот сейчас
ctrl — куда мы хотим его привести

ctrl[0:7]  = желаемые углы 7 суставов руки
ctrl[7]    = команда гриппера от 0 до 255
qpos       = реальные текущие координаты модели
```

```
Для каждого сустава MuJoCo примерно делает:
tau = K * (q_des - q) - D * qdot
q_des = ctrl[i]
q     = текущее положение сустава
qdot  = текущая скорость сустава
K     = stiffness / proportional gain
D     = damping / derivative gain
```

```
data.ctrl[0] = 1.0 - хочу, чтобы joint1 пришёл в положение 1.0 радиан
```

```
Keyframe — это сохранённая поза робота
qpos = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04, …]
Это значит, что в состоянии home робот физически стоит в такой конфигурации:
joint1 = 0
joint2 = 0
joint3 = 0
joint4 = -pi/2
joint5 = 0
joint6 = pi/2
joint7 = -0.7853
finger_joint1 = 0.04
finger_joint2 = 0.04
```

В `panda.xml` есть:
- 7 шарниров руки (`joint1` … `joint7`) и 2 шарнира гриппера
  (`finger_joint1`, `finger_joint2`, связанные `<equality>`);
- 8 актуаторов: первые 7 — **позиционные PD-актуаторы**
  (`<general gainprm="K" biasprm="0 -K -D"/>`), их `ctrl` — это **желаемое
  положение сустава**;
- актуатор `actuator8` для гриппера, `ctrl ∈ [0, 255]` (`0` — закрыт, `255` — открыт);
- keyframe `home`:
  ```
  qpos = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853,  0.04, 0.04, …]
  ctrl = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853,  255]
  ```

---

## 3. Модуль `src/arm`

### 3.1 `ArmWorld` ([world.py](src/arm/world.py))

Тонкая обёртка над `MjModel` + `MjData`, прячущая boilerplate.

```python
world = ArmWorld(scene_xml="scene.xml", extra_bodies=["cube"])
world.reset_home()           # загрузить keyframe + посчитать FK
world.q()                    # текущие 7 углов суставов  → (7,)
world.dq()                   # текущие 7 угл. скоростей  → (7,)
world.ee_pos()               # позиция EE (body "hand")  → (3,)
world.ee_quat()              # ориентация EE (w,x,y,z)   → (4,)
world.jacobian_arm()         # 6x7 якобиан hand по 7 суставам
world.jacobian_arm_pos()     # 3x7 — только линейная часть
world.body_pos("cube")       # позиция куба              → (3,)
world.set_arm_ctrl(q_target) # data.ctrl[:7] ← clip(q_target, ctrl_min, ctrl_max)
world.set_gripper_norm(0.5)  # 0 — закрыт, 1 — открыт
world.step()                 # mj_step
world.dt                     # шаг интегратора симуляции
world.t                      # текущее время симуляции
```

Доп. возможности:
- `set_freejoint_pose("cube_freejoint", [x,y,z])` — поставить куб
  в нужную точку перед стартом (используется `PickAndPlace.on_reset`);
- встроенный viewer:
  `world.open_viewer()`, `world.viewer_sync()`, `world.viewer_running()`,
  `world.close_viewer()`;
- атрибут `q_home` — конфигурация суставов из keyframe `home`, нужна
  алгоритмам как posture-цель в nullspace.


### 3.2 Кинематика (`src/arm/kinematics.py`)

Модуль для вычисления ошибок положения и ориентации, работы с кватернионами и обратной кинематики. Зависит только от `numpy`.

**Основные функции:**

- `quat_mul(a, b)`, `quat_conj(q)` — операции с кватернионами в формате MuJoCo `(w, x, y, z)`.
- `quat_log_vec(q)` — преобразование единичного кватерниона в 3D-вектор вращения `axis × angle`.
- `pose_error(p, q, p_d, q_d)` — вычисление 6D-ошибки позы: `[Δp, e_θ]`.

**Ошибка ориентации — `orientation_error(q_cur, q_des)`:**

```math
e_\theta = \log(q_{\mathrm{des}} \cdot q_{\mathrm{cur}}^{-1}) \in \mathbb{R}^3
```

Используется кратчайшее вращение к целевой ориентации.

**Обратная кинематика — `dls_step(J, e, λ, gain)`:**

```math
\Delta q = J^\top (JJ^\top + \lambda^2 I)^{-1} (\mathrm{gain} \cdot e)
```

Метод Damped Least Squares (DLS) преобразует декартовую ошибку в приращения углов суставов. Параметр `λ` обеспечивает устойчивость вблизи сингулярных конфигураций.


### 3.3 Задачи ([tasks.py](src/arm/tasks.py))

`Task.desired(t, world) -> Desired` — это «опорный сигнал», который
контроллер должен отслеживать. `Desired` содержит:

```python
@dataclass
class Desired:
    ee_pos:   np.ndarray   # (3,)  желаемая позиция EE
    ee_quat:  np.ndarray   # (4,)  желаемая ориентация EE (w,x,y,z)
    gripper:  float        # 0..1  (0 — закрыт, 1 — открыт)
    q_target: np.ndarray | None  # (7,) опц. целевые q (для arm_joint_pd)
```

Готовые задачи:
| Имя              | Что делает                                                                 |
|------------------|----------------------------------------------------------------------------|
| `hold_home`      | Возвращает в home; полезно как baseline                                    |
| `reach_pose`     | Постоянная целевая поза `(pos, quat, gripper)`                             |
| `ee_line`        | Синусоидально ходит между `p_a` и `p_b` с периодом `period`                |
| `pick_and_place` | Полный сценарий **approach → descend → close → lift → carry → release**    |

`PickAndPlace.on_reset(world)` вызывается перед стартом симуляции и
ставит куб в `cube_init_pos`.

`make_task(name, **kwargs)` — конструктор по имени из YAML.

### 3.4 `ArmRunner` ([runner.py](src/arm/runner.py))

Главный цикл. На каждом такте:

```
desired = task.desired(t, world)
q_cmd, gripper = controller.step(world, desired, dt)
q_applied = world.set_arm_ctrl(q_cmd)        # с клиппингом по ctrlrange
g_applied = world.set_gripper_norm(gripper)
hist.append(t, q, dq, q_target=q_applied, gripper=g_applied,
            ee_pos, ee_quat, ee_pos_d, ee_quat_d, calc_time)
world.step()                                 # mj_step
if viewer: world.viewer_sync()
```

Опции:
- `T` — длительность симуляции (с);
- `realtime=True` — спать `dt − wall_dt`, чтобы viewer показывал
  «честное» время;
- `open_viewer=True` — открыть `mujoco.viewer.launch_passive`.

История пишется в `dict[str, np.ndarray]` и затем сохраняется в `.npz`
(см. §6).

---

## 4. Алгоритмы

Все пять контроллеров используют единый интерфейс:

```python
class Controller:
    name: str

    def reset(self, world: ArmWorld) -> None: ...

    def step(
        self,
        world: ArmWorld,
        desired: Desired,
        dt: float,
    ) -> tuple[np.ndarray, float]:
        """Возвращает (q_target_arm (7,), gripper_openness ∈ [0, 1])."""
```

Контроллеры регистрируются в [`src/config/arm_control.py`](src/config/arm_control.py):

```python
arm_control_map = {
    "arm_joint_pd":     {"cls": JointPDController,    "params": {...}},
    "arm_jacobian_ik":  {"cls": JacobianIKController, "params": {...}},
    "arm_osc":          {"cls": OSCController,        "params": {...}},
    "arm_se3_pid":      {"cls": SE3PIDController,     "params": {...}},
    "arm_se3_lqr":      {"cls": SE3LQRController,     "params": {...}},
}

make_arm_controller("arm_osc", overrides={"kp_pos": 8.0})
```

### 4.1 `arm_joint_pd` — управление в суставном пространстве

[`arm_joint_pd/control.py`](src/algorithms/arm_joint_pd/control.py)

Задача задаёт `q_target` — семь целевых углов суставов. Контроллер постепенно перемещает команду `q_cmd` в сторону цели, ограничивая изменение за один такт:

```math
\Delta q = \operatorname{clip}\left(K_p(q_{\mathrm{target}}-q_{\mathrm{cmd}}),\,-\Delta_{\max},\,\Delta_{\max}\right)
```

```math
q_{\mathrm{cmd}} \leftarrow \operatorname{clip}\left(q_{\mathrm{cmd}} + \Delta q,\,q_{\min},\,q_{\max}\right)
```

Параметры:

- `kp` — пропорциональный коэффициент.
- `max_step` — максимальное приращение `q_cmd` за такт (рад).

**Когда применять:** если задача сразу задана в суставном пространстве, например, траекторией от внешнего планировщика. Для `ReachPose` и `PickAndPlace`, задающих цель через позу рабочего органа (EE), этот контроллер не подходит: без `q_target` он удерживает `home`.

### 4.2 `arm_jacobian_ik` — обратная кинематика с DLS

[`arm_jacobian_ik/control.py`](src/algorithms/arm_jacobian_ik/control.py)

Вычисляет приращения углов суставов по ошибке положения и ориентации EE. Использует Damped Least Squares (DLS) без отдельного временного PD-регулятора:

```math
e = \begin{bmatrix}
K_{\mathrm{pos}}(p_d-p) \\
K_{\mathrm{ori}}e_\theta
\end{bmatrix},\qquad
\Delta q_{\mathrm{task}} = J^\top(JJ^\top + \lambda^2 I)^{-1}e
```

Дополнительная задача в нулевом пространстве якобиана приближает конфигурацию к `home`:

```math
N = I-J^{\#}J,\qquad
\Delta q_{\mathrm{null}} = K_{\mathrm{post}}(q_{\mathrm{home}}-q_{\mathrm{cmd}})
```

```math
\Delta q = \operatorname{clip}\left(\Delta q_{\mathrm{task}} + N\Delta q_{\mathrm{null}},\,-\Delta_{\max},\,\Delta_{\max}\right)
```

Если `gain_ori = 0`, используется только позиционный якобиан размерности 3×7.

Параметры:

- `gain_pos`, `gain_ori` — коэффициенты для позиционной и ориентационной ошибок.
- `damping` — коэффициент `λ`, стабилизирующий DLS вблизи сингулярностей.
- `posture_gain` — стремление к домашней конфигурации.
- `max_step` — максимальное приращение углов за такт (рад).

**Когда применять:** простой базовый метод для достижения целевой позы EE. Скорость сходимости зависит от настроек и частоты обновления; за короткое время цель может не достигаться.

### 4.3 `arm_osc` — декартовый PD-контроллер

[`arm_osc/control.py`](src/algorithms/arm_osc/control.py)

Регулирует положение и ориентацию EE в декартовом пространстве, учитывая текущую скорость. Результат переводит в суставные команды через DLS.

**1. Скорости EE через якобиан:**

```math
v = J_{1:3}\dot q,\qquad \omega = J_{4:6}\dot q
```

**2. Желаемые линейная и угловая скорости:**

```math
v_{\mathrm{des}} = K_p^p(p_d-p)-K_d^p v
```

```math
\omega_{\mathrm{des}} = K_p^\theta e_\theta-K_d^\theta\omega
```

**3. Переход к суставным скоростям:**

```math
\dot q_{\mathrm{task}} = J^\top(JJ^\top+\lambda^2I)^{-1}
\begin{bmatrix}v_{\mathrm{des}} \\ \omega_{\mathrm{des}}\end{bmatrix}
```

**4. Дополнительная задача — стабилизация к `home`:**

```math
N = I-J^{\#}J,\qquad
\dot q_{\mathrm{null}} = K_{\mathrm{null}}(q_{\mathrm{home}}-q_{\mathrm{cmd}})
```

**5. Интеграция в целевые положения суставов:**

```math
q_{\mathrm{cmd}}\leftarrow\operatorname{clip}\left(
q_{\mathrm{cmd}}+
\operatorname{clip}\left((\dot q_{\mathrm{task}}+N\dot q_{\mathrm{null}})dt,\,-\Delta_{\max},\,\Delta_{\max}\right),
\,q_{\min},\,q_{\max}\right)
```

Параметры:

- `kp_pos`, `kd_pos` — коэффициенты PD для положения.
- `kp_ori`, `kd_ori` — коэффициенты PD для ориентации.
- `damping` — регуляризация DLS.
- `k_null` — сила стремления к `home`.
- `dq_max` — максимальное приращение углов за такт.

**Когда применять:** для плавного отслеживания траектории EE. В одном smoke-тесте ошибка положения уменьшилась с 23 см до 4 мм за 2 секунды.

> Здесь `OSC` реализован как **кинематический декартовый PD**, а не torque-level Operational Space Control с динамической компенсацией.

### 4.4 `arm_se3_pid` — PID по ошибке позы в SE(3)

[`arm_se3_pid/control.py`](src/algorithms/arm_se3_pid/control.py)

Адаптация PID из [`pid_v1`](src/algorithms/pid_v1/control.py) для манипулятора. Использует SE(3)-операции из [`src/utils/math.py`](src/utils/math.py).

```text
q, dq (7 суставов)
    |
    v
Прямая кинематика (FK) -> поза EE: X_ee ∈ SE(3)
    |
    v
Ошибка относительно цели: ψ = Log(X_d⁻¹ X_ee) ∈ R⁶
    |
    v
PID -> декартова команда u ∈ R⁶
    |
    v
DLS-псевдообратная J_b# -> суставные скорости
    |
    v
Интеграция -> q_cmd -> PD-актуаторы -> MuJoCo
```

**1. Ошибка текущей позы относительно целевой:**

```math
\Psi = X_d^{-1}X_{ee},\qquad
\psi = \operatorname{unhat}(\log\Psi)\in\mathbb{R}^6
```

**2. Body-Jacobian**, переводящий суставные скорости в скорость рабочего органа в его локальной системе координат:

```math
J_b = \begin{bmatrix}R^\top J_r \\ R^\top J_p\end{bmatrix}
\in\mathbb{R}^{6\times7},\qquad
\xi = J_b\dot q
```

**3. PID-регулятор** (по принципу `pid_v1`):

```math
I \leftarrow I+\psi\,dt,\qquad
u = K_p\psi+K_d\dot\psi+K_i I
```

Для неподвижной цели производная ошибки может быть выражена через скорость EE. Полная формула вычисляется в `psi_and_psidot`.

**4. Преобразование в суставные скорости через DLS:**

```math
\dot q_{\mathrm{task}} = J_b^{\#}u,\qquad
J_b^{\#}=J_b^\top(J_bJ_b^\top+\lambda^2I)^{-1}
```

**5. Nullspace-регулирование и интеграция:**

```math
N=I-J_b^{\#}J_b,\qquad
\dot q_{\mathrm{null}}=K_{\mathrm{post}}(q_{\mathrm{home}}-q_{\mathrm{cmd}})
```

```math
q_{\mathrm{cmd}}\leftarrow \operatorname{clip}\left(
q_{\mathrm{cmd}}+\operatorname{clip}\left(
(\dot q_{\mathrm{task}}+N\dot q_{\mathrm{null}})dt,\,-\Delta_{\max},\,\Delta_{\max}
\right),\,q_{\min},\,q_{\max}\right)
```

Параметры:

- `Kp`, `Kd`, `Ki` — шесть коэффициентов для компонент `[θ; ρ]`. Как и в `pid_v1`, используются **отрицательные знаки**, поскольку ошибка определена как `X_d⁻¹ X_ee`.
- `damping` — регуляризация DLS.
- `posture_gain`, `dq_max` — аналогично `arm_osc`.

### 4.5 `arm_se3_lqr` — LQR по ошибке позы в SE(3)

[`arm_se3_lqr/control.py`](src/algorithms/arm_se3_lqr/control.py)

Адаптация LTI LQR из [`lqr_v4`](src/algorithms/lqr_v4/control.py): регулирует 6D-ошибку **рабочего органа**, а не свободного тела. Состояние содержит ошибку позы и её производную:

```math
x=\begin{bmatrix}\psi\\\dot\psi\end{bmatrix}\in\mathbb{R}^{12},\qquad
\dot x=\begin{bmatrix}0&I_6\\0&0\end{bmatrix}x+
\begin{bmatrix}0\\I_6\end{bmatrix}u
```

Дискретизация с шагом `dt` (ZOH):

```math
A=\begin{bmatrix}I_6&dt\,I_6\\0&I_6\end{bmatrix},\qquad
B=\begin{bmatrix}\frac12dt^2I_6\\dt\,I_6\end{bmatrix}
```

При `reset()` один раз решается дискретное уравнение Риккати (DARE) с помощью `scipy.linalg.solve_discrete_are`. На каждом такте:

```math
u=-Kx,\qquad
\dot q_{\mathrm{cmd}}=J_b^{\#}u
```

```math
q_{\mathrm{cmd}}\leftarrow q_{\mathrm{cmd}}+
\left(\dot q_{\mathrm{cmd}}+N\dot q_{\mathrm{null}}\right)dt
```

Параметры:

- `Q` (12×12) — веса ошибок позы и скорости; поступательным компонентам можно назначать более высокие веса.
- `R` (6×6) — веса управляющих воздействий.
- `damping`, `posture_gain`, `dq_max` — как в `arm_se3_pid`.

В одном smoke-тесте `arm_se3_lqr` уменьшил ошибку положения с 23 см до **0,1 мм за 3 секунды** на `ReachPose` при `dq_max = 0.03`.

### 4.6 Связь с существующими SE(3)-функциями

| Источник | Что используется в `arm_se3_*` |
|---|---|
| `src/utils/math.error(X, X_d)` | Ошибка позы `Ψ = X_d⁻¹ X`, `ψ = unhat(log Ψ)` |
| `src/utils/math.hat / unhat` | Преобразование 6D-вектора в элемент `se(3)` и обратно |
| `src/utils/math.R_p_to_X` | Сборка матрицы позы `X` из вращения `R` и переноса `p` |
| `src/algorithms/pid_v1` | PID-закон `u = Kp·ψ + Kd·ψ̇ + Ki·I` |
| `src/algorithms/lqr_v3` | Линейная модель состояния и решение DARE |

**Что не используется напрямую:**

- `kin_dyn` / `kin_dyn_with_quat` — ODE-интегратор свободного тела; физика манипулятора рассчитывается в MuJoCo.
- `Initials` / `Body` / `Trajectory` — структуры для свободного тела, не подходящие для суставной модели.
- `run_pid` / `run_lqr` — монолитные процедуры; для симуляции робота реализован пошаговый интерфейс `reset()` / `step()`.

### 4.7 Добавление своего алгоритма

1. Создать `src/algorithms/arm_myalgo/__init__.py` и `control.py` с классом, реализующим `reset(world)` и `step(world, desired, dt) -> (q_arm_7, gripper_0_1)`.
2. Зарегистрировать его в [`src/config/arm_control.py`](src/config/arm_control.py):

   ```python
   arm_control_map["arm_myalgo"] = {
       "cls": MyAlgoController,
       "params": {...},
   }
   ```

3. Указать `controller_name: arm_myalgo` в `launch_arm.yaml`.

---

## 5. Запуск

Сцена выбирается в `launch_arm.py` автоматически: `scene.xml` для обычных задач и `scene_with_cube.xml` для `pick_and_place`.

```powershell
conda activate aiotter
python launch_arm.py -a launch_arm.yaml
```

Зависимости: `mujoco`, `numpy`, `scipy`, `pyyaml`, `tqdm`.

### 5.1 Поля `launch_arm.yaml`

```yaml
T: 12.0                     # длительность симуляции (с)
realtime: true              # синхронизация с реальным временем
viewer: true                # открыть MuJoCo viewer

controller_name: arm_osc    # arm_joint_pd | arm_jacobian_ik | arm_osc | arm_se3_pid | arm_se3_lqr
controller_overrides: {}    # переопределение параметров контроллера

task_name: pick_and_place  # hold_home | reach_pose | ee_line | pick_and_place
task_kwargs: {}             # параметры задачи

save_root_path: arm_runs    # каталог для history.npz и конфигурации
extra_info: ""             # суффикс имени результата
```

### 5.2 Примеры `task_kwargs`

**Достижение заданной позы:**

```yaml
task_name: reach_pose
task_kwargs:
  pos: [0.5, 0.0, 0.4]
  quat: [0.0, 1.0, 0.0, 0.0]
  gripper: 1.0
```

**Движение рабочего органа между двумя точками:**

```yaml
task_name: ee_line
task_kwargs:
  p_a: [0.5, -0.2, 0.4]
  p_b: [0.5, 0.2, 0.4]
  period: 6.0
```

**Перенос куба:**

```yaml
task_name: pick_and_place
task_kwargs:
  cube_init_pos: [0.60, 0.25, 0.035]
  place_xy: [0.30, -0.25]
  T_approach: 3.0
  T_descend: 3.0
  T_close: 1.0
  T_lift: 2.0
  T_carry: 3.0
  T_release: 1.0
```

### 5.3 Пример `controller_overrides`

```yaml
controller_name: arm_osc
controller_overrides:
  kp_pos: 8.0    # коэффициент по положению
  kd_pos: 3.0
  k_null: 0.2    # сила стабилизации к home
  dq_max: 0.03  # ограничение изменения угла за такт
```

---

## 6. Формат истории

После запуска данные сохраняются в `<save_root_path>/<auto-name>/history.npz`, где `N ≈ T / dt` — количество тактов симуляции.

| Ключ | Размерность | Описание |
|---|---|---|
| `t` | `(N,)` | Время симуляции |
| `q` | `(N, 7)` | Фактические углы суставов |
| `dq` | `(N, 7)` | Угловые скорости |
| `q_target` | `(N, 7)` | Команда для `data.ctrl[:7]` |
| `gripper` | `(N,)` | Команда грипперу |
| `ee_pos` | `(N, 3)` | Фактическое положение EE |
| `ee_quat` | `(N, 4)` | Фактическая ориентация `(w,x,y,z)` |
| `ee_pos_d` | `(N, 3)` | Целевое положение EE |
| `ee_quat_d` | `(N, 4)` | Целевая ориентация EE |
| `calc_time` | `(N,)` | Время выполнения `controller.step()` (с) |
| `e_pos` | `(N,)` | Норма ошибки положения EE |

Пример загрузки:

```python
import numpy as np

hist = np.load("arm_runs/arm_osc__pick_and_place__T12.0/history.npz")
print(hist["e_pos"].mean(), hist["e_pos"].max())
```

---

## 7. Сравнение алгоритмов

Smoke-тест `reach_pose`: целевая позиция `[0.5, 0.0, 0.4]`, начальная поза `home`, `T = 3 с`, `dt = 0.002 с`. Ошибка положения EE указана в метрах.

| Контроллер | Начало | Конец | Максимум | Примечание |
|---|---:|---:|---:|---|
| `arm_joint_pd` | 0.231 | 0.225 | 0.231 | Удерживает `home`, не использует декартовую цель |
| `arm_jacobian_ik` | 0.231 | 0.046 | 0.291 | DLS IK без временного D-члена |
| `arm_osc` | 0.231 | 0.001 | 0.231 | Декартовый PD |
| `arm_se3_pid` | 0.231 | 0.003 | 0.231 | PID по ошибке SE(3) |
| `arm_se3_lqr` | 0.231 | **0.0001** | 0.231 | LTI LQR по ошибке SE(3) |

Отдельный прогон при `T = 18 с`:

```text
arm_se3_lqr:      EE position error mean=0.0397 m max=0.2899 m final=0.0004 m
arm_se3_pid:      EE position error mean=0.0252 m max=0.2899 m final=0.0070 m
arm_osc:          EE position error mean=0.0517 m max=0.2899 m final=0.0024 m
arm_jacobian_ik:  EE position error mean=0.2070 m max=0.9654 m final=0.0221 m
arm_joint_pd:     EE position error mean=0.3745 m max=0.6009 m final=0.6009 m
```

Эти числа относятся к конкретным тестам, а не к универсальному рейтингу контроллеров.

---

## 8. Ограничения и расширения

- **Torque-level управление не реализовано:** актуаторы Panda в `panda.xml` — позиционные PD. Для непосредственного задания суставных моментов потребуется изменить модель актуаторов и выход контроллеров с `q_cmd` на `τ`.
- **Гриппер:** используется `actuator8` с диапазоном `[0, 255]`. Для другой модели робота нужно проверить `ArmWorld.GRIPPER_ACTUATOR_NAME`, `GRIPPER_OPEN_CTRL` и `GRIPPER_CLOSED_CTRL`.
- **Сингулярности IK:** вблизи вырожденных конфигураций можно увеличить `damping` и настроить nullspace-регулирование (`k_null` / `posture_gain`).
- **Совместимость:** `src/controller.py` и `launch.py` для 6-DOF свободного тела остаются отдельной подсистемой; новая регистрация контроллеров находится в `src/config/arm_control.py`.

---

## 9. Карта файлов

```text
custom/
├── launch_arm.py                    # запуск симуляции манипулятора
├── launch_arm.yaml                  # конфигурация
├── README_arm.md                    # документация
├── scene.xml                        # MuJoCo-сцена
├── scene_with_cube.xml              # сцена для pick_and_place
├── panda.xml                        # модель Franka Panda
└── src/
    ├── arm/
    │   ├── __init__.py
    │   ├── world.py                 # ArmWorld
    │   ├── kinematics.py            # кватернионы, DLS, ошибки позы
    │   ├── se3_bridge.py            # SE(3), body-Jacobian
    │   ├── tasks.py                 # задачи
    │   └── runner.py                # ArmRunner и запись истории
    ├── algorithms/
    │   ├── arm_joint_pd/control.py
    │   ├── arm_jacobian_ik/control.py
    │   ├── arm_osc/control.py
    │   ├── arm_se3_pid/control.py
    │   └── arm_se3_lqr/control.py
    └── config/
        └── arm_control.py           # реестр контроллеров
```
