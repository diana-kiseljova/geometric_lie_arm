# geometric_lie_arm

Этот модуль (`src/arm` + `src/algorithms/arm_*` + `launch_arm.py`) реализует
управление 7-DOF манипулятором Franka Panda в физической симуляции MuJoCo.
Дополнительно к 6-DOF ODE-контроллерам твёрдого тела
(`src/algorithms/{pid,lqr,mpc}_v*`), он использует *реальную* модель сцены
(`scene.xml` + `panda.xml`), низкоуровневые PD-актуаторы Panda, и решает
задачи в **картовом** и **суставном** пространствах.

### Связь с проектными SE(3)-контроллерами

Старые `pid_v*` / `lqr_v*` / `mpc_v*` управляют 6-DOF твёрдым телом в
свободном пространстве. Для руки они не годятся «как есть» (другое
состояние, другая динамика), поэтому здесь повторно используется не
сам runner, а его **математика**:

```
       q, dq ∈ R⁷  ──FK──→  X_ee(q) ∈ SE(3)  ──log──→  ψ ∈ R⁶  ──J_b(q)──→  u_joint
       (MuJoCo)           (src.arm.se3_bridge)        (src.utils.math       (ArmWorld
                                                       .error / hat /        .set_arm_ctrl)
                                                        unhat)
```

SE(3) остаётся **только на уровне ошибки рабочего органа**, а динамика и
управление живут в суставах. Это позволило прямо портировать формулу
PID из [src/algorithms/pid_v1/control.py](src/algorithms/pid_v1/control.py)
и LTI-LQR из [src/algorithms/lqr_v3/control.py](src/algorithms/lqr_v3/control.py)
на новый объект — без отдельного «runner»-а и ODE-интегратора. См. §4.4-4.5.

---

## 1. Архитектура

```
                              ┌─────────────────────────┐
                              │  launch_arm.py          │
                              │  (читает launch_arm.yaml)│
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


> Важное следствие: на нижнем уровне Panda управляется по позициям, а PD
> уже встроен в актуатор. Все наши алгоритмы поэтому работают так:
>
> *получи 6D-ошибку (или ошибку суставов) → вычисли желаемое
> приращение `Δq` → проинтегрируй в `q_cmd` → запиши в `data.ctrl[:7]`.*
>
> Если потребуется чистое управление по моментам, надо переписать
> актуаторы на `motor` и переделать алгоритмы — для этого
> предусмотрены отдельные ветки в `arm_osc` (она ближе всего к
> torque-варианту, отличается только финальной интеграцией).

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

### 3.2 Кинематика ([kinematics.py](src/arm/kinematics.py))

Без зависимостей кроме `numpy`. Используется обоими ориентированными
алгоритмами.

- `quat_mul(a, b)`, `quat_conj(q)` — кватернионы в формате MuJoCo `(w,x,y,z)`.
- `quat_log_vec(q)` — логарифм единичного кватерниона:
  возвращает 3-вектор `axis·angle`, нормированный в `[-π, π]`.
- `orientation_error(q_cur, q_des)`:
  $$
  e_{\theta} = \log\bigl(q_{\text{des}} \cdot q_{\text{cur}}^{-1}\bigr)
  \in \mathbb{R}^3
  $$
  с автоматическим выбором «короткого пути» (флипом знака при `w<0`).
- `pose_error(p, q, p_d, q_d)` → `[Δp; e_θ] ∈ R^6`.
- `dls_step(J, e, λ, gain)`:
  $$
  \Delta q \;=\; J^{\top}\,(J J^{\top} + \lambda^2 I)^{-1}\,(\text{gain}\cdot e),
  $$
  Damped Least Squares — устойчив у сингулярностей и для прямоугольных $J$.

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

Все три контроллера имеют один и тот же интерфейс:

```python
class Controller:
    name: str
    def reset(self, world: ArmWorld) -> None: ...
    def step(
        self,
        world:   ArmWorld,
        desired: Desired,
        dt:      float,
    ) -> tuple[np.ndarray, float]:
        """Возвращает (q_target_arm (7,), gripper_openness ∈ [0,1])."""
```

Реестр в [src/config/arm_control.py](src/config/arm_control.py):

```python
arm_control_map = {
    "arm_joint_pd":   {"cls": JointPDController,   "params": {...}},
    "arm_jacobian_ik":{"cls": JacobianIKController,"params": {...}},
    "arm_osc":        {"cls": OSCController,       "params": {...}},
}
make_arm_controller("arm_osc", overrides={"kp_pos": 8.0})
```

### 4.1 `arm_joint_pd` — управление в суставном пространстве

[arm_joint_pd/control.py](src/algorithms/arm_joint_pd/control.py)

Идея: задача задаёт `q_target` (7-вектор целевых углов). Контроллер
сдвигает накопленную команду `q_cmd` в сторону `q_target` шагом не
больше `max_step`:

$$
\Delta q = \mathrm{clip}\bigl(K_p\,(q_{\text{target}} - q_{\text{cmd}}),
\,-\Delta_{\max},\,\Delta_{\max}\bigr)
$$
$$
q_{\text{cmd}} \leftarrow \mathrm{clip}(q_{\text{cmd}} + \Delta q,\, q_{\min},\, q_{\max})
$$

Параметры:
- `kp` — пропорциональный коэффициент;
- `max_step` — максимальное приращение `q_cmd` за такт (рад).

Когда применять: когда задача напрямую формулируется в суставах
(например, плановая траектория от внешнего планировщика). Для задач
`ReachPose`/`PickAndPlace`, формулирующих цель в EE, этот контроллер не
подходит (он будет просто держать home).

### 4.2 `arm_jacobian_ik` — Jacobian-IK с DLS

[arm_jacobian_ik/control.py](src/algorithms/arm_jacobian_ik/control.py)

Это **обратная кинематика по производным**, без явного PD по времени:

$$
e = \begin{bmatrix}
K_{\text{pos}}\,(p_d - p) \\
K_{\text{ori}}\,e_{\theta}
\end{bmatrix}, \qquad
\Delta q_{\text{task}} = J^{\top}(J J^{\top} + \lambda^2 I)^{-1}\,e
$$

В нулевом пространстве (любое смещение $N\Delta q$ не меняет
картов. цели) тянем к домашней конфигурации:

$$
N = I - J^{\#} J, \qquad \Delta q_{\text{null}} = K_{\text{post}}\,(q_{\text{home}} - q_{\text{cmd}})
$$
$$
\Delta q = \mathrm{clip}\bigl(\Delta q_{\text{task}} + N\,\Delta q_{\text{null}},\,
\pm\Delta_{\max}\bigr)
$$

Если `gain_ori = 0`, используется только позиционный якобиан (3×7).

Параметры:
- `gain_pos`, `gain_ori` — усиления (безразм., применяются прямо к
  ошибке);
- `damping` — λ в DLS, защищает от близких к сингулярности
  конфигураций;
- `posture_gain` — усиление вторичной задачи (приведение к home);
- `max_step` — клиппинг `Δq` (рад/такт).

Когда применять: лучший компромисс «простой и работает» для задач с
целевой позой EE. Без явных усилений по времени, поэтому сходимость
неравномерна — при коротком T может не успеть.

### 4.3 `arm_osc` — Operational Space Control (Cartesian PD)

[arm_osc/control.py](src/algorithms/arm_osc/control.py)

PD по картовому пространству **по времени**, с переносом в суставы
через DLS-псевдообратную:

1. Считаем линейную/угловую скорость EE через якобиан:
   $$
   v = J_{1:3}\dot q,\qquad \omega = J_{4:6}\dot q
   $$
2. Желаемая 6D-скорость по PD:
   $$
   v_{\text{des}} = K_p^{p}(p_d - p) - K_d^{p}\,v
   $$
   $$
   \omega_{\text{des}} = K_p^{\theta}\,e_{\theta} - K_d^{\theta}\,\omega
   $$
3. Картовая → суставная скорость:
   $$
   \dot q_{\text{task}} = J^{\top}(JJ^{\top}+\lambda^2 I)^{-1}
   \begin{bmatrix} v_{\text{des}}\\ \omega_{\text{des}}\end{bmatrix}
   $$
4. Nullspace-задача — стабилизация к home:
   $$
   N = I - J^{\#}J,\qquad
   \dot q_{\text{null}} = K_{\text{null}}\,(q_{\text{home}} - q_{\text{cmd}})
   $$
5. Интеграция в позиционное управление:
   $$
   q_{\text{cmd}} \leftarrow \mathrm{clip}\!\left(
     q_{\text{cmd}} + \mathrm{clip}\bigl((\dot q_{\text{task}}+N\dot q_{\text{null}})\,dt,
     \pm\Delta_{\max}\bigr),
     q_{\min}, q_{\max}\right)
   $$

Параметры:
- `kp_pos`, `kd_pos` — PD по позиции (1/с, безразм. на скорость);
- `kp_ori`, `kd_ori` — PD по ориентации;
- `damping` — λ в DLS;
- `k_null` — усиление nullspace-задачи;
- `dq_max` — клиппинг $\Delta q$ за такт.

Когда применять: лучший выбор, когда нужно гладкое и быстрое
отслеживание траектории EE с осмысленными единицами усилений.
В смоук-тесте OSC сводит ошибку позиции с 23 см до 4 мм за 2 секунды.

### 4.4 `arm_se3_pid` — PID на SE(3)-ошибке EE

[arm_se3_pid/control.py](src/algorithms/arm_se3_pid/control.py)

**Прямой порт** PID из
[src/algorithms/pid_v1/control.py](src/algorithms/pid_v1/control.py) на
руку. Использует общую SE(3)-математику проекта
([src/utils/math.py](src/utils/math.py)). Архитектурно:

$$
\underbrace{q,\dot q}_{\text{MuJoCo}}
\xrightarrow{\text{FK}}
X_{ee}(q)\in SE(3)
\xrightarrow{\log}
\psi\in\mathbb{R}^{6}
\xrightarrow{\text{PID}}
u\in\mathbb{R}^{6}
\xrightarrow{J_b^{\#}}
\dot q_{\text{cmd}}
\xrightarrow{\int}
q_{\text{cmd}}
$$

На каждом такте:

1. Считается ошибка EE на группе Ли через `error()` из [src/utils/math.py](src/utils/math.py):
   $$
   \Psi = X_d^{-1} X_{ee},\qquad
   \psi = \widehat{}^{-1}(\log \Psi) \in \mathbb{R}^6.
   $$

2. Берётся **body-Jacobian** $J_b(q) \in \mathbb{R}^{6\times 7}$
   (`src/arm/se3_bridge.body_jacobian`):
   $$
   J_b = \begin{bmatrix} R^{\top}\,J_r \\ R^{\top}\,J_p \end{bmatrix},
   \qquad \xi = J_b\,\dot q.
   $$

3. PID — **в точности формула из `pid_v1`** (строки 41-48):
   $$
   I \mathrel{+}= \psi\,dt,\qquad
   u = K_p\,\psi + K_d\,\dot\psi + K_i\,I,
   $$
   где $\dot\psi$ для квазистатических целей упрощается до $\xi$ (в
   полной формуле — `unhat(hat(ξ) − Ψ⁻¹ hat(ξ_d) Ψ)`, см.
   `psi_and_psidot`).

4. Проекция в суставы через DLS-псевдообратную:
   $$
   \dot q_{\text{task}} = J_b^{\#}\,u,
   \quad J_b^{\#} = J_b^{\top}(J_b J_b^{\top}+\lambda^2 I)^{-1}.
   $$

5. Nullspace-задача (приведение к home) и интеграция:
   $$
   N = I - J_b^{\#} J_b,\quad
   \dot q_{\text{null}} = K_{\text{post}}(q_{\text{home}} - q_{\text{cmd}}),
   $$
   $$
   q_{\text{cmd}} \leftarrow \mathrm{clip}\!\left(
     q_{\text{cmd}} + \mathrm{clip}((\dot q_{\text{task}}+N\dot q_{\text{null}})\,dt, \pm\Delta_{\max}),
     q_{\min}, q_{\max}\right).
   $$

Параметры:
- `Kp`, `Kd`, `Ki` ∈ ℝ⁶ — те же знаки и порядок (`[θ; ρ]`), что в
  `pid_v1`: **отрицательные**, потому что $\psi$ положителен в сторону
  цели.
- `damping` — λ в $J_b^{\#}$.
- `posture_gain`, `dq_max` — как в OSC.

### 4.5 `arm_se3_lqr` — LTI LQR на SE(3)-ошибке EE

[arm_se3_lqr/control.py](src/algorithms/arm_se3_lqr/control.py)

Тот же принцип, что в [src/algorithms/lqr_v4/control.py](src/algorithms/lqr_v4/control.py):
SE(3)-error LTI baseline, но на 6D-ошибке **EE** вместо тела:

$$
x = \begin{bmatrix}\psi \\ \dot\psi\end{bmatrix} \in \mathbb{R}^{12},\qquad
\dot x = \begin{bmatrix}0 & I_6 \\ 0 & 0\end{bmatrix}x +
         \begin{bmatrix}0 \\ I_6\end{bmatrix}u
$$

Дискретизация (ZOH, шаг `dt`):

$$
A = \begin{bmatrix}I_6 & dt\,I_6 \\ 0 & I_6\end{bmatrix},\qquad
B = \begin{bmatrix}\tfrac{1}{2}dt^2\,I_6 \\ dt\,I_6\end{bmatrix}.
$$

Решается **дискретное DARE** один раз при `reset()`
(`scipy.linalg.solve_discrete_are`), получается постоянная матрица
$K \in \mathbb{R}^{6\times 12}$. На каждом такте:

$$
u = -K\,x,\qquad
\dot q_{\text{cmd}} = J_b^{\#}\,u,\qquad
q_{\text{cmd}} \mathrel{+}= \dot q_{\text{cmd}}\,dt + N\,\dot q_{\text{null}}\,dt.
$$

Параметры:
- `Q` ∈ ℝ¹²ˣ¹² — `diag([Q_ψ, Q_ψ̇])`. Полезно ставить translational
  части (`ρ_x, ρ_y, ρ_z`) в 2–3 раза больше угловых;
- `R` ∈ ℝ⁶ˣ⁶ — стоимость управления;
- `damping`, `posture_gain`, `dq_max` — как в `arm_se3_pid`.

В смоук-тесте `arm_se3_lqr` сводит ошибку позиции **с 23 см до 0.1 мм
за 3 секунды** на ReachPose — на порядок точнее, чем OSC, при тех же
ограничениях скорости (`dq_max=0.03`).

### 4.6 Связка с проектными SE(3)-функциями

| Источник                                            | Что переиспользовано в arm_se3_*                  |
|-----------------------------------------------------|---------------------------------------------------|
| `src/utils/math.error(X, X_d)`                      | формула $\Psi = X_d^{-1} X$, $\psi = \mathrm{unhat}(\log\Psi)$ |
| `src/utils/math.hat / unhat`                        | конвертация ξ ↔ ξ̂ ∈ se(3)                        |
| `src/utils/math.R_p_to_X`                           | сборка X из R и p                                 |
| `src/algorithms/pid_v1` (строки 41-48)              | формула `u = Kp·ψ + Kd·ψ̇ + Ki·I`                 |
| `src/algorithms/lqr_v3` (модель `A`, `B`, DARE)     | LTI baseline, `solve_discrete_are`               |

Что **не** переиспользовано (и почему):
- `kin_dyn` / `kin_dyn_with_quat` — это интегратор тела через `odeint`;
  для руки физикой занимается MuJoCo.
- `Initials` / `Body` / `Trajectory` из 6-DOF-проекта — там захардкожены
  поля под рабочее тело, не подходят для манипулятора.
- `run_pid` / `run_lqr` — они монолитные batch-функции, а для realtime-loop
  нужен «шаговый» интерфейс. Идея перенесена, но код шага написан заново.

### 4.7 Как добавить свой алгоритм

1. Создать `src/algorithms/arm_myalgo/__init__.py` и `control.py`
   с классом, у которого есть `reset(world)` и
   `step(world, desired, dt) -> (q_arm_7, gripper_0_1)`.
2. Добавить запись в [src/config/arm_control.py](src/config/arm_control.py):
   ```python
   arm_control_map["arm_myalgo"] = {
       "cls": MyAlgoController,
       "params": {...},
   }
   ```
3. Указать `controller_name: arm_myalgo` в `launch_arm.yaml`.

---

## 5. Запуск

```powershell
conda activate robot_mujoco
python launch_arm.py -a launch_arm.yaml
```

Зависимости: `mujoco`, `numpy`, `pyyaml`, `tqdm` (все в окружении
`robot_mujoco`).

### 5.1 `launch_arm.yaml` — описание полей

```yaml
scene_xml: scene.xml          # путь к MuJoCo сцене
T: 12.0                       # длительность симуляции, с
realtime: true                # синхронизация со стенными часами (для viewer)
viewer: true                  # открыть mujoco.viewer.launch_passive

controller_name: arm_osc      # arm_joint_pd | arm_jacobian_ik | arm_osc
controller_overrides: {}      # точечные оверрайды параметров контроллера

task_name: pick_and_place     # hold_home | reach_pose | ee_line | pick_and_place
task_kwargs: {}               # kwargs конструктора задачи

save_root_path: arm_runs      # куда писать history.npz и копию yaml
extra_info: ""                # суффикс в имени каталога результатов
```

### 5.2 Примеры `task_kwargs`

```yaml
# 1. Удерживать одну точку
task_name: reach_pose
task_kwargs:
  pos:  [0.5, 0.0, 0.4]
  quat: [0.0, 1.0, 0.0, 0.0]
  gripper: 1.0

# 2. Двигаться по линии EE
task_name: ee_line
task_kwargs:
  p_a:    [0.5, -0.2, 0.4]
  p_b:    [0.5,  0.2, 0.4]
  period: 6.0

# 3. Pick & place: куб в (0.6, 0.25, 0.035) → положить в (0.30, -0.25)
task_name: pick_and_place
task_kwargs:
  cube_init_pos: [0.60, 0.25, 0.035]
  place_xy:      [0.30, -0.25]
  T_approach: 3.0
  T_descend:  3.0
  T_close:    1.0
  T_lift:     2.0
  T_carry:    3.0
  T_release:  1.0
```

### 5.3 Примеры `controller_overrides`

```yaml
controller_name: arm_osc
controller_overrides:
  kp_pos: 8.0          # увеличить пропорционал по позиции
  kd_pos: 3.0
  k_null: 0.2          # ослабить тяготение к home
  dq_max: 0.03         # разрешить более быстрые движения суставов
```

---

## 6. Формат истории

После прогона в `<save_root_path>/<auto-name>/history.npz` сохраняются
массивы (`N = T / dt` шагов):

| Ключ        | Форма       | Описание                                  |
|-------------|-------------|-------------------------------------------|
| `t`         | `(N,)`      | время симуляции                           |
| `q`         | `(N, 7)`    | реальные углы суставов                    |
| `dq`        | `(N, 7)`    | угловые скорости                          |
| `q_target`  | `(N, 7)`    | управление, отправленное в `data.ctrl[:7]`|
| `gripper`   | `(N,)`      | управление гриппером (после клиппинга)    |
| `ee_pos`    | `(N, 3)`    | реальная позиция EE                       |
| `ee_quat`   | `(N, 4)`    | реальная ориентация (w,x,y,z)             |
| `ee_pos_d`  | `(N, 3)`    | желаемая позиция EE                       |
| `ee_quat_d` | `(N, 4)`    | желаемая ориентация EE                    |
| `calc_time` | `(N,)`      | время `controller.step()`, с              |
| `e_pos`     | `(N,)`      | `‖ee_pos_d − ee_pos‖₂`                    |

Загрузка:

```python
import numpy as np
hist = np.load("arm_runs/arm_osc__pick_and_place__T12.0/history.npz")
print(hist["e_pos"].mean(), hist["e_pos"].max())
```

---

## 7. Сравнение алгоритмов

Smoke-тест на задаче `reach_pose` ([0.5, 0.0, 0.4] из home), `T = 3 с`,
`dt = 0.002`:

| Контроллер         | старт   | конец   | макс    | примечание                                  |
|--------------------|---------|---------|---------|---------------------------------------------|
| `arm_joint_pd`     | 0.231   | 0.225   | 0.231   | не знает картовой цели — держит home        |
| `arm_jacobian_ik`  | 0.231   | 0.046   | 0.291   | DLS IK без D-члена, неравномерно            |
| `arm_osc`          | 0.231   | 0.001   | 0.231   | Cartesian PD по времени                     |
| `arm_se3_pid`      | 0.231   | 0.003   | 0.231   | PID из pid_v1 на ψ = log(X_d⁻¹·X_ee)        |
| `arm_se3_lqr`      | 0.231   | **0.0001** | 0.231 | **лучший** — LTI LQR на [ψ; ψ̇]              |



T: 18.0
arm_se3_lqr:       EE position error  mean=0.0397 m  max=0.2899 m  final=0.0004 m
arm_se3_pid:       EE position error  mean=0.0252 m  max=0.2899 m  final=0.0070 m
arm_osc:           EE position error  mean=0.0517 m  max=0.2899 m  final=0.0024 m
arm_jacobian_ik:   EE position error  mean=0.2070 m  max=0.9654 m  final=0.0221 m
arm_joint_pd:      EE position error  mean=0.3745 m  max=0.6009 m  final=0.6009 m
---

## 8. Ограничения и расширения

- **Чисто torque-режим** не реализован: актуаторы Panda в `panda.xml` —
  позиционные PD. Если нужен прямой torque-контроль (например, для
  Computed Torque Control с динамическим компенсатором), надо
  заменить актуаторы на `<motor>` и переписать финальный шаг алгоритма
  (вместо `q_cmd += dq·dt` — вычисление и подача `τ`).
- **Сцена**: гриппер сейчас контролируется через нормированный
  `actuator8 ∈ [0,255]`. Если перенесёте проект на другую модель —
  поправьте `ArmWorld.GRIPPER_ACTUATOR_NAME` и константы
  `GRIPPER_OPEN_CTRL/CLOSED_CTRL`.
- **IK и сингулярности**: при близких к плоской конфигурации позах
  Jacobian становится почти вырожденным. Поднимайте `damping` или
  включайте более агрессивный nullspace (`k_null`/`posture_gain`).
- **Совместимость со старым проектом**: `src/controller.py` и
  `launch.py` для 6-DOF тела остались без изменений — это две
  независимые подсистемы. Только `src/config/` теперь содержит ещё
  один файл (`arm_control.py`).

---

## 9. Карта файлов

```
custom/
├── launch_arm.py                      # entrypoint для arm
├── launch_arm.yaml                    # его конфиг
├── README_arm.md                      # этот файл
├── scene.xml                          # MuJoCo сцена (Panda + куб)
├── panda.xml                          # модель робота
│
└── src/
    ├── arm/
    │   ├── __init__.py                # реэкспорты
    │   ├── world.py                   # ArmWorld
    │   ├── kinematics.py              # quat, dls_step, pose_error
    │   ├── se3_bridge.py              # FK→X_ee, J_b, ψ̇, dls_pinv
    │   ├── tasks.py                   # HoldHome, ReachPose, EELine, PickAndPlace
    │   └── runner.py                  # ArmRunner + история
    │
    ├── algorithms/
    │   ├── arm_joint_pd/control.py    # суставный PD
    │   ├── arm_jacobian_ik/control.py # DLS IK (позиция + ориентация)
    │   ├── arm_osc/control.py         # Operational Space Control
    │   ├── arm_se3_pid/control.py     # PID из pid_v1 на ψ=log(X_d⁻¹·X_ee)
    │   └── arm_se3_lqr/control.py     # LTI LQR из lqr_v3 на [ψ; ψ̇]
    │
    └── config/
        └── arm_control.py             # arm_control_map + make_arm_controller
```
