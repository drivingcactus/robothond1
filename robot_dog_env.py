import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class RobotDogEnv(gym.Env):
    # ---- reward-gewichten op één plek zodat je ze makkelijk kunt tunen ----
    W_TRACKING = 1.0       # snelheid-tracking richting TARGET_VEL (max 1.0 per stap)
    W_AIR_TIME = 3.0       # beloning per voet-neerzetting voor een fatsoenlijke zwaaiduur
    W_ACTION_RATE = 0.1    # straf op snel veranderende acties (trillen)
    W_TORQUE = 0.02        # straf op servo-koppel (energie)
    W_JOINT_VEL = 0.001    # straf op gewrichtssnelheid
    W_TILT = 0.5           # straf op scheef staan
    W_TURN = 0.2           # straf op draaien (yaw-rate)
    W_DRAG = 2.0           # straf op een voet die op de grond staat en toch horizontaal beweegt
    FALL_PENALTY = 50.0

    TARGET_VEL = 0.3       # m/s vooruit - we belonen het HALEN van deze snelheid, niet "zo snel mogelijk"
    TRACKING_SIGMA = 0.05  # scherpte van de tracking-beloning: exp(-err^2 / sigma). Stilstaan levert ~0.17 op, doel = 1.0
    AIR_TIME_MIN = 0.1     # s: een kortere zwaai (trillen / hupsen) levert straf op bij neerzetten
    AIR_TIME_MAX = 0.5     # s: langer in de lucht dan dit levert niks extra op
    ACTION_SCALE = 0.5     # rad: actie [-1, 1] wordt een offset van +-0.5 rad rond de nominale (gehurkte) pose

    def __init__(self, xml_path="robot_dog.xml", max_steps=1000, frame_skip=20):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # frame_skip: aantal physics-stappen per policy-actie. Met timestep 0.001 en frame_skip 20
        # beslist de policy op 50 Hz (zoals een servo-loop op de Pi) en is een episode van
        # max_steps=1000 stappen 20 s sim-tijd.
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.step_count = 0

        self.n_actuators = self.model.nu  # 8 (4 legs x hip+knee)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_actuators,), dtype=np.float32)

        # obs: joint pos (8) + joint vel (8) + accelerometer (3) + gyro (3) + gefilterde pitch/roll (2) + vorige actie (8)
        # de pitch/roll komen uit een complementary filter (accel+gyro fusie) - net zoals je op de Pi zou draaien
        # de vorige actie zit erin zodat de policy zijn eigen acties kan gladstrijken (zie action-rate straf)
        obs_dim = self.n_actuators * 2 + 3 + 3 + 2 + self.n_actuators
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.dt = self.model.opt.timestep            # physics timestep
        self.control_dt = self.dt * self.frame_skip  # tijd tussen twee policy-acties
        self.filter_alpha = 0.98  # hoeveel we op de gyro vertrouwen t.o.v. de accelerometer
        self.filt_pitch = 0.0
        self.filt_roll = 0.0

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1]

        # nominale (gehurkte) pose uit de "home" keyframe in de XML. De policy stuurt offsets rond deze pose,
        # zodat "niks doen" (actie 0) = netjes gehurkt staan i.p.v. met gestrekte, vergrendelde knieën.
        self.home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.nominal_ctrl = self.model.key_qpos[self.home_key_id, 7:7 + self.n_actuators].copy()

        self.torso_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "torso_center")
        self.torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        legs = ["fl", "fr", "rl", "rr"]
        self.foot_site_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot") for leg in legs]
        # geom -> been-index, voor echte contactdetectie (alleen de onderbenen hebben collision aan)
        lower_leg_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_lower_leg") for leg in legs]
        self.foot_geom_to_leg = {}
        for g in range(self.model.ngeom):
            for i, b in enumerate(lower_leg_body_ids):
                if self.model.geom_bodyid[g] == b:
                    self.foot_geom_to_leg[g] = i

        # dof indices for the 8 leg joints (dof 0-5 = free joint: 3 pos + 3 rot)
        self.leg_dof_start = 6
        self.leg_dof_end = 6 + self.n_actuators

        # remember base values so randomization each episode doesn't drift/stack
        self.base_damping = self.model.dof_damping[self.leg_dof_start:self.leg_dof_end].copy()
        self.base_kp = self.model.actuator_gainprm[:, 0].copy()
        self.base_floor_friction = self.model.geom_friction[self.floor_geom_id].copy()

        # gait-state
        self.prev_action = np.zeros(self.n_actuators)
        self.feet_air_time = np.zeros(4)
        self.last_contact = np.zeros(4, dtype=bool)

    def _randomize_domain(self):
        # joint "play"/backlash: random friction loss on each leg joint
        self.model.dof_frictionloss[self.leg_dof_start:self.leg_dof_end] = self.np_random.uniform(0.0, 0.05, self.n_actuators)

        # damping variance (mimics play/wear differences between servos)
        self.model.dof_damping[self.leg_dof_start:self.leg_dof_end] = self.base_damping * self.np_random.uniform(0.6, 1.4, self.n_actuators)

        # servo strength variance (kp) - some servos slightly weaker/stronger
        self.model.actuator_gainprm[:, 0] = self.base_kp * self.np_random.uniform(0.8, 1.2, self.n_actuators)
        self.model.actuator_biasprm[:, 1] = -self.model.actuator_gainprm[:, 0]  # keep position actuator internally consistent

        # floor friction variance (different surfaces)
        self.model.geom_friction[self.floor_geom_id] = self.base_floor_friction * self.np_random.uniform(0.7, 1.3, 3)

    def _update_orientation_filter(self, dt):
        accel = self.data.sensordata[0:3]
        gyro = self.data.sensordata[3:6]

        # accelerometer-gebaseerde hoek (betrouwbaar op lange termijn, ruisig per stap)
        accel_pitch = np.arctan2(-accel[0], np.sqrt(accel[1] ** 2 + accel[2] ** 2))
        accel_roll = np.arctan2(accel[1], accel[2])

        # gyro-integratie (nauwkeurig kortstondig, drift op lange termijn)
        gyro_pitch = self.filt_pitch + gyro[1] * dt
        gyro_roll = self.filt_roll + gyro[0] * dt

        # complementary filter: combineer beide
        self.filt_pitch = self.filter_alpha * gyro_pitch + (1 - self.filter_alpha) * accel_pitch
        self.filt_roll = self.filter_alpha * gyro_roll + (1 - self.filter_alpha) * accel_roll

    def _foot_contacts(self):
        # echte contacten uit de solver: welke onderbenen raken de vloer?
        contact = np.zeros(4, dtype=bool)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.floor_geom_id:
                other = c.geom2
            elif c.geom2 == self.floor_geom_id:
                other = c.geom1
            else:
                continue
            leg = self.foot_geom_to_leg.get(int(other))
            if leg is not None:
                contact[leg] = True
        return contact

    def _get_obs(self):
        joint_pos = self.data.qpos[7:7 + self.n_actuators]  # skip free joint (7 dof: 3 pos + 4 quat)
        joint_vel = self.data.qvel[6:6 + self.n_actuators]  # skip free joint (6 dof: 3 lin + 3 ang)
        accel = self.data.sensordata[0:3]  # accelerometer (m/s^2), like MPU6050
        gyro = self.data.sensordata[3:6]   # gyro (rad/s), like MPU6050
        return np.concatenate([joint_pos, joint_vel, accel, gyro, [self.filt_pitch, self.filt_roll], self.prev_action]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)  # start gehurkt, niet gestrekt
        self._randomize_domain()
        # small random perturbation on start pose so training sees varied starts
        self.data.qpos[7:7 + self.n_actuators] += self.np_random.uniform(-0.05, 0.05, self.n_actuators)
        self.data.ctrl[:] = self.nominal_ctrl  # servo's houden de hurk-pose vast tijdens het settelen
        mujoco.mj_forward(self.model, self.data)

        # laat de fysica even settelen (contactkrachten stabiliseren) voor we de filter starten -
        # direct na reset is de accelerometer nog een rare overgangswaarde
        for _ in range(300):
            mujoco.mj_step(self.model, self.data)

        self.step_count = 0
        self.prev_action = np.zeros(self.n_actuators)
        self.feet_air_time = np.zeros(4)
        self.last_contact = self._foot_contacts()
        accel = self.data.sensordata[0:3]
        self.filt_pitch = np.arctan2(-accel[0], np.sqrt(accel[1] ** 2 + accel[2] ** 2))
        self.filt_roll = np.arctan2(accel[1], accel[2])
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # actie = offset rond de nominale hurk-pose, begrensd door de ctrlrange van de servo's
        ctrl = np.clip(self.nominal_ctrl + action * self.ACTION_SCALE, self.ctrl_low, self.ctrl_high)
        self.data.ctrl[:] = ctrl

        # houd hetzelfde doel vast gedurende frame_skip physics-stappen (zoals een servo tussen twee PWM-updates)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.step_count += 1
        # filter draait op de control-frequentie, net zoals hij op de Pi met de MPU6050 zou draaien
        self._update_orientation_filter(self.control_dt)

        # action-rate: hoe hard verandert de actie t.o.v. de vorige stap? (trillen = duur, vloeiend = gratis)
        action_rate = np.sum(np.square(action - self.prev_action))
        self.prev_action = action.copy()

        obs = self._get_obs()

        # ---- toestand ----
        forward_vel = self.data.qvel[0]  # x-snelheid in wereld-frame
        torso_height = self.data.site_xpos[self.torso_site_id, 2]
        torso_x = self.data.site_xpos[self.torso_site_id, 0]
        torso_y = self.data.site_xpos[self.torso_site_id, 1]
        torso_xmat = self.data.xmat[self.torso_body_id].reshape(3, 3)
        up_world_z = torso_xmat[2, 2]  # 1.0 = perfect rechtop, 0 = op zij, -1 = ondersteboven
        tilt_penalty = 1.0 - up_world_z  # 0 als rechtop, oploopt tot 2 bij ondersteboven
        contact = self._foot_contacts()

        # ---- snelheid-tracking: beloon het HALEN van de doelsnelheid (niet "zoveel mogelijk snelheid") ----
        # alleen als hij redelijk rechtop staat (<~30 graden), anders kan hij tijdens een val een snelheidspiek pakken
        if tilt_penalty < 0.15:
            vel_err = forward_vel - self.TARGET_VEL
            tracking_reward = np.exp(-vel_err ** 2 / self.TRACKING_SIGMA)
        else:
            tracking_reward = 0.0

        # ---- zwaaiduur-beloning (legged_gym-stijl): bij het neerzetten van een voet kijken we hoe lang hij in de lucht was.
        # korter dan AIR_TIME_MIN (trillen/hupsen) = straf, langer = beloning, boven AIR_TIME_MAX niks extra.
        contact_filt = contact | self.last_contact  # debounce: één stap zonder contact telt nog als contact
        first_contact = (self.feet_air_time > 0.0) & contact_filt
        air_time_reward = np.sum((np.clip(self.feet_air_time, 0.0, self.AIR_TIME_MAX) - self.AIR_TIME_MIN) * first_contact)
        self.feet_air_time += self.control_dt
        self.feet_air_time[contact_filt] = 0.0
        self.last_contact = contact

        # ---- energie: echt servo-koppel i.p.v. actie^2, plus gewrichtssnelheid ----
        torque_penalty = np.sum(np.square(self.data.actuator_force))
        joint_vel_penalty = np.sum(np.square(self.data.qvel[self.leg_dof_start:self.leg_dof_end]))

        # ---- draai-straf: we trainen rechtdoor lopen, dus onnodig draaien (yaw) wordt afgestraft ----
        yaw_rate = self.data.sensordata[5]  # gyro z-as (rad/s)
        turn_penalty = abs(yaw_rate)

        # ---- sleep-straf: een voet die ECHT contact maakt en toch horizontaal beweegt, sleept i.p.v. te stappen.
        # framelinvel geeft de voetsnelheid in het wereld-frame, dus x/y zijn echt horizontaal.
        drag_penalty = 0.0
        for i in range(4):
            if contact[i]:
                foot_vel_world = self.data.sensordata[6 + i * 3: 6 + i * 3 + 3]
                drag_penalty += np.sqrt(foot_vel_world[0] ** 2 + foot_vel_world[1] ** 2)

        reward = (
            self.W_TRACKING * tracking_reward
            + self.W_AIR_TIME * air_time_reward
            - self.W_ACTION_RATE * action_rate
            - self.W_TORQUE * torque_penalty
            - self.W_JOINT_VEL * joint_vel_penalty
            - self.W_TILT * tilt_penalty
            - self.W_TURN * turn_penalty
            - self.W_DRAG * drag_penalty
        )

        # termination: fell over (te laag OF te scheef gekanteld - vangt een "duik" vroeg af)
        fell = torso_height < 0.12 or tilt_penalty > 0.5
        terminated = bool(fell)
        if fell:
            reward -= self.FALL_PENALTY

        # off the edge of the ground plane: dit is geen "slechte" toestand, dus truncation i.p.v. termination -
        # PPO bootstrapt dan gewoon de value van deze state door i.p.v. hem op 0 te zetten
        off_edge = abs(torso_x) > 4.5 or abs(torso_y) > 4.5
        truncated = bool(self.step_count >= self.max_steps or off_edge)

        info = {
            "forward_vel": float(forward_vel),
            "r_tracking": float(tracking_reward),
            "r_air_time": float(air_time_reward),
            "p_action_rate": float(action_rate),
            "p_torque": float(torque_penalty),
            "p_joint_vel": float(joint_vel_penalty),
            "p_tilt": float(tilt_penalty),
            "p_turn": float(turn_penalty),
            "p_drag": float(drag_penalty),
            "contacts": contact.copy(),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass
