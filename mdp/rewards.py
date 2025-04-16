from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Reward the agent for reaching the object using tanh-kernel.
    
    Se imprimen la posición y orientación del efector final, la posición y orientación del objeto,
    y se calcula un error angular entre las orientaciones (si se dispone de estos datos).
    """
    # Extraer elementos (para type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # Posición del objeto (cubo) en el frame mundial (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    
    # Posición del efector final extraída del target (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    
    # Imprimir posiciones
    #print("Posición del objeto (cubo):")
    #print(cube_pos_w)
    #print("Posición del efector final:")
    #print(ee_w)
    
    # Imprimir orientación del efector final (si está disponible)
    if hasattr(ee_frame.data, "target_quat_w"):
        ee_orient = ee_frame.data.target_quat_w[..., 0, :]
        #print("Orientación del efector final (cuaternión):")
        #print(ee_orient)
    else:
        ee_orient = None
        #print("No se encontró información de orientación en el efector final.")
    
    # Intentar imprimir la orientación del objeto (cubo)
    # Se asume que la orientación se encuentra en root_state_w[:, 3:7] si está disponible
    if hasattr(object.data, "root_state_w"):
        cube_orient = object.data.root_state_w[:, 3:7]
        #print("Orientación del objeto (cubo) (cuaternión):")
        #print(cube_orient)
    else:
        cube_orient = None
        #print("No se encontró información de orientación en el objeto.")
    
    # Calcular la distancia euclídea entre el objeto y el efector final.
    pos_error = torch.norm(cube_pos_w - ee_w, dim=1)
    #print("Error en posición (distancia euclídea) entre efector y objeto:")
    #print(pos_error)
    
    # Calcular error angular entre las orientaciones, si se dispone de ambas
    if ee_orient is not None and cube_orient is not None:
        # Se calcula el producto escalar entre cuaterniones para cada ambiente.
        dot_product = torch.abs(torch.sum(ee_orient * cube_orient, dim=1))
        # Asegurarse de que el valor esté en el rango [0, 1]
        dot_product = torch.clamp(dot_product, max=1.0)
        # Error angular (en radianes): 2 * arccos(|dot(q1,q2)|)
        orient_error = 2 * torch.acos(dot_product)
        #print("Error angular (radianes) entre efector y objeto:")
        #print(orient_error)
    else:
        orient_error = None
    
    # Se retorna la recompensa basada en la distancia en posición (usando tanh para suavizar el error)
    return 1 - torch.tanh(pos_error / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # Extraer elementos para type-hinting
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # Calcular la posición deseada en el frame mundial usando la transformación del robot.
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        des_pos_b
    )
    # Calcular la distancia entre la posición deseada y la posición del objeto.
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # Recompensa solo si el objeto está elevado por encima de la altura mínima.
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def gripper_close_reward(
    env,
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.5,
    near_thresh: float = 0.7  # Distancia mínima para considerar que el efector está cerca del objeto.
) -> torch.Tensor:
    import math

    # Obtener el robot
    robot = env.scene[gripper_cfg.name]
    
    # Definir los nombres y extraer los índices de las articulaciones del gripper.
    left_joint_name = "robotiq_85_left_knuckle_joint"
    right_joint_name = "robotiq_85_right_knuckle_joint"
    left_index = robot.data.joint_names.index(left_joint_name)
    right_index = robot.data.joint_names.index(right_joint_name)
    
    # Extraer las posiciones actuales de las articulaciones del gripper
    left_value = robot.data.joint_pos[:, left_index]
    right_value = robot.data.joint_pos[:, right_index]
    
    # Combinar en un tensor de forma (num_envs, 2)
    gripper_state = torch.stack([left_value, right_value], dim=1)
    
    # Definir el objetivo de cierre (41° convertidos a radianes)
    target_close = torch.tensor([math.radians(41.0), math.radians(41.0)], device=gripper_state.device)
    
    # Calcular el error en el gripper (norma de la diferencia)
    error = torch.norm(gripper_state - target_close, dim=1)
    
    # Obtener la posición actual del efector final:
    ee_frame = env.scene["ee_frame"]  # Asegúrate de que el nombre registrado en la escena sea "ee_frame"
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # [num_envs, 3]
    
    # Obtener la posición del objeto (cubo)
    object_state = env.scene["object"]
    cube_pos = object_state.data.root_pos_w  # [num_envs, 3]
    
    # Calcular la distancia entre el efector y el objeto
    ee_obj_distance = torch.norm(ee_pos - cube_pos, dim=1)
    
    # Definir una máscara: solo consideramos el reward de gripper si el efector está cerca del objeto
    near_object = ee_obj_distance < near_thresh  # Esto es un tensor booleano
    
    # Calcular el reward del gripper de forma binaria (o gradual) únicamente cuando esté cerca
    # (1 - tanh(error / threshold)) dará valores cercanos a 1 cuando el error sea pequeño.
    gripper_reward = 1 - torch.tanh(error / threshold)
    
    # Si no está cerca, forzamos la recompensa a 0 para el gripper (o podrías considerar otro valor neutro).
    reward = torch.where(near_object, gripper_reward, torch.zeros_like(gripper_reward))
    
    # Opcional: imprimir datos para depuración
    # print("gripper_state:", gripper_state)
    # print("target_close:", target_close)
    # print("Error en gripper (norma) =", error)
    # print("Distancia EE-objeto =", ee_obj_distance)
    # print("Recompensa de cierre de gripper =", reward)
    
    return reward
