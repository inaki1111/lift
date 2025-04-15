from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
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
    print("Posición del objeto (cubo):")
    print(cube_pos_w)
    print("Posición del efector final:")
    print(ee_w)
    
    # Imprimir orientación del efector final (si está disponible)
    if hasattr(ee_frame.data, "target_quat_w"):
        ee_orient = ee_frame.data.target_quat_w[..., 0, :]
        print("Orientación del efector final (cuaternión):")
        print(ee_orient)
    else:
        ee_orient = None
        print("No se encontró información de orientación en el efector final.")
    
    # Intentar imprimir la orientación del objeto (cubo)
    # Se asume que la orientación se encuentra en root_state_w[:, 3:7] si está disponible
    if hasattr(object.data, "root_state_w"):
        cube_orient = object.data.root_state_w[:, 3:7]
        print("Orientación del objeto (cubo) (cuaternión):")
        print(cube_orient)
    else:
        cube_orient = None
        print("No se encontró información de orientación en el objeto.")
    
    # Calcular la distancia euclídea entre el objeto y el efector final.
    pos_error = torch.norm(cube_pos_w - ee_w, dim=1)
    print("Error en posición (distancia euclídea) entre efector y objeto:")
    print(pos_error)
    
    # Calcular error angular entre las orientaciones, si se dispone de ambas
    if ee_orient is not None and cube_orient is not None:
        # Se calcula el producto escalar entre cuaterniones para cada ambiente.
        dot_product = torch.abs(torch.sum(ee_orient * cube_orient, dim=1))
        # Asegurarse de que el valor esté en el rango [0, 1]
        dot_product = torch.clamp(dot_product, max=1.0)
        # Error angular (en radianes): 2*arccos(|dot(q1,q2)|)
        orient_error = 2 * torch.acos(dot_product)
        print("Error angular (radianes) entre efector y objeto:")
        print(orient_error)
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
