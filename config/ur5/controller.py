import math
import torch

def control_gripper(ur5, open=True):
    # Nombres de los joints de la garra
    left_joint_name = "robotiq_85_left_knuckle_joint"
    right_joint_name = "robotiq_85_right_knuckle_joint"

    # Obtener índices desde la lista de nombres
    left_index = ur5.data.joint_names.index(left_joint_name)
    right_index = ur5.data.joint_names.index(right_joint_name)

    if open:
        target_left = 0.0
        target_right = 0.0
    else:
        # Convertimos los grados a radianes:
        target_left = math.radians(41.0)
        target_right = math.radians(41.0)

    # Crear tensor con los valores (asegurando tipo float)
    gripper_target = torch.tensor([[target_left, target_right]], dtype=torch.float32, device=ur5.device)
    joint_ids = torch.tensor([left_index, right_index], device=ur5.device)

    # Mensaje de depuración para verificar los targets
    print(f"Setting gripper targets -> Left: {target_left:.4f} rad, Right: {target_right:.4f} rad")

    ur5.set_joint_position_target(gripper_target, joint_ids=joint_ids)