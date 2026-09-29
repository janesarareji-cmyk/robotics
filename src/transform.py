import numpy as np

def dh_to_transform(theta_deg, d, a, alpha_deg):
    theta = np.deg2rad(theta_deg)
    alpha = np.deg2rad(alpha_deg)
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])

def build_transform_chain(dh_table):
    local = []
    cumulative = []
    current = np.eye(4)

    for joint in dh_table:
        t = dh_to_transform(joint.theta, joint.d, joint.a, joint.alpha)
        local.append(t)
        current = current @ t
        cumulative.append(current.copy())

    return local, cumulative
