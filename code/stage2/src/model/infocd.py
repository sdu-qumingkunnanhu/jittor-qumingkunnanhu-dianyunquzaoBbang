import jittor as jt

from src.utils.pointops import nearest_neighbor_distance


def calc_cd_like_InfoV2(p1: jt.Var, p2: jt.Var):
    d1, d2 = nearest_neighbor_distance(p1, p2)
    d1 = jt.maximum(d1, 1e-9)
    d2 = jt.maximum(d2, 1e-9)

    exp_d1 = jt.exp(-0.5 * d1)
    exp_d2 = jt.exp(-0.5 * d2)
    distances1 = -jt.log(exp_d1 / ((exp_d1 + 1e-7).sum(dim=-1).unsqueeze(-1) ** 1e-7))
    distances2 = -jt.log(exp_d2 / ((exp_d2 + 1e-7).sum(dim=-1).unsqueeze(-1) ** 1e-7))
    return (distances1.sum() + distances2.sum()) / (2 * p1.shape[0])
