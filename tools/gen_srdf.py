#!/usr/bin/env python3
"""[강사용 도구] URDF → SRDF 생성 (플래닝 그룹 + ACM 자동 산출).

MoveIt Setup Assistant 의 collision matrix 계산을 스크립트로 재현한다.
GUI 대신 스크립트인 이유: scene.yaml(그리퍼·카메라 장착)이나 URDF 를 바꾸면
다시 만들어야 하는데, 명령 한 줄로 되풀이할 수 있다.

ACM 분류(Setup Assistant 와 동일 규칙):
  Adjacent  : 관절로 직접 연결된 링크쌍 → 항상 접촉, 무시
  Always    : 무작위 자세 전부에서 충돌 → 모델 자체 겹침, 무시
  Never     : 무작위 자세 어디서도 충돌 안 함 → 검사 불필요, 무시
  Default   : 시작 자세에서 이미 충돌 → 무시
  (그 외 = 실제로 충돌 가능 → 검사 대상으로 남긴다)

사용:
  xacro src/edu_pick_description/urdf/edu_pick.urdf.xacro > /tmp/edu.urdf
  python3 tools/gen_srdf.py /tmp/edu.urdf src/edu_pick_moveit/config/edu_pick.srdf 10000
  OMX 판:
  xacro src/edu_pick_description/urdf/edu_pick_omx.urdf.xacro > /tmp/omx.urdf
  python3 tools/gen_srdf.py /tmp/omx.urdf src/edu_pick_moveit/config/omx/edu_pick_omx.srdf 10000
  (pip 의존: trimesh yourdfpy python-fcl — 학생은 실행할 필요 없음)
"""
import os
import sys
from collections import defaultdict

import numpy as np
import trimesh
import yourdfpy
from ament_index_python.packages import get_package_share_directory

N_SAMPLES = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
ALWAYS_FRAC = 0.95      # 이 비율 이상 충돌 → Always


def res(fname):
    f = fname
    if f.startswith("package://"):
        rest = f[len("package://"):]
        pkg, _, tail = rest.partition("/")
        try:
            return os.path.join(get_package_share_directory(pkg), tail)
        except Exception:
            return f
    return f


urdf_path, out_path = sys.argv[1], sys.argv[2]
# build_collision_scene_graph=True 가 있어야 collision_scene 이 만들어진다
# (기본 False → collision_scene 이 None). force_collision_mesh=False 로 해야
# collision geometry 가 없는 링크를 visual 로 대체하지 않는다.
robot = yourdfpy.URDF.load(
    urdf_path, filename_handler=res,
    load_collision_meshes=True, build_collision_scene_graph=True,
    force_collision_mesh=False,
)

# ── 링크별 충돌 mesh 를 링크 프레임에 baking ──────────────────────────────
# ⚠ visual 이 아니라 collision_scene 을 쓴다. MoveIt 은 URDF 의 collision
#    geometry 로 검사하므로, ACM 을 visual 로 산출하면 MoveIt 이 실제로 보는
#    형상과 어긋난다(collision mesh 가 더 두꺼우면 "Never" 로 끈 쌍이 실제로는 충돌).
robot.update_cfg({j: 0.0 for j in robot.actuated_joint_names})
scene = robot.collision_scene
geom_of_link = defaultdict(list)
parents = scene.graph.transforms.parents
for gname in scene.geometry:
    node = gname
    while node in parents and node not in robot.link_map:
        node = parents[node]
    if node in robot.link_map:
        geom_of_link[node].append(gname)

link_mesh = {}
for link, gnames in geom_of_link.items():
    inv = np.linalg.inv(robot.get_transform(link))
    parts = []
    for g in gnames:
        T = scene.graph.get(g)[0]
        m = scene.geometry[g].copy()
        if len(m.faces) == 0:
            continue
        m.apply_transform(inv @ T)      # 링크 프레임 기준으로 baking
        parts.append(m)
    if parts:
        link_mesh[link] = trimesh.util.concatenate(parts)

links = sorted(link_mesh)
print(f"충돌 대상 링크 {len(links)}개")

mgr = trimesh.collision.CollisionManager()
for l in links:
    mgr.add_object(l, link_mesh[l])

# ── Adjacent: 관절로 직접 연결된 쌍 ───────────────────────────────────────
adjacent = set()
for jn, j in robot.joint_map.items():
    p, c = j.parent, j.child
    if p in link_mesh and c in link_mesh:
        adjacent.add(frozenset((p, c)))
# fixed joint 로 이어진 링크는 상위 실체 링크로 승격돼 인접성이 끊길 수 있어
# 링크 트리상 2-hop(같은 부모를 공유)도 인접 취급하지 않는다(Setup Assistant 동일).

def set_cfg(cfg):
    robot.update_cfg(cfg)
    for l in links:
        mgr.set_transform(l, np.array(robot.get_transform(l), float))


def collide_pairs():
    _, names = mgr.in_collision_internal(return_names=True)
    return {frozenset(p) for p in names}


# ── 시작 자세(Default) ────────────────────────────────────────────────────
zero = {j: 0.0 for j in robot.actuated_joint_names}
set_cfg(zero)
default_hits = collide_pairs() - adjacent
print(f"시작자세 충돌쌍(Default) {len(default_hits)}")

# ── 무작위 샘플링 ─────────────────────────────────────────────────────────
lims = {}
for j in robot.actuated_joint_names:
    jj = robot.joint_map[j]
    lo = jj.limit.lower if jj.limit and jj.limit.lower is not None else -np.pi
    hi = jj.limit.upper if jj.limit and jj.limit.upper is not None else np.pi
    lims[j] = (lo, hi)

rng = np.random.default_rng(0)
count = defaultdict(int)
for i in range(N_SAMPLES):
    set_cfg({j: rng.uniform(*lims[j]) for j in robot.actuated_joint_names})
    for p in collide_pairs():
        count[p] += 1
    if (i + 1) % 2000 == 0:
        print(f"  샘플 {i+1}/{N_SAMPLES}")

all_pairs = {frozenset((a, b)) for i, a in enumerate(links) for b in links[i + 1:]}
rows = []          # (link1, link2, reason)
kept = []
for pr in sorted(all_pairs, key=lambda s: sorted(s)):
    a, b = sorted(pr)
    n = count.get(pr, 0)
    if pr in adjacent:
        rows.append((a, b, "Adjacent"))
    elif n >= N_SAMPLES * ALWAYS_FRAC:
        rows.append((a, b, "Always"))
    elif n == 0:
        rows.append((a, b, "Never"))
    elif pr in default_hits:
        rows.append((a, b, "Default"))
    else:
        kept.append((a, b, n))

print(f"\nACM: 무시 {len(rows)}쌍 / 검사유지 {len(kept)}쌍 (전체 {len(all_pairs)})")
print("검사 유지되는(=실제 충돌 가능) 쌍 상위:")
for a, b, n in sorted(kept, key=lambda t: -t[2])[:12]:
    print(f"  {a:<22}{b:<22}{n/N_SAMPLES*100:5.1f}% 샘플에서 충돌")

# ── SRDF 출력 ─────────────────────────────────────────────────────────────
# 팔 그룹 = link0 → tool_tip(손끝). MoveIt 의 목표는 tool_tip 프레임 자세로 준다.
ARM_CHAIN = ("link0", "tool_tip")
# 팔 사슬 밖의 움직이는 관절(그리퍼 손가락) = passive — MoveIt 이 계획하지 않는다
chain, link = set(), ARM_CHAIN[1]
child_joint = {j.child: j for j in robot.joint_map.values()}
while link != ARM_CHAIN[0] and link in child_joint:
    chain.add(child_joint[link].name)
    link = child_joint[link].parent
passive = [n for n, j in robot.joint_map.items() if j.type != "fixed" and n not in chain]   # mimic 관절 포함
# 그리퍼 턱끼리: 같은 링크에 붙은 손가락 관절(passive)의 자식 링크 쌍은 닫으면 서로 닿는 것이 정상
#   (OMX-F 의 두 턱은 완전히 닫으면 끝이 맞닿는다) → 검사에서 뺀다
listed = {frozenset((a, b)) for a, b, _ in rows}
fingers = defaultdict(list)
for jn in passive:
    j = robot.joint_map[jn]
    if j.child in link_mesh:
        fingers[j.parent].append(j.child)
for kids in fingers.values():
    for i, a in enumerate(kids):
        for b in kids[i + 1:]:
            if frozenset((a, b)) not in listed:
                rows.append((*sorted((a, b)), "Gripper"))
rows.sort(key=lambda r: (r[0], r[1]))

L = ['<?xml version="1.0" encoding="UTF-8"?>',
     "<!-- 자동 생성: tools/gen_srdf.py — 직접 편집하지 말 것.",
     f"     무작위 {N_SAMPLES} 자세로 충돌 무시 쌍(ACM)을 산출. 로봇·그리퍼·카메라 형상을 바꾸면 재생성. -->",
     f'<robot name="{robot.robot.name}">',
     '    <group name="arm">',
     f'        <chain base_link="{ARM_CHAIN[0]}" tip_link="{ARM_CHAIN[1]}"/>',
     "    </group>",
     '    <virtual_joint name="fixed_base" type="fixed" parent_frame="world" child_link="world"/>',
     *[f'    <passive_joint name="{j}"/>' for j in passive],
     "    <!-- 충돌 무시 쌍 (Adjacent=붙어있음 / Always=항상겹침 / Never=절대안닿음 / Default=초기자세에서겹침 / Gripper=그리퍼 턱끼리) -->"]
for a, b, why in rows:
    L.append(f'    <disable_collisions link1="{a}" link2="{b}" reason="{why}"/>')
L.append("</robot>")
with open(out_path, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"\n생성: {out_path}")
