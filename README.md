# RB5-850e + D405 Pick and Place 실습 (ROS 2 Humble)

테이블 위 물체(박스·캔·물병·공)를 **손목 카메라(RealSense D405)로 위에서 보고**, 협동로봇 **RB5-850e + OnRobot RG2** 로 집어 옮기는 실습입니다.

```
 1 초기위치 → 2 대기위치 → 3 물체인식 → 4 물체 위 이동 → 5 하강 → 6 잡기 → 7 상승
 → 8 하역 위치 이동 → 9 하강 → 10 놓기 → 11 상승 → (결과 확인) → 12 초기위치
```

로봇·카메라가 없어도 **가상 모드(sim)** 로 시나리오 전체를 똑같이 돌려 볼 수 있습니다. 집에서 설치하고 점검까지 끝내 오세요.

---

## 1. 사전 설치 (실습 전에 집에서)

**준비물:** Ubuntu **22.04** 노트북(가상머신도 가능, 단 RViz 가 느릴 수 있음) · 인터넷 · 디스크 여유 5 GB 이상

### 1-1. ROS 2 Humble 설치 (이미 있으면 건너뛰기)
공식 안내: <https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html> → `ros-humble-desktop` 설치

### 1-2. 실습 패키지 설치
```bash
cd ~
unzip edu_pick_ws.zip          # 받은 파일을 홈 폴더에 풀면 ~/edu_pick_ws 가 생깁니다
cd ~/edu_pick_ws
bash install.sh                # 10~20분. 중간에 sudo 비밀번호를 한 번 묻습니다
```
`install.sh` 가 하는 일: apt 패키지(MoveIt·ros2_control·RealSense 드라이버 …) 설치 → 레인보우 로봇 SDK 빌드 → 레인보우 ROS 2 드라이버 받기 + 실습용 패치 → 빌드 → `~/.bashrc` 에 `source` 줄 추가

### 1-3. 점검
```bash
source ~/.bashrc
bash ~/edu_pick_ws/check_setup.sh
```
마지막 줄이 **`✅ 모두 통과 — 실습 준비 완료`** 면 끝입니다. 가상 모드로 로봇이 물체 하나를 실제로 옮기는 것까지 자동으로 확인합니다.
`⚠ D405 연결 안 됨` 경고는 집에서는 정상입니다. ❌ 가 있으면 화면 내용을 캡처해 강사에게 보내 주세요.

---

## 2. 구성

| 패키지 | 내용 |
|---|---|
| `edu_pick_description` | 로봇 모델(URDF): 테이블 + RB5 + RG2 + D405. **`config/scene.yaml` = 실습 설정 파일(유일하게 고치는 파일)** |
| `edu_pick_moveit` | MoveIt 설정(경로 계획) |
| `edu_pick_perception` | `object_detector` 물체 인식 · `fake_camera` 가상 카메라(sim) |
| `edu_pick_task` | `pick_place` **12단계 시나리오 노드 — 실습의 주인공** |
| `edu_pick_bringup` | 전체 실행 launch · RG2 그리퍼 어댑터 |
| `edu_pick_msgs` | 인식 결과 메시지 `DetectedObject` |

```
 [D405 카메라 / 가상 카메라] ──점군──▶ [object_detector] ──/edu_pick/objects──▶ [pick_place]
                                                                                  │  ▲
                        /compute_ik · /move_action · /compute_cartesian_path ◀────┘  │
                        /execute_trajectory                                          │
                                 [move_group (MoveIt)] ──궤적──▶ [arm_controller] ──▶ RB5 (제어박스)
   [pick_place] ──/gripper_controller/follow_joint_trajectory──▶ [gripper_adapter] ──▶ RG2
```

**좌표계:** `world` = 로봇 베이스 바닥 중심 (x 앞, y 왼쪽, z 위). **`tool_tip`** = 그리퍼 손끝. z축이 접근 방향, x축이 손가락 닫히는 방향입니다. "위에서 집기" = `tool_tip` z축이 아래(-z)를 향하는 자세.

---

## 3. 실습 (2시간 수업 기준)

| 시간 | 내용 |
|---|---|
| 0:00–0:35 | **이론** — 노드·토픽·서비스·액션 / TF·URDF / MoveIt(IK·경로계획·직선경로) / 깊이로 물체 찾기 |
| 0:35–0:45 | 실습 1 — 로봇 모델과 TF |
| 0:45–1:10 | 실습 2 — 가상 모드로 시나리오 실행·관찰 |
| 1:10–1:25 | 실습 3 — 설정 바꿔 보기 |
| 1:25–1:55 | 실습 4 — **실제 로봇** (강사 진행·조별 순서) |
| 1:55–2:00 | 정리 |

### 실습 1 — 로봇 모델과 TF (10분)
```bash
ros2 launch edu_pick_description display.launch.py
```
- 슬라이더로 관절 6개를 움직여 보세요. `rg2_finger_joint1` 은 그리퍼(0 = 닫힘, 0.6 ≈ 100% 열림)입니다.
- RViz 의 TF 에서 `tool_tip`(손끝)과 `d405_link`(카메라)를 찾으세요. 카메라는 손끝에서 얼마나 떨어져 있나요?
  ```bash
  ros2 run tf2_ros tf2_echo world tool_tip     # 손끝 위치·자세
  ros2 run tf2_ros tf2_echo tool_tip d405_link # 손끝 → 카메라
  ```
- ❓ `shoulder + elbow + wrist1 = 90°` 로 맞추면 손끝이 어디를 보나요?

### 실습 2 — 가상 모드로 시나리오 실행 (25분)
**터미널 1** — 시스템 전체(가짜 하드웨어 + 가상 카메라 + 인식 + MoveIt + RViz)
```bash
ros2 launch edu_pick_bringup pick_place.launch.py
```
**터미널 2** — 시나리오. `--step` 이면 **움직이기 직전마다 멈춥니다.** RViz 의 반투명 '계획된 경로'를 보고 Enter 를 누르세요.
```bash
ros2 run edu_pick_task pick_place --step
```
**터미널 3** — 들여다보기
```bash
ros2 node list
ros2 topic echo /edu_pick/objects --once      # 인식 결과: 모양·위치·폭·방향
ros2 run rqt_image_view rqt_image_view /edu_pick/debug_image   # 인식 과정 그림
ros2 action list                               # 시나리오가 쓰는 액션들
```
확인할 것
- 3단계 출력: 물체마다 `round`/`box`, 폭, 높이. **큰 박스는 왜 `✕ 너무 넓음`인가요?** (RG2 최대 벌림 ≈ 11 cm)
- 박스를 집을 때 그리퍼가 회전합니다(`집기 방향 30°`). 왜 짧은 변 방향으로 닫아야 할까요?
- 4단계(물체 위 이동)는 **관절 경로**, 5단계(하강)는 **손끝 직선 경로**입니다. RViz 에서 차이를 보세요.
- `--count 3` 으로 여러 개를 옮겨 보세요. 두 번째 물체는 어디에 놓이나요?

**인식 원리 (`object_detector.py`)** — 학습 없이 깊이만으로 찾습니다.
1. 점군을 TF 로 `world` 좌표로 바꾼다
2. 높이 히스토그램에서 가장 많은 높이 = 테이블 면
3. 테이블보다 1.5 cm 이상 높은 점 = 물체 → 영상에서 덩어리(윤곽선)로 묶는다
4. 덩어리의 **윗면**에 최소 외접 사각형 → 가로·세로가 비슷하고 사각형을 덜 채우면(원은 78.5%) `round`, 아니면 `box`. 짧은 변 = 집을 폭과 방향

### 실습 3 — 설정 바꿔 보기 (15분)
`~/edu_pick_ws/src/edu_pick_description/config/scene.yaml` 을 고치고 **시나리오만 다시 실행**하면 됩니다(빌드 불필요).

| 바꿔 볼 것 | 항목 | 관찰 |
|---|---|---|
| 하역 위치 | `place.xy` | 닿지 않는 곳(예: `[0.9, 0.5]`)이면? → IK 실패 메시지 |
| 잡는 깊이 | `motion.grip_depth` | 0.005 로 하면? (가상 카메라가 '집힘'으로 볼까?) |
| 인식 기준 | `detection.min_height` | 0.10 으로 하면 낮은 박스가 사라짐 |
| 가상 물체 | `sim_objects` | 물체를 추가·회전 (`launch` 재시작 필요) |
| 속도 | `motion.velocity_scaling` | ⚠ 실기에서는 0.12 를 넘기지 말 것 |

### 실습 4 — 실제 로봇 (강사 진행)
**① Simulation 모드** — 제어박스에 접속하지만 팔은 움직이지 않습니다. 카메라 인식은 실제입니다.
```bash
ros2 launch edu_pick_bringup pick_place.launch.py mode:=real robot_ip:=<제어박스 IP>
ros2 run edu_pick_task pick_place --step
```
→ 펜던트 화면의 로봇이 따라 움직이는지, RViz 인식 결과가 실제 물체 위치와 맞는지 확인

**② 실제 구동** — 강사가 안전을 확인한 뒤에만
```bash
ros2 launch edu_pick_bringup pick_place.launch.py mode:=real robot_ip:=<제어박스 IP> cb_simulation:=false
ros2 run edu_pick_task pick_place --step
```

---

## 4. 안전 수칙 (실기)

1. **비상정지 버튼 담당자를 정하고, 그 사람은 실행 내내 버튼에 손을 둔다.**
2. 실기에서는 **항상 `--step`**. 각 이동 전에 RViz 의 '계획된 경로'를 보고 이상하면 Enter 대신 **Ctrl+C**.
   (Ctrl+C 는 진행 중 동작도 멈추고, 실제로 멈췄는지 관절값으로 확인해 알려 줍니다. "정지 확인 실패"가 나오면 즉시 비상정지)
3. 로봇 작업 반경(약 1 m) 안에 손·몸을 넣지 않는다. 물체를 옮겨 놓을 때는 **로봇이 초기위치에서 멈춘 상태**에서만.
4. 속도는 바꾸지 않는다. 하드웨어 상한 0.5 rad/s 가 걸려 있어서, 더 빠르게 계획하면 추종 오차로 중간에 멈춘다.
5. 이 시스템은 **카메라가 본 물체를 장애물로 피하지 않는다**(테이블과 로봇 자신만 피함). 작업 영역에는 실습 물체만 둔다.

---

## 5. 강사용 — 실습실 준비 체크리스트

**네트워크 (가장 중요)**
- [ ] **유선 LAN** 으로 연결한다. 와이파이에서는 서보 명령이 지연에 끊겨 10~20초 만에 로봇이 멈춘 이력이 있다. 무선을 써야 하면 `servo_t2:=0.1` 을 붙인다.
- [ ] 노트북 IP 를 제어박스와 같은 대역으로 설정 → `ping <제어박스 IP>`
- [ ] 한 번에 **한 노트북만** 접속한다. 학생 교대 시 앞 사람의 launch 를 반드시 Ctrl+C 로 끈다.

**로봇·펜던트**
- [ ] E-stop 해제 · 로봇 활성화(펜던트) · **모드 확인**. 펜던트로 활성화할 때 모드가 Real 로 바뀌어 있던 이력이 있다.
- [ ] 그리퍼가 RG2 이고 펜던트에서 열고 닫기가 된다.

**장면 맞추기 (`scene.yaml`)**
- [ ] `table.top_z` — 테이블 윗면 높이(로봇 베이스 바닥 기준). 로봇이 테이블 위에 직접 고정돼 있으면 0
- [ ] `poses.observe` — ① Simulation 모드로 대기위치까지 보낸 뒤 RViz 카메라 점군에 테이블 작업 영역이 다 보이는지 확인. 카메라 높이는 **0.3~0.45 m**(D405 는 50 cm 를 넘으면 깊이 오차가 커진다)
- [ ] `detection.roi_x/roi_y` — 대기위치에서 카메라가 보는 범위 **안쪽**으로
- [ ] `place.xy` — 실제로 비어 있는 자리, 물체를 여러 개 옮기면 `step_xy` 만큼 옆으로 놓는다
- [ ] 물체: 그리퍼 폭(≈ 9.5 cm 이하) · 무광택(반짝이는 캔·투명 병은 깊이가 잘 안 잡힌다 → 무광 테이프를 붙이면 된다) · 테이블과 높이 차 2 cm 이상

**자세 도구 `go_pose`** — 첫 구동 확인·자세 맞추기용 (계획을 보여 주고 Enter 후 실행)
```bash
ros2 run edu_pick_task go_pose --show            # 지금 관절값(도)
ros2 run edu_pick_task go_pose --jog wrist3 5    # 관절 하나를 +5° (±30° 이내)
ros2 run edu_pick_task go_pose home              # scene.yaml 의 home / observe 자세로
```
카메라 없이 팔만 확인할 때는 launch 에 `camera:=false`.

**첫 실행 순서 (수업 전 강사가 반드시 한 번)**
0. `mode:=real camera:=false` (Simulation) → `go_pose --jog wrist3 5` → `go_pose home` → `go_pose observe` 로 펜던트 가상 로봇의 경로 확인 → Real 에서 `--jog` 부터
1. `mode:=real` (Simulation) → `pick_place --step` 으로 12단계 전체. 인식 위치와 실제 물체 위치 차이를 확인한다(1 cm 이내면 양호).
2. `cb_simulation:=false` → `--step` 으로 **물체 없이** 한 번 → 물체 놓고 한 번
3. 잡는 높이가 안 맞으면 `motion.grip_depth`, 손끝 위치가 체계적으로 어긋나면 `tool_tip_offset`·`camera_mount` 를 확인한다.

---

## 6. 문제 해결

| 증상 | 원인·조치 |
|---|---|
| `앞 실행의 노드가 남아 있습니다` | 다른 터미널의 launch 를 Ctrl+C. 그래도 남으면 터미널을 모두 닫고 다시 실행 |
| `제어박스 … 에 접속할 수 없습니다` | 랜선·IP·`ping`. 로봇 없이 하려면 `mode:=sim` |
| `IK 해 없음` | 목표 위치가 팔이 닿지 않는 곳이거나 테이블과 충돌. `place.xy`·물체 위치 확인 |
| `직선 경로를 xx% 밖에 못 만듦` | 하강 중 손목 한계·충돌. 물체를 로봇 정면 쪽으로 옮긴다 |
| 인식 결과 0개 | 대기위치에서 물체가 시야 밖이거나 `roi` 밖. `rqt_image_view /edu_pick/debug_image` 확인 |
| `✕ 가장자리에 잘림` | 물체가 시야나 인식 영역 경계에 걸려 크기를 알 수 없음. 물체를 안쪽으로 |
| 실행 중 `Unexpected robot state changed` | 네트워크 지연. 유선 연결, 무선이면 `servo_t2:=0.1` |
| move_group 로그의 `No 3D sensor plugin(s) defined` | 정상(이 실습은 옥토맵을 쓰지 않는다) |

---

## 7. 한계 (알고 쓰기)

- **가상 모드는 물리 시뮬레이션이 아닙니다.** "그리퍼를 닫을 때 손끝 사이에 물체가 있었나"만 봅니다. 미끄러짐·무게·힘은 실기에서만 확인됩니다.
- RG2 는 **쥐었는지 알려 주지 않습니다**(제어박스가 명령 접수만 응답). 그래서 대기위치에서 다시 찍어 **원래 자리에서 사라졌는지**로 성공을 판단합니다.
- 공처럼 낮고 둥근 물체는 가운데 높이에서 잡도록 되어 있지만 굴러가기 쉽습니다.

## 8. 출처·라이선스

- 실습 코드: Apache-2.0
- RB5 모델·드라이버: [RainbowRobotics/rbpodo_ros2](https://github.com/RainbowRobotics/rbpodo_ros2) (Apache-2.0, `install.sh` 가 받음) · SDK [rbpodo](https://github.com/RainbowRobotics/rbpodo) (Apache-2.0)
- RG2 메시: [AndrejOrsula/ur5_rg2_ign](https://github.com/AndrejOrsula/ur5_rg2_ign) (BSD)
- `patches/` : 레인보우 드라이버 수정 3건(서보 유지시간 · 관절 속도 상한 · 그리퍼 명령 직후 끊김) — 실기 시험에서 확인한 문제를 고친 것
