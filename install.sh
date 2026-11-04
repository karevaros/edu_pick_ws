#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Pick and place 실습 — 사전 설치 스크립트
#
#   전제: Ubuntu 22.04 (데스크톱) · 인터넷 · sudo 비밀번호 — ROS 2 가 없으면 이 스크립트가 설치한다
#   사용:  cd ~/edu_pick_ws && bash install.sh
#   끝나면: bash check_setup.sh  (설치가 제대로 됐는지 자동 점검)
#   몇 번 다시 실행해도 된다 — 이미 된 단계는 건너뛴다.
#
#   하는 일
#     [0] ROS 2 Humble 설치 (없을 때만 · 공식 apt 저장소 · 약 3 GB 내려받음)
#     [1] apt 패키지 설치 (MoveIt·ros2_control·RealSense 드라이버·OpenCV …) + RealSense USB 권한(udev) 규칙
#     [2] 레인보우 로봇 C++ SDK(rbpodo) 빌드 → ~/edu_pick_ws/thirdparty/install (sudo 불필요)
#     [3] 레인보우 ROS 2 드라이버(rbpodo_ros2) 받기 + 실습용 패치 3개 적용
#     [4] 워크스페이스 빌드
#
#   옵션:  SKIP_APT=1  apt 단계를 건너뜀(이미 다 설치된 PC)
#          NO_BASHRC=1 ~/.bashrc 에 source 줄을 추가하지 않음
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 버전 고정 — 실기에서 검증한 버전. 바꾸면 패치가 안 맞을 수 있다.
RBPODO_SDK_TAG="v0.16.14"
RBPODO_ROS2_COMMIT="5e8294a985e7ce5e20e70564c2681130afc5f502"

# apt 가 설치 중에 질문(키보드 배치·시간대 등)을 띄우면 스크립트가 멈춘다 → 묻지 않고 기본값으로
#   (sudo 는 환경변수를 지우므로 명령마다 붙인다)
APT="sudo DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold"

step() { echo; echo "━━━ $* ━━━"; }
die()  { echo "❌ $*" >&2; exit 1; }

[ "$(id -u)" -ne 0 ] || die "sudo 없이 일반 사용자로 실행하세요: bash install.sh"
. /etc/os-release
[ "${VERSION_CODENAME:-}" = "jammy" ] || die "Ubuntu 22.04(jammy) 가 아닙니다(지금: ${PRETTY_NAME:-알 수 없음}). ROS 2 Humble 은 22.04 에서만 동작합니다."
case "$WS" in *" "*) die "설치 폴더 경로에 빈칸이 있습니다: $WS — 홈 폴더 바로 아래 edu_pick_ws 로 옮기세요." ;; esac

# ── [0] ROS 2 Humble ─────────────────────────────────────────────────────────
# 공식 설치 절차(docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)를 그대로 따른다.
step "[0/4] ROS 2 Humble"
# 앞 설치가 중간에 끊겨 반쯤 설치된 패키지가 있으면 마저 끝낸다
#   (dpkg --audit 는 root 가 아니면 잠금 파일을 못 읽어 '이상 없음'처럼 빈 결과를 낸다 → 쓰지 않는다)
half_installed() {
  [ -n "$(ls -A /var/lib/dpkg/updates 2>/dev/null)" ] ||
  dpkg-query -W -f='${db:Status-Abbrev}\n' 2>/dev/null | grep -qE '^.[UFHW]'
}
if [ "${SKIP_APT:-0}" != "1" ] && half_installed; then
  echo "  중간에 끊긴 설치를 마저 끝냅니다"
  sudo DEBIAN_FRONTEND=noninteractive dpkg --configure -a
fi
if [ -f /opt/ros/humble/setup.bash ]; then
  echo "  이미 설치됨 — 건너뜀"
elif [ "${SKIP_APT:-0}" = "1" ]; then
  die "ROS 2 Humble 이 없는데 SKIP_APT=1 입니다."
else
  echo "  ROS 2 Humble 을 설치합니다(약 3 GB · 인터넷 속도에 따라 10~30분)."
  $APT update
  $APT install locales software-properties-common curl
  sudo locale-gen en_US en_US.UTF-8
  sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
  sudo add-apt-repository -y universe
  ROS_APT_SOURCE_VERSION=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
                           | grep -F '"tag_name"' | awk -F'"' '{print $4}')
  [ -n "$ROS_APT_SOURCE_VERSION" ] || die "ROS 저장소 정보를 받지 못했습니다 — 인터넷 연결을 확인하세요."
  curl -fsSL -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.jammy_all.deb"
  sudo dpkg -i /tmp/ros2-apt-source.deb
  $APT update
  $APT upgrade
  $APT install ros-humble-desktop
  [ -f /opt/ros/humble/setup.bash ] || die "ROS 2 Humble 설치에 실패했습니다 — 위 메시지를 확인하세요."
fi

# ── [1] apt ──────────────────────────────────────────────────────────────────
APT_PKGS=(
  ros-humble-desktop ros-humble-moveit
  ros-humble-ros2-control ros-humble-ros2-controllers
  ros-humble-xacro ros-humble-joint-state-publisher ros-humble-joint-state-publisher-gui
  ros-humble-cv-bridge ros-humble-tf2-ros ros-humble-tf2-geometry-msgs
  ros-humble-realsense2-camera ros-humble-rqt-image-view
  python3-opencv python3-numpy python3-yaml python3-colcon-common-extensions
  git cmake build-essential
  fonts-noto-cjk            # RViz 의 한글 표시 이름 (데스크톱에는 보통 이미 있다)
)
step "[1/4] apt 패키지 + RealSense USB 권한"
if [ "${SKIP_APT:-0}" = "1" ]; then
  echo "  SKIP_APT=1 — 건너뜀"
else
  MISSING=()
  for p in "${APT_PKGS[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p"); done
  if [ ${#MISSING[@]} -eq 0 ]; then
    echo "  전부 설치돼 있음"
  else
    echo "  설치할 패키지: ${MISSING[*]}"
    $APT update
    $APT install "${MISSING[@]}"
  fi
  # RealSense 카메라를 sudo 없이 쓰기 위한 USB 권한 규칙 (없으면 실습실에서 카메라가 안 잡힌다)
  if [ ! -f /etc/udev/rules.d/99-realsense-libusb.rules ]; then
    curl -fsSL -o /tmp/99-realsense-libusb.rules \
      https://raw.githubusercontent.com/IntelRealSense/librealsense/v2.58.2/config/99-realsense-libusb.rules
    sudo cp /tmp/99-realsense-libusb.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules && sudo udevadm trigger
    echo "  RealSense USB 권한 규칙 설치"
  else
    echo "  RealSense USB 권한 규칙 이미 있음"
  fi
fi

# ── [2] 레인보우 SDK ─────────────────────────────────────────────────────────
# apt 에 없어서 소스로 빌드한다. 없으면 rbpodo_hardware 가 빌드에서 '조용히' 빠지고
# 실기 접속 때에야 "plugin rbpodo_hardware/RBPodoHardwareInterface not found" 로 드러난다.
step "[2/4] 레인보우 C++ SDK(rbpodo $RBPODO_SDK_TAG)"
SDK_SRC="$WS/thirdparty/rbpodo"
SDK_PREFIX="$WS/thirdparty/install"
if [ -f "$SDK_PREFIX/lib/librbpodo.a" ]; then
  echo "  이미 설치됨 — 건너뜀 ($SDK_PREFIX)"
else
  mkdir -p "$WS/thirdparty"
  [ -d "$SDK_SRC/.git" ] || git clone --depth 1 --branch "$RBPODO_SDK_TAG" \
      https://github.com/RainbowRobotics/rbpodo.git "$SDK_SRC"
  cmake -S "$SDK_SRC" -B "$SDK_SRC/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$SDK_PREFIX"
  cmake --build "$SDK_SRC/build" -j"$(nproc)"
  cmake --install "$SDK_SRC/build"
fi

# ── [3] 레인보우 ROS 2 드라이버 + 패치 ───────────────────────────────────────
#   01 servo_t2     : 서보 유지시간을 늘릴 수 있게 함 (네트워크 지연에 끊기는 문제)
#   02 speed_limit  : 관절 속도 상한 max_joint_velocity (속도바는 서보 명령에 안 먹는다)
#   03 script_grace : 그리퍼 명령 직후 팔 하드웨어가 끊기는 문제
step "[3/4] 레인보우 ROS 2 드라이버(rbpodo_ros2) + 패치"
VENDOR="$WS/src/thirdparty/rbpodo_ros2"
if [ ! -d "$VENDOR/.git" ]; then
  mkdir -p "$WS/src/thirdparty"
  git clone https://github.com/RainbowRobotics/rbpodo_ros2.git "$VENDOR"
  git -C "$VENDOR" checkout -q "$RBPODO_ROS2_COMMIT"
fi
PATCHES=("$WS/patches/"[0-9][0-9]-rbpodo_hardware_*.patch)
if git -C "$VENDOR" diff --quiet -- rbpodo_hardware; then
  for P in "${PATCHES[@]}"; do git -C "$VENDOR" apply "$P"; echo "  패치 적용: $(basename "$P")"; done
elif git -C "$VENDOR" apply --reverse --check "${PATCHES[-1]}" 2>/dev/null; then
  echo "  패치 이미 적용됨"
else
  die "rbpodo_hardware 에 패치와 다른 변경이 있습니다. 되돌린 뒤 다시 실행: git -C $VENDOR checkout -- rbpodo_hardware"
fi
# 실습에 안 쓰는 벤더 패키지는 빌드에서 뺀다(의존성·시간 절약)
for d in rbpodo_bringup rbpodo_moveit_config; do touch "$VENDOR/$d/COLCON_IGNORE"; done

# ── [4] 빌드 ─────────────────────────────────────────────────────────────────
step "[4/4] colcon build"
cd "$WS"
set +u; source /opt/ros/humble/setup.bash; set -u
# 벤더 패키지: 일반 빌드 (⚠ --symlink-install 을 섞으면 캐시가 깨진다)
CMAKE_PREFIX_PATH="$SDK_PREFIX:${CMAKE_PREFIX_PATH:-}" \
  colcon build --packages-select rbpodo_msgs rbpodo_description rbpodo_hardware \
               --cmake-args -DCMAKE_BUILD_TYPE=Release
# 실습 패키지: --symlink-install (설정 파일·파이썬을 고치면 재빌드 없이 바로 반영)
set +u; source "$WS/install/setup.bash"; set -u
colcon build --symlink-install --packages-select \
  edu_pick_msgs edu_pick_description edu_pick_moveit edu_pick_perception edu_pick_task edu_pick_bringup

# colcon 은 없는 패키지를 --packages-select 에서 에러 없이 건너뛴다 → 직접 확인
for p in rbpodo_hardware edu_pick_msgs edu_pick_description edu_pick_moveit edu_pick_perception edu_pick_task edu_pick_bringup; do
  [ -d "$WS/install/$p" ] || die "패키지 '$p' 가 빌드되지 않았습니다 — 위 colcon 로그를 확인하세요."
done

# 새 터미널에서 자동으로 이 워크스페이스를 쓰게 한다 (NO_BASHRC=1 이면 건너뜀)
LINE="source $WS/install/setup.bash"
if [ "${NO_BASHRC:-0}" != "1" ] && ! grep -qF "$LINE" ~/.bashrc; then
  echo "$LINE" >> ~/.bashrc
  echo "  ~/.bashrc 에 추가: $LINE"
fi

echo
echo "✅ 설치 완료. 새 터미널을 열거나 'source ~/.bashrc' 후 점검하세요:"
echo "     bash $WS/check_setup.sh"
