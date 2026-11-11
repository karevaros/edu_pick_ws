#!/usr/bin/env python3
"""OMX 손목 카메라 보정용 ChArUco 판 (A4 가로 · 실제 크기 인쇄용 PDF).

왜 ChArUco 인가: OMX 손목 카메라는 화면 아래쪽 절반을 그리퍼 턱이 가린다.
  일반 체커판은 판 전체가 보여야 인식되지만, ChArUco 는 칸마다 ArUco 번호가 있어 '일부만 보여도' 모서리를 찾는다.

판: 9(가로) × 6(세로) 칸 · 칸 25 mm · 마커 18 mm · DICT_4X4_50  → 225 × 150 mm
놓는 법(종이 안내문에도 인쇄): 판의 '로봇 쪽' 긴 변을 로봇(1번 축)에서 정면 10 cm 에 두고 좌우 가운데를 맞춘다
  → 판이 x 0.10~0.25 m · y −0.1125~+0.1125 m 를 덮는다(토마토 놓는 자리 전체).

사용:  python3 tools/make_charuco_board.py [출력.pdf]
⚠ 인쇄할 때 '실제 크기(100 %)' — '페이지에 맞춤' 금지. 인쇄 후 아래 100 mm 눈금을 자로 재서 확인.
"""
import sys
import tempfile

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

COLS, ROWS = 9, 6            # 칸 수 (가로 = 로봇 좌우 방향 · 세로 = 로봇 앞뒤 방향)
SQUARE, MARKER = 25.0, 18.0  # mm
DICT = cv2.aruco.DICT_4X4_50
DPI_PX_PER_MM = 20           # 그림 해상도 (508 dpi)


def board():
    d = cv2.aruco.getPredefinedDictionary(DICT)
    return cv2.aruco.CharucoBoard((COLS, ROWS), SQUARE / 1000.0, MARKER / 1000.0, d)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "omx_charuco_A4.pdf"
    w_mm, h_mm = COLS * SQUARE, ROWS * SQUARE
    img = board().generateImage((int(w_mm * DPI_PX_PER_MM), int(h_mm * DPI_PX_PER_MM)), marginSize=0, borderBits=1)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    cv2.imwrite(tmp, img)

    pw, ph = landscape(A4)
    c = canvas.Canvas(out, pagesize=(pw, ph))
    x0 = (pw - w_mm * mm) / 2
    y0 = 30 * mm                                  # 아래(로봇 쪽) 여백 — 안내문
    c.drawImage(tmp, x0, y0, w_mm * mm, h_mm * mm)
    c.setLineWidth(0.3)
    c.rect(x0, y0, w_mm * mm, h_mm * mm)

    # 로봇 쪽 변 가운데 표시 + 화살표(로봇 방향)
    cx = x0 + w_mm * mm / 2
    c.setLineWidth(1.0)
    c.line(cx, y0, cx, y0 - 8 * mm)
    c.line(cx, y0 - 8 * mm, cx - 3 * mm, y0 - 4 * mm)
    c.line(cx, y0 - 8 * mm, cx + 3 * mm, y0 - 4 * mm)
    c.setFont("Helvetica", 8)
    c.drawCentredString(cx, y0 - 12 * mm, "ROBOT SIDE (link1 axis) - this edge center at x = 100 mm in front of the robot")

    # 100 mm 눈금 (인쇄 배율 확인)
    sx, sy = x0, 12 * mm
    c.setLineWidth(0.8)
    c.line(sx, sy, sx + 100 * mm, sy)
    for k in range(11):
        t = 3 * mm if k % 5 == 0 else 1.5 * mm
        c.line(sx + k * 10 * mm, sy, sx + k * 10 * mm, sy + t)
    c.drawString(sx + 102 * mm, sy, "100 mm  (print at 100 % / actual size - check with a ruler)")
    c.drawRightString(x0 + w_mm * mm, ph - 12 * mm,
                      f"ChArUco {COLS}x{ROWS} squares {SQUARE:.0f} mm, markers {MARKER:.0f} mm, DICT_4X4_50 - edu_pick OMX camera calibration")
    c.showPage()
    c.save()
    print(f"저장: {out}  (판 {w_mm:.0f} × {h_mm:.0f} mm)")


if __name__ == "__main__":
    main()
