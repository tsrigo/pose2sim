#!/usr/bin/env python3
"""Generate the HZVision Pose2Sim 阶段性优化总结 deck (.pptx).

Reader-first, novice-facing narrative: problem → root cause → method → result →
meaning. One message per slide, minimal text, no decorative symbols. The four
technique slides share one 现象 / 做法 / 效果 structure so the talk reads
coherently. Source: docs/hzvision-integration.md + project memory notes.

Usage:
    python scripts/make_optimization_summary_ppt.py [output.pptx]
"""
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# --- palette (clean, professional) -----------------------------------------
NAVY = RGBColor(0x14, 0x36, 0x5C)   # titles
BLUE = RGBColor(0x25, 0x63, 0xEB)   # method / accent
INK = RGBColor(0x22, 0x2A, 0x35)    # body text
GREY = RGBColor(0x70, 0x7A, 0x86)   # secondary
RED = RGBColor(0xC8, 0x3B, 0x36)    # problem
GREEN = RGBColor(0x1E, 0x8E, 0x3E)  # result
LIGHT = RGBColor(0xEE, 0xF3, 0xFA)  # panel bg
RULE = RGBColor(0xD3, 0xDE, 0xEC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Microsoft YaHei"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]
_idx = [0]


def _set(run, size, color=INK, bold=False):
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.name = FONT


def _rich(p, text, size, color, bold=False):
    """Render text with **...** -> bold emphasis (same color)."""
    for i, part in enumerate(text.split("**")):
        if not part:
            continue
        r = p.add_run(); r.text = part
        _set(r, size, color, bold=(bold or i % 2 == 1))


def box(slide, l, t, w, h, anchor=None):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Pt(0)
    tf.margin_top = tf.margin_bottom = Pt(0)
    if anchor:
        tf.vertical_anchor = anchor
    return tf


def rect(slide, shp, l, t, w, h, fill=None, line=None):
    sp = slide.shapes.add_shape(shp, l, t, w, h)
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(1)
    sp.shadow.inherit = False
    return sp


def new_slide(kicker=None, headline=None, full=None):
    s = prs.slides.add_slide(BLANK)
    _idx[0] += 1
    if full is not None:
        rect(s, MSO_SHAPE.RECTANGLE, 0, 0, SW, SH, fill=full)
        return s
    if kicker:
        tf = box(s, Inches(0.75), Inches(0.46), Inches(11.8), Inches(0.32))
        r = tf.paragraphs[0].add_run(); r.text = kicker; _set(r, 15, BLUE, bold=True)
    if headline:
        tf = box(s, Inches(0.72), Inches(0.80), Inches(12.0), Inches(0.85))
        _rich(tf.paragraphs[0], headline, 27, NAVY, bold=True)
    rect(s, MSO_SHAPE.RECTANGLE, Inches(0.75), Inches(1.64), Inches(11.83), Pt(2), fill=RULE)
    # page number
    tf = box(s, Inches(12.4), Inches(7.05), Inches(0.8), Inches(0.3))
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    r = p.add_run(); r.text = f"{_idx[0]:02d}"; _set(r, 11, GREY)
    return s


def pill(slide, l, t, text, color, w=1.18, h=0.44):
    sp = rect(slide, MSO_SHAPE.ROUNDED_RECTANGLE, l, t, Inches(w), Inches(h), fill=color)
    tf = sp.text_frame; tf.word_wrap = False
    tf.margin_left = tf.margin_right = Pt(2); tf.margin_top = tf.margin_bottom = Pt(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text; _set(r, 14, WHITE, bold=True)


def card(slide, top, label, label_color, text, text_color):
    """One 现象/做法/效果 row: colored pill + supporting line(s)."""
    pill(slide, Inches(0.78), Inches(top + 0.06), label, label_color)
    tf = box(slide, Inches(2.18), Inches(top), Inches(10.45), Inches(1.3),
             anchor=MSO_ANCHOR.MIDDLE)
    _rich(tf.paragraphs[0], text, 18, text_color)


# ===========================================================================
# 1. Title
# ===========================================================================
s = new_slide(full=NAVY)
rect(s, MSO_SHAPE.RECTANGLE, 0, Inches(4.9), SW, Inches(0.06), fill=BLUE)
tf = box(s, Inches(0.95), Inches(1.75), Inches(11.4), Inches(2.6))
r = tf.paragraphs[0].add_run(); r.text = "Pose2Sim · HZVision 分支"; _set(r, 22, RGBColor(0xBF, 0xD6, 0xF2))
p = tf.add_paragraph(); p.space_before = Pt(10)
r = p.add_run(); r.text = "给无标记动捕补上「人体常识」"; _set(r, 40, WHITE, bold=True)
p = tf.add_paragraph(); p.space_before = Pt(8)
r = p.add_run(); r.text = "让 4 相机还原的 3D 骨架不再乱跳、不再左右颠倒"; _set(r, 22, RGBColor(0xE2, 0xEC, 0xF8))
tf = box(s, Inches(0.95), Inches(5.25), Inches(11.4), Inches(1.0))
r = tf.paragraphs[0].add_run()
r.text = "阶段性优化总结　·　相比原始 Pose2Sim 的改进"
_set(r, 17, RGBColor(0xCF, 0xE2, 0xF5))

# ===========================================================================
# 2. 一页读懂
# ===========================================================================
s = new_slide("一页读懂", "我们做了什么，为什么重要")
rows = [
    ("问题", RED, "原始方法在真实试验上会出现：腰部瞬移、左右腿互换、转身抖动、手臂甩飞", INK),
    ("做法", BLUE, "把「人体是刚体 / 有确定左右 / 肢体定长 / 朝向连续」四条常识，注入三角化", INK),
    ("结果", GREEN, "这些伪影在真实数据上**清零**，且不过度平滑真实运动；默认关闭，不配置即与原版一致", INK),
]
for i, (lab, c, txt, tc) in enumerate(rows):
    card(s, 2.05 + i * 1.55, lab, c, txt, tc)

# ===========================================================================
# 3. 背景
# ===========================================================================
s = new_slide("背景", "无标记动捕：几个普通相机 → 自动还原 3D 骨架")
flow = ["多机位视频", "2D 关键点", "多视角三角化", "3D 骨架 (TRC)"]
x = Inches(0.85); top = Inches(2.25); bw = Inches(2.55); bh = Inches(1.05); gap = Inches(0.42)
for i, name in enumerate(flow):
    fill = BLUE if i == 2 else LIGHT
    fg = WHITE if i == 2 else NAVY
    sp = rect(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, top, bw, bh, fill=fill)
    tf = sp.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = name; _set(r, 17, fg, bold=True)
    if i < 3:
        atf = box(s, x + bw, top, gap, bh, anchor=MSO_ANCHOR.MIDDLE)
        ap = atf.paragraphs[0]; ap.alignment = PP_ALIGN.CENTER
        ar = ap.add_run(); ar.text = "→"; _set(ar, 22, GREY, bold=True)
    x = x + bw + gap
tf = box(s, Inches(0.85), Inches(3.95), Inches(11.7), Inches(1.6))
_rich(tf.paragraphs[0], "**三角化**：用多个相机里同一个点的 2D 位置，交会出它的 3D 坐标。", 18, INK)
p = tf.add_paragraph(); p.space_before = Pt(10)
_rich(p, "不穿戴传感器、不贴反光球，适合运动、医疗、户外等场景——也因此对算法鲁棒性要求更高。", 18, GREY)

# ===========================================================================
# 4. 问题
# ===========================================================================
s = new_slide("问题", "在真实生产数据上，原始三角化会「崩」")
probs = [
    ("腰部整体突然蹦一下", "单帧瞬移 ~20 cm"),
    ("走路时左右髋 / 腿互换", "一段几十帧持续反向"),
    ("转身、侧站时髋部疯狂抖动", "甚至整盆翻面"),
    ("手臂、肩膀突然甩飞", "长度炸到 1–2 米"),
]
top = 2.0
for sym, mag in probs:
    rect(s, MSO_SHAPE.OVAL, Inches(0.85), Inches(top + 0.12), Inches(0.18), Inches(0.18), fill=RED)
    tf = box(s, Inches(1.25), Inches(top), Inches(7.6), Inches(0.6), anchor=MSO_ANCHOR.MIDDLE)
    _rich(tf.paragraphs[0], sym, 19, INK, bold=True)
    tf = box(s, Inches(8.9), Inches(top), Inches(3.7), Inches(0.6), anchor=MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = mag; _set(r, 18, RED, bold=True)
    top += 0.95
tf = box(s, Inches(0.85), Inches(6.2), Inches(11.7), Inches(0.7))
_rich(tf.paragraphs[0], "这些都不是相机或 2D 检测坏了——而是**三角化这一步**的系统性缺陷。", 17, GREY)

# ===========================================================================
# 5. 根因（中间结论）
# ===========================================================================
s = new_slide("根本原因", "根因只有一个：三角化「不懂人体」")
tf = box(s, Inches(0.78), Inches(1.95), Inches(11.8), Inches(1.0))
_rich(tf.paragraphs[0],
      "原始做法：每个关键点**各自独立**三角化，间隙用上一帧的值简单填补。它没有用上任何人体先验——",
      18, INK)
consts = [
    "人体局部是刚体（骨盆、颅骨形状固定）",
    "人有确定的左右（左右髋不可能互换）",
    "肢体长度固定（前臂不会突然变 2 米）",
    "朝向随时间连续（30fps 间不会瞬间翻面）",
]
top = 3.05
for c in consts:
    rect(s, MSO_SHAPE.OVAL, Inches(1.05), Inches(top + 0.08), Inches(0.16), Inches(0.16), fill=BLUE)
    tf = box(s, Inches(1.42), Inches(top), Inches(10.8), Inches(0.5), anchor=MSO_ANCHOR.MIDDLE)
    _rich(tf.paragraphs[0], c, 18, INK)
    top += 0.62
sp = rect(s, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.78), Inches(top + 0.15), Inches(11.8), Inches(0.7),
          fill=LIGHT)
tf = sp.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
tf.margin_left = Pt(14)
_rich(tf.paragraphs[0], "把这四条常识补回去，四类伪影就会**同时消失**。", 19, NAVY, bold=True)

# ===========================================================================
# 6. 方法地图
# ===========================================================================
s = new_slide("方法", "解法：把四条人体常识，逐层注入三角化")
chips = ["① 刚体约束", "② 左右一致", "③ 朝向稳健", "④ 肢体定长"]
x = Inches(0.85); top = Inches(2.0); cw = Inches(2.78); ch = Inches(1.0); gap = Inches(0.18)
for name in chips:
    sp = rect(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, top, cw, ch, fill=BLUE)
    tf = sp.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = name; _set(r, 17, WHITE, bold=True)
    x = x + cw + gap
# layer below: 2D
sp = rect(s, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.85), Inches(3.35), Inches(11.68), Inches(0.85), fill=LIGHT)
tf = sp.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.margin_left = Pt(16)
_rich(tf.paragraphs[0], "前置 ·  先稳 2D 输入：时序去抖；弯腰、遮挡时不把整帧关键点清空", 17, NAVY, bold=True)
# foundation
sp = rect(s, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.85), Inches(4.4), Inches(11.68), Inches(0.85), fill=RGBColor(0xE7, 0xEB, 0xF0))
tf = sp.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.margin_left = Pt(16)
_rich(tf.paragraphs[0], "工程基座 ·  avi2trc 一键 AVI→TRC　·　批量推理　·　并行三角化　·　mosaic 可视化质检", 17, GREY)
tf = box(s, Inches(0.85), Inches(5.6), Inches(11.7), Inches(0.7))
_rich(tf.paragraphs[0], "全部**默认关闭、互不影响**；下面逐条展开——每条都讲清现象、做法、效果。", 17, GREY)

# ===========================================================================
# 7-10. 四条常识（统一三段式）
# ===========================================================================
def technique(kicker, headline, phenom, method, effect):
    s = new_slide(kicker, headline)
    card(s, 2.05, "现象", RED, phenom, INK)
    card(s, 3.65, "做法", BLUE, method, INK)
    card(s, 5.25, "效果", GREEN, effect, GREEN)
    return s


technique(
    "方法 · 常识 ①", "人体局部是刚体 —— 钉死骨盆 / 颅骨的形状",
    "腰被拉伸 / 塌缩，弯腰时整盆瞬移 ~20 cm；低头时脸部关键点乱跳",
    "把骨盆、颅骨各设为「刚体组」，用固定模板拟合轨迹；Procrustes 对齐避免转身时模板被压扁",
    "弯腰瞬移 **21.9 → 0.4 cm**；静态骨盆宽度恢复 **14.5 → 22.4 cm**（找回真实尺寸）")

technique(
    "方法 · 常识 ②", "人有确定的左右 —— 让独立解当「裁判」",
    "走路侧对相机时，骨盆几乎零代价就能 180° 翻转，左右髋持续互换",
    "逐点独立解永远不会换边——用它当左右基准，翻转的帧一律拒绝；基准缺失时靠时间连续性兜底",
    "行走互换帧 **50 → 0、20 → 0**……多个试验全部归零，非互换帧与改前逐字节一致")

technique(
    "方法 · 常识 ③", "朝向应平滑且符合解剖 —— 抑制病态旋转",
    "侧站 / 转身时骨盆朝向「病态」（轻微转动就能乱解），逐帧晃动甚至瞬时翻面",
    "旋转在 SO(3) 上取稳健中值，剔除尖峰、保留真实转动；再把骨盆相对躯干的扭转钳在解剖范围内",
    "转身试验「抖动 + 翻面」**→ 0**；行走真实转动范围不变（不过度平滑）")

technique(
    "方法 · 常识 ④", "肢体长度固定 —— 拉回炸飞的手臂",
    "两相机分歧时，手臂 / 肩膀的独立解会炸到 1–2 米",
    "沿骨架自根到叶检查每根骨段，超长的把远端拉回常态长度；设绝对阈值，正常脚部抖动不误伤",
    "前臂 **116 → 60 cm**；干净 / 快速 / 行走试验逐点 **0.000 mm**（完全不动）")

# ===========================================================================
# 11. 结果汇总
# ===========================================================================
s = new_slide("结果", "真实 PnVision 试验：四类伪影全部清零")
data = [
    ("场景", "改前", "改后"),
    ("弯腰 · 腰部瞬移", "21.9 cm", "0.4 cm"),
    ("行走 · 左右互换", "最多 50 帧", "0 帧"),
    ("转身 · 抖动 / 翻面", "多处", "0"),
    ("静态 · 骨盆宽度", "14.5 cm（错）", "22.4 cm（真实）"),
    ("肢体 · 前臂爆炸", "116 cm", "60 cm"),
]
left, top = Inches(1.4), Inches(2.0)
table = s.shapes.add_table(len(data), 3, left, top, Inches(10.5), Inches(3.7)).table
table.columns[0].width = Inches(4.5)
table.columns[1].width = Inches(3.0)
table.columns[2].width = Inches(3.0)
for ci, name in enumerate(data[0]):
    cell = table.cell(0, ci); cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = cell.text_frame.paragraphs[0].add_run(); r.text = name; _set(r, 16, WHITE, bold=True)
for ri in range(1, len(data)):
    for ci in range(3):
        cell = table.cell(ri, ci)
        cell.fill.solid(); cell.fill.fore_color.rgb = LIGHT if ri % 2 else WHITE
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_top = Pt(4); cell.margin_bottom = Pt(4); cell.margin_left = Pt(12)
        r = cell.text_frame.paragraphs[0].add_run(); r.text = data[ri][ci]
        col = GREEN if ci == 2 else (RED if ci == 1 else INK)
        _set(r, 15, col, bold=(ci != 0))
sp = rect(s, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.4), Inches(6.0), Inches(10.5), Inches(0.75), fill=LIGHT)
tf = sp.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.margin_left = Pt(14)
_rich(tf.paragraphs[0], "同时：干净 / 快速 / 行走试验逐点 **0.000 mm** —— 不引入新抖动、不过度平滑真实动作。", 17, NAVY)

# ===========================================================================
# 12. 为什么可信
# ===========================================================================
s = new_slide("为什么可信", "三条工程纪律，保证「修了不坏」")
disc = [
    ("向后兼容", "所有功能默认关闭，不配置即与原版 Pose2Sim 逐字节一致"),
    ("根因修复", "每个伪影先用真实代码复现、定位到具体阶段，再修——不在报告里粉饰"),
    ("可复现验证", "随附验证脚本，分钟级从缓存 2D 重跑，前后对比留下证据"),
]
top = 2.1
for lab, txt in disc:
    pill(s, Inches(0.85), Inches(top + 0.1), lab, BLUE, w=1.7)
    tf = box(s, Inches(2.75), Inches(top), Inches(9.8), Inches(1.1), anchor=MSO_ANCHOR.MIDDLE)
    _rich(tf.paragraphs[0], txt, 18, INK)
    top += 1.5

# ===========================================================================
# 13. 小结
# ===========================================================================
s = new_slide(full=NAVY)
tf = box(s, Inches(0.95), Inches(1.4), Inches(11.4), Inches(1.0))
r = tf.paragraphs[0].add_run(); r.text = "小结"; _set(r, 34, WHITE, bold=True)
tf = box(s, Inches(0.95), Inches(2.6), Inches(11.5), Inches(3.4))
lines = [
    "一句话：给无标记动捕补上人体常识，让 3D 骨架在弯腰、转身、行走、遮挡下都稳。",
    "顺带完成工程化：avi2trc 一键流程、批量推理、并行三角化、可视化质检。",
    "后续：≥3 背向相机时左右基准缺失的兜底；滤波层细微宽度凹陷；持续同步官方更新。",
]
first = True
for t in lines:
    p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
    p.space_after = Pt(16)
    r = p.add_run(); r.text = "›  "; _set(r, 19, RGBColor(0xBF, 0xD6, 0xF2), bold=True)
    r = p.add_run(); r.text = t; _set(r, 19, WHITE)
tf = box(s, Inches(0.95), Inches(6.5), Inches(11.4), Inches(0.6))
r = tf.paragraphs[0].add_run()
r.text = "细节见 docs/hzvision-integration.md　·　分支 pose2sim-hzvision"
_set(r, 14, RGBColor(0xCF, 0xE2, 0xF5))

out = sys.argv[1] if len(sys.argv) > 1 else "Pose2Sim_HZVision_优化总结.pptx"
prs.save(out)
print(f"saved: {out}  ({len(prs.slides._sldIdLst)} slides)")
