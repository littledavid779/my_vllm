#!/usr/bin/env python3
"""Build a standalone, editable slide for paired promotion events."""

from pathlib import Path

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

from build_proposal_ppt import (
    BLUE,
    GREEN,
    NAVY,
    ORANGE,
    PALE_BLUE,
    PRESENTATION_BACKGROUND,
    SLIDE_H,
    SLIDE_W,
    add_background,
    add_metric,
    add_statement,
    add_table,
    add_text,
    add_title,
)


def build(output: Path) -> None:
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    prs.core_properties.title = "Restore / Recompute Paired Events"
    prs.core_properties.subject = "Direct ShareGPT event-level promotion evidence"
    prs.core_properties.comments = "Editable PowerPoint table generated from local experiment summary."

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, PRESENTATION_BACKGROUND)
    add_title(
        slide,
        "三套独立 ShareGPT 中存在 Restore / Recompute Crossover",
        "事件级实验",
        1,
        "[R6, R7] · strict paired events",
    )
    add_text(
        slide,
        "Qwen3-0.6B · GPU KV 1000 blocks · CPU KV 4 GiB · active = 32 · 128 requests / seed",
        0.68,
        1.43,
        9.08,
        0.30,
        size=11.2,
        bold=True,
        color_hex="B9C8DA",
    )

    data = [
        [
            "Dataset\nseed",
            "Paired\nevents",
            "Restore\nwait",
            "Recompute\nwait",
            "Restore\nwins",
            "Recompute\nwins",
            "Oracle\ngain",
        ],
        ["0", "41", "10.308 ms", "15.672 ms", "32", "9", "5.53%"],
        ["1", "32", "9.479 ms", "13.975 ms", "27", "5", "6.19%"],
        ["2", "36", "9.283 ms", "12.318 ms", "26", "10", "9.11%"],
        ["Total / Macro", "109", "9.726 ms", "14.066 ms", "85", "24", "6.85%"],
    ]
    add_table(
        slide,
        data,
        0.68,
        1.86,
        8.95,
        3.98,
        widths=[1.08, 0.92, 1.28, 1.42, 0.94, 1.10, 0.94],
        font_size=11.2,
        highlight_rows={4: PALE_BLUE},
    )

    add_metric(slide, "109", "严格配对事件", 9.92, 1.86, 2.70, accent=BLUE)
    add_metric(slide, "85 / 24", "Restore / Recompute wins", 9.92, 3.08, 2.70, accent=GREEN)
    add_metric(slide, "22.0%", "事件中 recompute 更优", 9.92, 4.30, 2.70, accent=ORANGE)
    add_text(
        slide,
        "6.85%  Promotion Oracle gap",
        9.92,
        5.53,
        2.70,
        0.44,
        size=11.5,
        bold=True,
        fill=NAVY,
        line=NAVY,
        radius=True,
        align=PP_ALIGN.CENTER,
        color_hex="FFFFFF",
    )

    add_statement(
        slide,
        "结论：CPU hit 不等于必须 RESTORE；109 个严格配对事件中有 24 个更适合 RECOMPUTE。",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(output)


if __name__ == "__main__":
    target = Path("presentations/restore_recompute_paired_events_20260820.pptx")
    build(target)
    reopened = Presentation(target)
    print(f"created={target}")
    print(f"slides={len(reopened.slides)}")
    print(f"size_bytes={target.stat().st_size}")
