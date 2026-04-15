# /// script
# requires-python = ">=3.12"
# dependencies = ["reportlab>=4.0"]
# ///
"""Generate sample.pdf — a fictional tech brief used for the RAG demo.

Run with:
    uv run --script make_sample.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors

OUT = Path(__file__).parent / "sample.pdf"

styles = getSampleStyleSheet()
h1 = ParagraphStyle("h1", parent=styles["Heading1"], spaceAfter=10)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6)
body = ParagraphStyle("body", parent=styles["BodyText"], spaceAfter=6, leading=14)


def p(text: str):
    return Paragraph(text, body)


def h(text: str, level: int = 2):
    return Paragraph(text, h1 if level == 1 else h2)


def fact_table(rows: list[tuple[str, str]]):
    t = Table(rows, colWidths=[5 * cm, 10 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def build() -> None:
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Project Aurora — Technical Brief",
    )

    story = [
        h("Project Aurora — Technical Brief", level=1),
        p(
            "<i>Revision 3.2 · prepared by the Aurora core team · fictional document "
            "used for the Maestro RAG example.</i>"
        ),
        Spacer(1, 0.4 * cm),
        #
        h("1. Overview"),
        p(
            "Project Aurora is a distributed weather sensor network designed for "
            "alpine monitoring at elevations above 2,000 metres. The goal is to "
            "provide minute-resolution readings of temperature, humidity, wind "
            "speed, barometric pressure, and snow depth to climate researchers "
            "and regional avalanche services. The network was first deployed in "
            "March 2023 and entered full operation in November 2024."
        ),
        #
        h("2. Hardware"),
        p(
            "Each Aurora node is a ruggedised, solar-powered unit built around an "
            "STM32L4 microcontroller with a LoRaWAN radio. Nodes are rated for "
            "continuous operation between &minus;40&nbsp;°C and +60&nbsp;°C."
        ),
        fact_table(
            [
                ("Sensors", "BME280 (T/H/P), SEN0170 anemometer, MaxBotix MB7389 snow depth, SI1145 UV"),
                ("Battery", "18 Ah LiFePO\u2084, 30-day autonomy without sun"),
                ("Solar panel", "15 W monocrystalline, MPPT charge controller"),
                ("Radio", "LoRaWAN 868 MHz EU, spreading factor SF9"),
                ("Enclosure", "IP67, powder-coated aluminium, UV-stable gasket"),
                ("Firmware", "Zephyr RTOS 3.6, OTA updates via LoRa FUOTA"),
            ]
        ),
        #
        h("3. Deployment Sites"),
        p(
            "As of April 2026, 27 nodes are live across four mountain ranges. "
            "Each site has between 4 and 9 nodes forming a local mesh that "
            "relays to a gateway via the strongest link."
        ),
        fact_table(
            [
                ("Dolomites (IT)", "9 nodes, gateway at Rifugio Lagazuoi (2,752 m)"),
                ("Hohe Tauern (AT)", "7 nodes, gateway at Sonnblick Observatory"),
                ("Bernese Alps (CH)", "6 nodes, gateway at Jungfraujoch research station"),
                ("Julian Alps (SI)", "5 nodes, gateway at Kredarica weather station"),
            ]
        ),
        #
        h("4. Data Pipeline"),
        p(
            "Readings are transmitted every 60 seconds. Gateways forward packets "
            "over LTE to a Kafka cluster in Frankfurt, where a stream processor "
            "validates, deduplicates, and writes into TimescaleDB. Public data is "
            "exposed through a GraphQL API at api.aurora-network.eu with a "
            "fair-use limit of 1,000 requests per hour per key. Historical "
            "exports are available as Parquet on S3."
        ),
        #
        h("5. Known Limitations"),
        p(
            "Snow depth readings from the MB7389 degrade above 3.5 m of "
            "accumulation; the sensor saturates and reports 3.5 m as a ceiling. "
            "Wind speed above 45 m/s is clipped due to anemometer tolerance. "
            "During extended overcast periods of more than 21 consecutive days, "
            "nodes enter a reduced-cadence mode of one reading every 10 minutes."
        ),
        #
        h("6. Team and Timeline"),
        p(
            "The core team is six people split between ETH Zurich and the "
            "University of Innsbruck. Lead engineer is Dr. Elena Brunner; "
            "firmware is maintained by Marco Toth. Funding runs until Q4 2027 "
            "under EU Horizon grant 101094312. The next hardware revision "
            "(Aurora v4) is scheduled for field trials in September 2026 and "
            "adds a radar-based snow sensor to address the saturation issue "
            "noted in section 5."
        ),
        #
        h("7. Contact"),
        p(
            "General enquiries: hello@aurora-network.eu · "
            "Data access: data@aurora-network.eu · "
            "Source code: github.com/aurora-network (fictional)."
        ),
    ]

    doc.build(story)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
