import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ── Palette ────────────────────────────────────────────────────────────────────
BLUE_DARK  = colors.HexColor("#1A3A5C")
BLUE_MID   = colors.HexColor("#2E6DA4")
BLUE_LIGHT = colors.HexColor("#D6E8F7")
WHITE      = colors.white
GREY_TEXT  = colors.HexColor("#4A4A4A")

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm


# ── Styles ─────────────────────────────────────────────────────────────────────
def _build_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", fontSize=20, textColor=WHITE,
            fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", fontSize=10, textColor=BLUE_LIGHT,
            fontName="Helvetica", alignment=TA_CENTER,
        ),
        "section": ParagraphStyle(
            "section", fontSize=12, textColor=BLUE_DARK,
            fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body", fontSize=10, textColor=GREY_TEXT,
            fontName="Helvetica", leading=15, alignment=TA_JUSTIFY,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer", fontSize=8, textColor=colors.HexColor("#888888"),
            fontName="Helvetica-Oblique", alignment=TA_CENTER, leading=11,
        ),
        "pred_class": ParagraphStyle(
            "pred_class", fontSize=22, textColor=BLUE_DARK,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
        "pred_conf": ParagraphStyle(
            "pred_conf", fontSize=13, textColor=BLUE_MID,
            fontName="Helvetica", alignment=TA_CENTER,
        ),
    }


# ── Page template (header band + footer) ──────────────────────────────────────
def _make_page_template(doc):
    def on_page(canvas, doc):
        canvas.saveState()

        # Header band
        canvas.setFillColor(BLUE_DARK)
        canvas.rect(0, PAGE_H - 3.2 * cm, PAGE_W, 3.2 * cm, fill=1, stroke=0)

        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 16)
        canvas.drawCentredString(PAGE_W / 2, PAGE_H - 1.6 * cm,
                                 "Rapport d'Analyse Tumorale Cérébrale")
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(BLUE_LIGHT)
        canvas.drawCentredString(PAGE_W / 2, PAGE_H - 2.4 * cm,
                                 "Système de Détection Intelligente des Tumeurs")

        # Footer line
        canvas.setStrokeColor(BLUE_MID)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 1.4 * cm, PAGE_W - MARGIN, 1.4 * cm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GREY_TEXT)
        canvas.drawString(MARGIN, 0.9 * cm, f"Page {doc.page}")
        canvas.drawRightString(PAGE_W - MARGIN, 0.9 * cm,
                               f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}")

        canvas.restoreState()

    frame = Frame(MARGIN, 2 * cm, PAGE_W - 2 * MARGIN, PAGE_H - 5.5 * cm, id="main")
    return PageTemplate(id="medical", frames=[frame], onPage=on_page)


# ── Image helper ───────────────────────────────────────────────────────────────
def _safe_image(path: str, width: float, height: float):
    if path and os.path.exists(path):
        return Image(path, width=width, height=height)
    placeholder = Table(
        [["Image\nnon disponible"]],
        colWidths=[width], rowHeights=[height],
    )
    placeholder.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BLUE_LIGHT),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TEXTCOLOR",  (0, 0), (-1, -1), BLUE_MID),
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
    ]))
    return placeholder


# ── Main class ─────────────────────────────────────────────────────────────────
class PDFExporter:

    def generate_report(self, data: dict, output_path: str) -> str:
        """
        Génère un rapport PDF médical.

        Args:
            data: dict avec patient_name, date, image_path, prediction,
                  gradcam_path, llm_report.
            output_path: chemin du fichier PDF à créer.

        Returns:
            Chemin absolu du PDF généré.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        doc = BaseDocTemplate(
            output_path, pagesize=A4,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=3.5 * cm, bottomMargin=2.2 * cm,
        )
        doc.addPageTemplates([_make_page_template(doc)])

        styles = _build_styles()
        story  = []

        # ── Info patient ───────────────────────────────────────────────────────
        story.append(Spacer(1, 0.3 * cm))
        info_data = [
            ["Patient :", data.get("patient_name", "—"),
             "Date :",    data.get("date", "—")],
        ]
        info_table = Table(info_data, colWidths=[2.5 * cm, 7 * cm, 2 * cm, 5.5 * cm])
        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BLUE_LIGHT),
            ("FONTNAME",   (0, 0), (0, 0),   "Helvetica-Bold"),
            ("FONTNAME",   (2, 0), (2, 0),   "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("TEXTCOLOR",  (0, 0), (-1, -1), BLUE_DARK),
            ("ROWPADDING", (0, 0), (-1, -1), 8),
            ("ROUNDEDCORNERS", [4]),
            ("BOX",        (0, 0), (-1, -1), 0.5, BLUE_MID),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.5 * cm))

        # ── Section 1 : Images ─────────────────────────────────────────────────
        story.append(Paragraph("1. Imagerie IRM", styles["section"]))
        story.append(HRFlowable(width="100%", thickness=1, color=BLUE_MID, spaceAfter=8))

        img_w, img_h = 7.5 * cm, 6 * cm
        orig  = _safe_image(data.get("image_path"),  img_w, img_h)
        gcam  = _safe_image(data.get("gradcam_path"), img_w, img_h)

        label_style = ParagraphStyle(
            "lbl", fontSize=9, fontName="Helvetica-Bold",
            textColor=BLUE_DARK, alignment=TA_CENTER,
        )
        images_table = Table(
            [[orig, Spacer(0.5 * cm, 1), gcam],
             [Paragraph("Image originale", label_style), "",
              Paragraph("Carte d'activation Grad-CAM", label_style)]],
            colWidths=[img_w, 0.5 * cm, img_w],
        )
        images_table.setStyle(TableStyle([
            ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOX",    (0, 0), (0, 0),   0.5, BLUE_MID),
            ("BOX",    (2, 0), (2, 0),   0.5, BLUE_MID),
        ]))
        story.append(images_table)
        story.append(Spacer(1, 0.6 * cm))

        # ── Section 2 : Résultat prédiction ───────────────────────────────────
        story.append(Paragraph("2. Résultat de la Prédiction", styles["section"]))
        story.append(HRFlowable(width="100%", thickness=1, color=BLUE_MID, spaceAfter=8))

        pred       = data.get("prediction", {})
        pred_class = pred.get("class", "—")
        confidence = pred.get("confidence", 0.0)

        conf_bar_w  = 10 * cm
        filled_w    = conf_bar_w * (confidence / 100)
        bar_color   = (colors.HexColor("#E74C3C") if confidence < 60
                       else colors.HexColor("#27AE60") if confidence >= 85
                       else colors.HexColor("#F39C12"))

        bar_table = Table(
            [[""]],
            colWidths=[filled_w if filled_w > 0 else 0.01],
            rowHeights=[0.5 * cm],
        )
        bar_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bar_color),
            ("ROUNDEDCORNERS", [3]),
        ]))

        pred_content = Table(
            [[Paragraph(pred_class, styles["pred_class"])],
             [Paragraph(f"Confiance : {confidence:.1f} %", styles["pred_conf"])],
             [bar_table]],
            colWidths=[PAGE_W - 2 * MARGIN],
        )
        pred_content.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, -1), BLUE_LIGHT),
            ("ALIGN",          (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ("ROWPADDING",     (0, 0), (-1, -1), 10),
            ("BOX",            (0, 0), (-1, -1), 1, BLUE_MID),
            ("ROUNDEDCORNERS", [6]),
        ]))
        story.append(pred_content)
        story.append(Spacer(1, 0.6 * cm))

        # ── Section 3 : Rapport médical LLM ───────────────────────────────────
        story.append(Paragraph("3. Rapport Médical Détaillé", styles["section"]))
        story.append(HRFlowable(width="100%", thickness=1, color=BLUE_MID, spaceAfter=8))

        llm_text = data.get("llm_report", "Aucun rapport disponible.")
        for paragraph in llm_text.split("\n"):
            stripped = paragraph.strip()
            if stripped:
                story.append(Paragraph(stripped, styles["body"]))
                story.append(Spacer(1, 0.2 * cm))

        story.append(Spacer(1, 0.8 * cm))

        # ── Disclaimer ─────────────────────────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(
            "⚠️  AVERTISSEMENT MÉDICAL — Ce rapport est généré automatiquement par un système d'intelligence "
            "artificielle à des fins d'aide au diagnostic uniquement. Il ne constitue en aucun cas un "
            "diagnostic médical définitif et ne remplace pas l'avis d'un médecin qualifié. "
            "Toute décision thérapeutique doit être prise par un professionnel de santé habilité.",
            styles["disclaimer"],
        ))

        doc.build(story)
        return os.path.abspath(output_path)


# ── Fonction utilitaire (rétrocompatibilité) ───────────────────────────────────
def export_to_pdf(report_text: str, output_path: str) -> str:
    data = {"llm_report": report_text, "prediction": {}, "patient_name": "", "date": ""}
    return PDFExporter().generate_report(data, output_path)
