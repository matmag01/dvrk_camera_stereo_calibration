import cv2
from cv2 import aruco
from reportlab.platypus import SimpleDocTemplate, Image
from reportlab.lib.units import mm
from reportlab.lib.pagesizes import A4
from io import BytesIO
from PIL import Image as PILImage

# =========================
# BOARD
# =========================
squaresX, squaresY = 5, 7
squareLength_mm = 15
markerLength_mm = 9
margin_mm = 10
dpi = 300

output_pdf = "charuco_board_A4_1to1_5x7_15mm_9mm.pdf"

# =========================
# CREATION OF CHARUCO BOARD
# =========================
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

squareLength = squareLength_mm / 1000.0
markerLength = markerLength_mm / 1000.0

try:
    board = aruco.CharucoBoard((squaresX, squaresY),
                               squareLength,
                               markerLength,
                               dictionary)
except AttributeError:
    board = aruco.CharucoBoard_create(squaresX,
                                      squaresY,
                                      squareLength,
                                      markerLength,
                                      dictionary)

# Dimension board
board_w_mm = squaresX * squareLength_mm + 2 * margin_mm
board_h_mm = squaresY * squareLength_mm + 2 * margin_mm

# Conversion mm → pixel
px_w = int(board_w_mm / 25.4 * dpi)
px_h = int(board_h_mm / 25.4 * dpi)
margin_px = int(margin_mm / 25.4 * dpi)

# Render image
if hasattr(board, "generateImage"):
    img = board.generateImage((px_w, px_h),
                              marginSize=margin_px,
                              borderBits=1)
else:
    img = board.draw((px_w, px_h),
                     marginSize=margin_px,
                     borderBits=1)


pil_img = PILImage.fromarray(img)
buffer = BytesIO()
pil_img.save(buffer, format="PNG")
buffer.seek(0)

# =========================
#  PDF A4 1:1
# =========================
doc = SimpleDocTemplate(output_pdf, pagesize=A4)

pdf_img = Image(buffer)
pdf_img.drawWidth = board_w_mm * mm
pdf_img.drawHeight = board_h_mm * mm

doc.build([pdf_img])

print("PDF generato:", output_pdf)
