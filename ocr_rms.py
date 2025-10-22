try:
    import os
    import glob
    import cv2
    import inspect
    import pytesseract as pt
    import numpy as np
    import shutil as sh
    import tempfile as tf
    from pylovepdf.ilovepdf import ILovePdf as ilp
    from pathlib import Path
    from io import BytesIO
    from PIL import Image
    from pdf2image import convert_from_path
    from PyPDF2 import PdfMerger
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.colors import Color
    import time
except ImportError as e:
    print(f"Required module(s) not found. Please run 'pip install PIL pytesseract pdf2image PyPDF2 reportlab pylovepdf opencv-python numpy' in the command line to install dependencies.")
    exit()

years = ['1954','1955','1956','1957','1958','1959','1960','1961','1962','1963','1964','1965','1966','1967','1968']

def compress_pdf(pdf_bytes: bytes) -> bytes:
    api_public_key = os.getenv("COMPRESS_PDF_API_USER")
    if not api_public_key:
        print("API keys for PDF compression (iLovePDF) are not set in environment variables. (COMPRESS_PDF_API_USER)")
        proceed : str = input("Would you like to continue without compression? [WARNING: FILE SIZES MAY BE LARGE] (y/n): ")
        if proceed.lower() != 'y':
            print("Your hard drive will be spared this time. Exiting...")
            exit()
        else:
            return pdf_bytes
        
    ilovepdf = ilp(api_public_key, verify_ssl=True)
    task = ilovepdf.new_task("compress")

    out_dir = Path(tf.mkdtemp())
    in_fd, in_path = tf.mkstemp(suffix=".pdf", dir=out_dir)
    os.close(in_fd)
    with open(in_path, "wb") as f:
        f.write(pdf_bytes)

    start_time = time.time()
    try:
        task.add_file(in_path)
        task.execute()

        #call download according to SDK signature (some versions accept a path, some don't)
        try:
            sig = inspect.signature(task.download)
            params = [p for p in sig.parameters.values() if p.name != 'self']
            if len(params) >= 1:
                task.download(out_dir)
            else:
                task.download()
        except (ValueError, TypeError):
            # fallback: try both forms
            try:
                task.download(out_dir)
            except TypeError:
                task.download()

        #look for output file in out_dir first
        out_files = glob.glob(os.path.join(out_dir, "*.pdf"))
        #if none found, search likely locations for recently created/modified PDFs
        if not out_files:
            candidates = []
            search_dirs = [os.getcwd(), tf.gettempdir(), out_dir]
            for d in search_dirs:
                try:
                    for p in glob.glob(os.path.join(d, "*.pdf")):
                        try:
                            mtime = os.path.getmtime(p)
                        except Exception:
                            mtime = 0
                        candidates.append((mtime, p))
                except Exception:
                    continue
            #prefer files modified after start_time - 5s, else pick newest candidate
            recent = [p for m, p in candidates if m >= (start_time - 5)]
            if recent:
                out_files = recent
            elif candidates:
                candidates.sort(reverse=True)
                out_files = [candidates[0][1]]
            else:
                out_files = []

        if not out_files:
            raise RuntimeError("No output PDF file found after compression.")

        #pick the first matching output file
        with open(out_files[0], "rb") as f:
            compressed_pdf_bytes = f.read()
        return compressed_pdf_bytes
    finally:
        try:
            os.remove(in_path)
        except Exception:
            pass
        sh.rmtree(out_dir, ignore_errors=True)
    
def enhance_image(img) -> Image:
    img = img.convert('RGB') #ensure image is in RGB mode
    img = np.array(img) #convert PIL image to numpy array
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) #resize image to improve ocr accuracy
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #convert to grayscale
    img = cv2.GaussianBlur(img, (5, 5), 0) #apply gaussian blur to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)) #kernel it up yo
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) #apply morphological operations to enhance text
    img = Image.fromarray(img.astype('uint8')) #convert back to PIL image
    return img

def image_to_searchable_pdf_bytes(img, dpi = 600, psm = 6, oem = 1) -> bytes:
    #build a PDF with searchable text layer including all OCR tokens (even low-confidence)
    img = img.convert("RGB")
    width_px, height_px = img.size
    #points = pixels * 72 / dpi
    page_w = width_px * 72.0 / dpi
    page_h = height_px * 72.0 / dpi

    buf = BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=(page_w, page_h))

    #draw the raster image to fill the page
    img_buf = BytesIO()
    img.save(img_buf, format="PNG")
    img_buf.seek(0)
    ir = ImageReader(img_buf)
    c.drawImage(ir, 0, 0, width=page_w, height=page_h, preserveAspectRatio=False, mask='auto')

    #get word boxes/confidences (we include all words regardless of confidence)
    config = f"--oem {oem} --psm {psm}"
    data = pt.image_to_data(img, output_type=pt.Output.DICT, config=config)

    #set an invisible fill color so text is searchable but not visible
    invisible = Color(0, 0, 0, alpha=0)
    c.setFillColor(invisible)

    n = len(data['text'])
    for i in range(n):
        raw = (data['text'][i] or "").strip()
        if not raw:
            #still might want to include very-low-confidence blanks, but skip empty strings
            continue
        #bounding box in pixels
        left = data['left'][i]
        top = data['top'][i]
        width = data['width'][i]
        height = data['height'][i]

        #convert to PDF points and flip Y axis
        x_pt = left * 72.0 / dpi
        y_pt = page_h - ((top + height) * 72.0 / dpi)

        #choose a font size roughly matching box height
        font_size = max(4, height * 72.0 / dpi)
        try:
            c.setFont("Helvetica", font_size)
            #drawString places text at baseline; using y_pt computed above is OK
            c.drawString(x_pt, y_pt, raw)
        except Exception:
            #skip any problematic token
            continue

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()
    img = enhance_image(img) #improve image quality for ocr
    return image_to_searchable_pdf_bytes(img, dpi=450, psm=6, oem=1) #searchable pdf after ocr

def read_merge(mayor_doc, tax_doc, counter) -> None:
    merger = PdfMerger()
    for img in mayor_doc:
        img = img.rotate(270, expand=True)
        img = enhance_image(img) #improve image quality for ocr
        page = pt.image_to_pdf_or_hocr(img) #searchable pdf after ocr
        merger.append(BytesIO(page))
    for img in tax_doc:
        if counter < 14: #for years before 1967, rotate tax images to be upright
            img = img.rotate(270, expand=True)
        img = enhance_image(img) #improve image quality for ocr
        page = image_to_searchable_pdf_bytes(img, dpi=600, psm=6, oem=1) #searchable pdf after ocr
        merger.append(BytesIO(page))

    merger.write(f'ocr_rms_{years[counter]}.pdf')
    merger.close()

    out_name = f'ocr_rms_{years[counter]}.pdf' if 0 <= counter < len(years) else f'ocr_rms_{counter}.pdf'
    buf = BytesIO()
    merger.write(buf)
    merger.close()
    buf.seek(0)
    merged_bytes = buf.read()
    try:
        compressed_bytes = compress_pdf(merged_bytes)
    except Exception as e:
        print(f"PDF compression skipped/failed: {e}")
        compressed_bytes = merged_bytes
    with open(out_name, 'wb') as f:
        f.write(compressed_bytes)


current_dir = os.path.dirname(os.path.abspath(__file__))
mayors = os.path.join(current_dir, 'mayors')
taxes = os.path.join(current_dir, 'taxes')
pp = r"C:\Poppler\poppler-25.07.0\Library\bin" #path to poppler bin directory

for i in range(0, min(len(os.listdir(mayors)), len(os.listdir(taxes)))):
    mayor_pdfs = sorted(glob.glob(os.path.join(mayors, '*.pdf')))
    tax_pdfs = sorted(glob.glob(os.path.join(taxes, '*.pdf')))
    read_merge(convert_from_path(mayor_pdfs[i], dpi=600, poppler_path=pp), convert_from_path(tax_pdfs[i], dpi=600, poppler_path=pp), i)