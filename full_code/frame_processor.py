from collections import Counter
from image_processor import ImageProcessor

class FrameProcessor:
    def __init__(self, model, plate_class_indices, reader, width, height, padding, cur, conn, min_confidence):
        self.model = model
        self.plate_class_indices = plate_class_indices
        self.reader = reader
        self.width = width
        self.height = height
        self.padding = padding
        self.cur = cur
        self.conn = conn
        self.min_confidence = min_confidence
        self.processed_numbers = set()
    
    def process_frame(self, frame_queue):
        while True:
            frames = frame_queue.get()
            if frames is None:
                break

            ocr_results = []

            for frame in frames:
                results = self.model(frame)
                boxes = results[0].boxes if len(results) > 0 else []

                for box in boxes:
                    cls = int(box.cls)
                    if cls in self.plate_class_indices:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1, y1 = max(x1 - self.padding, 0), max(y1 - self.padding, 0)
                        x2, y2 = min(x2 + self.padding, self.width), min(y2 + self.padding, self.height)

                        plate_image = frame[y1:y2, x1:x2]
                        if plate_image.size == 0:
                            continue

                        preprocessed_img = ImageProcessor.preprocess_image(plate_image)
                        ocr_result = self.reader.readtext(preprocessed_img, detail=1)

                        for res in ocr_result:
                            text, confidence = res[1], res[2]
                            if confidence >= self.min_confidence:
                                text = ''.join(filter(str.isdigit, text))
                                if 2 <= len(text) <= 4:
                                    ocr_results.append(text)

            if ocr_results:
                most_common_text = Counter(ocr_results).most_common(1)[0][0]
                if most_common_text not in self.processed_numbers:
                    print(f"Detected text: {most_common_text}")

                    row = self.cur.query_number(most_common_text)

                    if row:
                        print(f"Matching number in DB: {row[0]}")
                        self.processed_numbers.add(most_common_text)
