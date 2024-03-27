from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
# PDF

single_img_doc = DocumentFile.from_images("input2.png")
# Analyze
result = model(single_img_doc)
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

synthetic_pages = result.synthesize(font_family='arial.ttf', font_size=13)
plt.imshow(synthetic_pages[0]); plt.axis('off'); plt.show()