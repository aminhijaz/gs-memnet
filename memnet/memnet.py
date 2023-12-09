from resmem import ResMem, transformer
from PIL import Image
import numpy as np
def score(solution):
    model = ResMem(pretrained=True)
    img = solution
    img = img.convert('RGB') 
    model.eval()
    image_x = transformer(img)
    prediction = model(image_x.view(-1, 3, 227, 227))
    print(prediction.item())
