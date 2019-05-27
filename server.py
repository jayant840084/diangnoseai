from flask import Flask, request, render_template
app = Flask(__name__)

import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms

model = torch.load('./jayant_train_v3', map_location='cpu')
model.eval()

@app.route("/", methods=['GET', 'POST'])
def home():
  if request.method == 'GET':
    return render_template('home.html')
  elif request.method == 'POST':
    print(request.files['image'])
    image = Image.open(request.files['image'])
    if image.mode != 'RGB':
      image = image.convert("RGB")
    image = transforms.Resize(224)(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
    image = image.unsqueeze(0)

    probability=model(Variable(image)).cpu().detach().numpy()[0]

    return render_template('home.html', probability = {'yes': str(probability[0] * 100)[:5], 'no': str(probability[1] * 100)[:5]})
