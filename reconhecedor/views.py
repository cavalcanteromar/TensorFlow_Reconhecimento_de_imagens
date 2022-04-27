from django.shortcuts import render
from django.conf import settings
from .forms import UploadFileForm
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf


def index(request):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    context = {}
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            model = tf.keras.models.load_model(settings.BASE_DIR / 'my_model.h5')
            img = Image.open(request.FILES['file'])
            img = ImageOps.grayscale(img)
            output_size = (28, 28)
            img.thumbnail(output_size)
            img.save(settings.BASE_DIR / 'grayscale-thumbnail.jpg')
            im2array = np.array(img)
            im2array = (np.expand_dims(im2array, 0))
            print(request.FILES['file'], img, img.width, img.height, im2array.shape)
            predictions_single = model.predict(im2array)
            print(predictions_single)
            predicted = class_names[np.argmax(predictions_single[0])]
            context['predicted'] = predicted
            context['precision'] = max(predictions_single[0])
    else:
        form = UploadFileForm()
    context['form'] = form
    return render(request, 'reconhecedor/index.html', context)
