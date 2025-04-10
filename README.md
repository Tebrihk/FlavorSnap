# FlavorSnap
FlavorSnap is a food image classification web app powered by deep learning. Simply upload a picture of a local dish, and the model will tell you what it is. The app uses a fine-tuned ResNet18 model to recognize and classify various food types such as Akara, Bread, Egusi, Moi Moi, Rice and Stew, and Yam.

 Features

- Image Upload & Preview  
  Instantly preview your uploaded image in the sidebar.

- AI-Powered Classification  
  Classifies the image using a ResNet18 model trained on local food images.

- Automatic Image Organization  
  Uploaded images are saved into their predicted class folders.

- Simple Web UI with Panel  
  Intuitive interface built with Panel (based on Bokeh).

---

 Model Info

- Backbone: `ResNet18`
- Framework: `PyTorch`
- Custom classes:  
  - `Akara`
  - `Bread`
  - `Egusi`
  - `Moi Moi`
  - `Rice and Stew`
  - `Yam`

The model is saved at:  
