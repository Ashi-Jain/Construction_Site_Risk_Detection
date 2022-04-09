# streamlit-ml-app


## About Project

An application that facilitates farmers, scientists and botanists to detect the type of plant or crops, detect any kind of diseases in them. The app sends the image of the plant to the server where it is analysed using CNN classifier model. Once detected, the disease and its solutions are displayed to the user.

---

## Model

Trained to identify 5 classes for **Disease Detection** and 24 classes for **Disease Classification**

           - Disease Classification Classes

                       - Apple___Apple_scab
                       - Apple___Black_rot
			   - Apple___Cedar_apple_rust
			   - Apple___healthy
			   - Blueberry___healthy
			   - Cherry___healthy
			   - Cherry___Powdery_mildew
			   - Grape___Black_rot
			   - Grape___Esca_Black_Measles
			   - Grape___healthy
			   - Grape___Leaf_blight_Isariopsis_Leaf_Spot
			   - Orange___Haunglongbing
			   - Peach___Bacterial_spot
			   - Peach___healthy
			   - Pepper,_bell___Bacterial_spot
			   - Pepper,_bell___healthy
			   - Potato___Early_blight
			   - Potato___healthy
			   - Raspberry___healthy
			   - Soybean___healthy
			   - Squash___Powdery_mildew
			   - Strawberry___healthy
			   - Strawberry___Leaf_scorch
			
            - Disease Detection Classes
            
			   - Cherry___healthy
			   - Cherry___Powdery_mildew
			   - Grape___Black_rot
			   - Grape___Esca_Black_Measles
			   - Grape___healthy
			   - Grape___Leaf_blight_Isariopsis_Leaf_Spot 
---


## Usage 
 
 1. Install the required dependencies 
 ```
	pip install -r requirements.txt 
```
2. Command for running app 

```
	streamlit run app.py
```


## Images

![alt text](./images/1.PNG  "About")
![alt text](./images/2.PNG  "Disease Detection")
![alt text](./images/3.PNG  "Image Upload")
![alt text](./images/4.PNG  "Image Output")
![alt text](./images/5.PNG  "Disease Classification")
![alt text](./images/6.PNG  "Image Output")
![alt text](./images/7.PNG  "Treatment information")
![alt text](./images/8.PNG  "Treatment information")

---

## Enviornments

1. This app is deployed on [Heroku](https://bot-beats-ml-app.herokuapp.com/)

 **Note: The tensorflow model load into the memory and hence can be slow on heroku as compared to the local enviornment**

## Developer

### Hi there, I'm Ameya Upalanchi 

#### I'm a Project Enthusiast, Developer, and Life long learner!

-  I am currently learning everything 
-  2020 Goals: Contribute more to Open Source projects
-  Fun fact: I love to play table tennis and guitar / drums

### Connect with me:

[<img align="left" alt="codeSTACKr.com" width="22px" src="https://raw.githubusercontent.com/iconic/open-iconic/master/svg/globe.svg" />][website]
[<img align="left" alt="codeSTACKr | LinkedIn" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" />][linkedin]
[<img align="left" alt="codeSTACKr | Instagram" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/instagram.svg" />][instagram]
<br />
<br />

[website]:http://ameyaupalanchi.tk/
[instagram]:https://www.instagram.com/ameya_uplanchi/
[linkedin]: https://in.linkedin.com/in/ameya-upalanchi-a9a883191


---

