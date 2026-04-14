# MelanomaClassifier-MobileApp

# Authors
* Jakub Padło
* Paweł Palcar

# Pierwsze uruchomienie
```sh
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

# Każde kolejne 
```sh
venv\Scripts\activate
```

# Trening odbywa się na Collabie, ponieważ korzystamy wtedy z GPU
### Repo jest prywatne, więc najpierw na GH trzeba wygenerować PAT i wkleić go do sekretów w Collabie
### Następnie do komórki w collabie wpisujemy
```
from google.colab import userdata
token = userdata.get('GITHUB_TOKEN')

!git clone https://{token}@github.com/kubapadlo/MelanomaClassifier-MobileApp.git
%cd MelanomaClassifier-MobileApp
!python main.py
```