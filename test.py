#make a post request to the server 


#make a post request to the server 

import requests

url = "http://127.0.0.1:5000/infer"

files = {'file': open('75ml.jpg', 'rb')}

r = requests.post(url, files=files)

print(r.text)