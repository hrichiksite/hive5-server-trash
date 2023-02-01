from flask import Flask, request, jsonify, render_template
from infer import predict_image

app = Flask(__name__)

#load the catagory index
category_index = {1: {'id': 1, 'name': 'Flexibles'},
 2: {'id': 2, 'name': 'Bottle'},
 3: {'id': 3, 'name': 'Jar'},
 4: {'id': 4, 'name': 'Carton'},
 5: {'id': 5, 'name': 'Sachets-&-Pouch'},
 6: {'id': 6, 'name': 'Blister-pack'},
 7: {'id': 7, 'name': 'Tray'},
 8: {'id': 8, 'name': 'Tube'},
 9: {'id': 9, 'name': 'Can'},
 10: {'id': 10, 'name': 'Tub'},
 11: {'id': 11, 'name': 'Cosmetic'},
 12: {'id': 12, 'name': 'Box'},
 13: {'id': 13, 'name': 'Clothes'},
 14: {'id': 14, 'name': 'Bulb'},
 15: {'id': 15, 'name': 'Cup-&-glass'},
 16: {'id': 16, 'name': 'Book-&-magazine'},
 17: {'id': 17, 'name': 'Bag'},
 18: {'id': 18, 'name': 'Lid'},
 19: {'id': 19, 'name': 'Clamshell'},
 20: {'id': 20, 'name': 'Mirror'},
 21: {'id': 21, 'name': 'Tangler'},
 22: {'id': 22, 'name': 'Cutlery'},
 23: {'id': 23, 'name': 'Cassette-&-tape'},
 24: {'id': 24, 'name': 'Electronic-devices'},
 25: {'id': 25, 'name': 'Battery'},
 26: {'id': 26, 'name': 'Pen-&-pencil'},
 27: {'id': 27, 'name': 'Paper-products'},
 28: {'id': 28, 'name': 'Foot-wear'},
 29: {'id': 29, 'name': 'Scissor'},
 30: {'id': 30, 'name': 'Toys'},
 31: {'id': 31, 'name': 'Brush'},
 32: {'id': 32, 'name': 'Pipe'},
 33: {'id': 33, 'name': 'Foil'},
 34: {'id': 34, 'name': 'Hangers'}}

type_category_index = {
1: [23, 24, 25, 33],
2: [2, 3, 4, 5, 9, 12, 15, 17, 19, 10, 7, 6],
3: [32,8, 20, 22, 14, 1, 18],
4: [11, 13, 28, 21, 26, 27, 29, 30, 31, 34, 16]
}

@app.route("/")
def hello_world():
    return render_template("index.html", title="Hello")

@app.route("/infer", methods=["POST"])
def infer():
    print(request.files)
    # get uploaded image 
    f = request.files['file']
    f.save("a"+f.filename)   
    #now pass the image to the model and get the result
    result = predict_image("a"+f.filename)
    print(result)
    #now make the result json
    #return the first class detected
    #convert array to list
    
    
    detected_class = result['detection_classes'].tolist()
    print(detected_class)
    detected_class = detected_class[0][0]
    #check which type it belongs to
    segtype = 0
    for intype in type_category_index:
        if detected_class in type_category_index[intype]:
           segtype = intype
    return jsonify({'detected': segtype})


