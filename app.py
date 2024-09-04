from flask import Flask, request, jsonify
import base64
from flask_cors import CORS
from ptModels import predict_image


app = Flask(__name__)
CORS(app)

@app.route('/process_strings', methods=['POST'])
def process_strings():
    # Get the JSON data from the request
    data = request.json

    # Check if the request contains the 'string1' and 'string2' keys
    if 'string1' in data and 'string2' in data:
        string1 = data['string1']
        string2 = data['string2']

        # Perform some processing on the strings (example: concatenate them)
        result = string1 + " " + string2
        
        print(result)
        
        # Example usage
        base64_string=string1
        output_file = "output_image.png"  # Output file path

        base64_to_image(base64_string, output_file)

        # Run the prediction model on the saved image
        prediction = predict_image(output_file,string2)

        # Return the result as JSON
        return jsonify({ 'prediction': prediction}), 200
    else:
        # If the keys are missing, return an error response
        return jsonify({'error': 'Missing keys in JSON data'}), 400
    


def base64_to_image(base64_string, output_file):
    try:
        # Decode the base64 string into binary data
        image_data = base64.b64decode(base64_string)
        
        # Write the binary data to a file
        with open(output_file, 'wb') as f:
            f.write(image_data)
        
        print(f"Image saved as {output_file}")
    except Exception as e:
        print("Error:", e)




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
