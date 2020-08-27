from flask import Flask, render_template, request, redirect
from werkzeug import secure_filename
app = Flask(__name__)

#Messing Around With FLASK;
#Learning How It Works
#Learning How To Implement It In Order To Predict Any Image Selected
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]

def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


@app.route('/upload')
def upload_image():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == "POST":

        if request.files:

            image = request.files["file"]

            if image.filename == "":
                print("No filename")
                return redirect(request.url)

            if allowed_image(image.filename):
                filename = secure_filename(image.filename)

                return redirect(request.url)

            else:
                print("That file extension is not allowed")
                return redirect(request.url)
    return "Done"
		
if __name__ == '__main__':
   app.run(debug = True)