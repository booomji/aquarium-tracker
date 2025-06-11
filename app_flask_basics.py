#TUTORIAL Python Flask tutorial for Beginners by Gogetmyguru https://www.youtube.com/playlist?list=PLf9umJdQ546h26s7VKQVUir5GoOZ-1JTP 
from flask import Flask # Import the Flask class from the flask module

# Create an instance of the Flask class
app = Flask(__name__) # __name__ is a special variable in Python that represents the name of the module.It goes to the Flask constructor to determine the root path of the application.
#This is important for locating resources such as templates and static files. The Flask instance is the WSGI application.
# The Flask instance is the WSGI application.

# Which url to use to access the application 
@app.route('/')
def hello_world():
    return 'Hello, World!'  

# Another route . You can define multiple routes in a Flask application.
# Each route is associated with a function that returns a response.
# The function is executed when the route is accessed. You can access the route by going to http://localhost:5000/hello in your web browser.
# the names that follow after a '/' are also called endpoints. e.g /hello

@app.route('/hello')
def hello():
    return 'Hello, Flask at another endpoint!'

def hello_alt():
    return 'Alternate way to call hello function!'

app.add_url_rule('/hello_alt', 'hello_alt', hello_alt) # This line adds a URL rule for the '/hello' endpoint, associating it with the hello function.


print("Flask application root page http://localhost:5000")
print("Flask application hello page http://localhost:5000/hello") 
print("Flask application hello_alt page http://localhost:5000/hello_alt") 


if __name__ == '__main__':
    app.run(debug=True)
# This is a simple Flask application that defines two routes: '/' and '/hello'.